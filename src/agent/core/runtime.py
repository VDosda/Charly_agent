from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from agent.config.settings import Settings
from agent.providers.llm.base import LLMProvider, ToolCall
from agent.providers.embeddings.base import EmbeddingProvider
from agent.skills.base import ToolContext
from agent.skills.registry import ToolRegistry
from agent.core.tracing import JSONTracer, new_correlation_id
from agent.core.tool_runtime import ToolRuntime
from agent.core.planner import Planner, ToolPolicy
from agent.memory.distill_mt import maybe_create_episode


@dataclass
class RuntimeLimits:
    """
    Safety limits to keep the agent predictable in production.
    """
    max_tool_iterations: int = 6
    max_history_turns: int = 30  # ST turns injected into prompt


class AgentRuntime:
    """
    Core runtime that orchestrates:
    - Persisting messages (ST)
    - Building the prompt context
    - Calling LLM
    - Tool-calling loop
    - Persisting tool results and final assistant output
    - Hooks for MT/LT distillation + cleanup
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        llm: LLMProvider,
        embeddings: EmbeddingProvider,
        skills: ToolRegistry,
        settings: Settings,
        limits: Optional[RuntimeLimits] = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.embeddings = embeddings
        self.skills = skills
        self.settings = settings
        self.limits = limits or RuntimeLimits(
            max_tool_iterations=6,
            max_history_turns=settings.memory.st_active_turns,
        )
        self.tracer = JSONTracer(enabled=True)
        self.tool_runtime = ToolRuntime(
            registry=self.skills,
            tracer=self.tracer,
            default_timeout_s=15.0,
            max_workers=4,
        )
        self.planner = Planner(policy=ToolPolicy.from_settings(settings))

    def handle_message(self, user_id: str, session_id: str, message: str) -> str:
        """
            Main entrypoint of the agent runtime.

            Flow:
                user message
                    ↓
                persist ST
                    ↓
                planner policy
                    ↓
                LLM call
                    ↓
                tool loop
                    ↓
                persist assistant answer
                    ↓
                MT/LT hooks
            """

        correlation_id = new_correlation_id()

        self.tracer.emit(
            event="request.start",
            level="info",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={"message": message},
        )

        # 1 Persist user message (ST)
        self._persist_turn(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=message,
        )

        # 2 Build context for LLM
        messages = self._build_llm_messages(user_id, session_id)
        policy_state = self._policy_state_for_session(user_id=user_id, session_id=session_id)

        available_tools = self.skills.list_tools() if self.llm.supports_tools() else None
        tool_specs = None

        # 3 Planner policy
        decision = self.planner.decide(
            user_message=message,
            available_tools=available_tools,
            session_state=policy_state,
        )

        if decision.blocked_response:
            self.tracer.emit(
                event="planner.block",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={
                    "reason": decision.blocked_response,
                    "tool_choice": decision.tool_choice,
                    "user_message": message,
                },
            )
            return decision.blocked_response

        if decision.system_overrides:
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": decision.system_overrides,
                },
            )

        if available_tools:
            before = len(available_tools)
            allowed_tools = self.planner.filter_tools(available_tools, session_state=policy_state)
            tool_specs = [t.spec for t in allowed_tools]
            after = len(allowed_tools)
            self.tracer.emit(
                event="planner.tools_filtered",
                level="debug",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"before": before, "after": after},
            )

        tool_choice = decision.tool_choice
        if not tool_specs:
            tool_choice = "none"

        # 4 Tool execution loop
        final_text = ""
        tool_iterations = 0

        while True:

            self.tracer.emit(
                event="llm.request",
                level="debug",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={
                    "messages": len(messages),
                    "tool_choice": tool_choice,
                },
            )

            result = self.llm.generate(
                messages=messages,
                tools=tool_specs,
                tool_choice=tool_choice,
            )

            if result.text:
                final_text = result.text

            self.tracer.emit(
                event="llm.response",
                level="debug",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={
                    "tool_calls": len(result.tool_calls or []),
                    "text_len": len(result.text or ""),
                },
            )

            # If no tools requested -> exit loop
            if not result.tool_calls:
                break

            tool_iterations += 1

            if tool_iterations > self.limits.max_tool_iterations:

                final_text = (
                    "ERROR: tool loop exceeded max iterations. "
                    "Stopping execution for safety."
                )

                break

            # Execute tools via ToolRuntime
            tool_msgs = self._execute_tool_calls(
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                tool_calls=result.tool_calls,
                session_state=policy_state,
            )

            # Append assistant message that triggered tools
            messages.append(
                {
                    "role": "assistant",
                    "content": result.text or "",
                }
            )

            # Append tool feedback messages
            messages.extend(tool_msgs)

        # 5 Persist assistant answer
        self._persist_turn(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=final_text,
        )

        # 6 Memory hooks
        self._maybe_create_episode(
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
        )

        self._maybe_distill_profile(user_id, session_id)


        # 7 Cleanup
        self._cleanup(user_id, session_id)

        # 8 Tracing end
        self.tracer.emit(
            event="request.end",
            level="info",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={"response_len": len(final_text)},
        )

        return final_text


    # ---------------------------------------------------------------------
    # Prompt / Context building
    # ---------------------------------------------------------------------

    def _build_llm_messages(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Build the LLM message list from:
        - System instructions (stable)
        - Long-term memories (LT) [hook placeholder]
        - Medium-term episodes (MT) [hook placeholder]
        - Recent short-term turns (ST)

        For now, we implement ST-only injection + placeholders for MT/LT.
        """
        system_text = self._system_prompt()

        st_turns = self._read_recent_turns(
            session_id=session_id,
            limit=self.limits.max_history_turns,
        )

        # Placeholders: in later phases, you will retrieve MT/LT here.
        lt_block = self._retrieve_lt_block(user_id, session_id)  # can be ""
        mt_block = self._retrieve_mt_block(user_id, session_id)  # can be ""

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_text}]

        # Inject LT/MT as a compact system-ish memory block (keeps it high priority)
        mem_parts = []
        if lt_block:
            mem_parts.append("LONG-TERM MEMORY:\n" + lt_block)
        if mt_block:
            mem_parts.append("MEDIUM-TERM EPISODES:\n" + mt_block)

        if mem_parts:
            messages.append({"role": "system", "content": "\n\n".join(mem_parts)})

        # Add ST turns
        for t in st_turns:
            role = t["role"]
            content = t["content"]
            # Tool outputs are stored in ST as role='tool'.
            if role == "tool":
                if self._uses_flat_tool_transcript():
                    tool_name = t.get("tool_name") or "tool"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"[TOOL LOG] {tool_name}: {content}",
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "name": t.get("tool_name") or "tool",
                            "content": t["content"],
                        }
                    )
            else:
                messages.append({"role": role, "content": content})

        return messages

    def _system_prompt(self) -> str:
        """
        Keep system prompt stable and minimal.
        You can make it configurable later.
        """
        return (
            "You are an agentic assistant. "
            "Be precise and robust. "
            "Use tools when needed. "
            "If uncertain, state what is missing and propose how to verify."
        )

    def _uses_flat_tool_transcript(self) -> bool:
        """
        Ollama's /api/chat can reject OpenAI-style tool transcript fields.
        For Ollama, flatten tool state into assistant text-only logs.
        """
        return self.llm.__class__.__module__.endswith(".providers.llm.ollama")

    def _flat_tool_fallback_text(self, tool_msgs: Sequence[Dict[str, Any]]) -> str:
        """
        Fallback text when Ollama returns no assistant content after a tool run.
        """
        lines: List[str] = []
        for tm in tool_msgs:
            name = tm.get("name") or "tool"
            content = tm.get("content") or ""
            lines.append(f"[TOOL RESULT] {name} => {content}")
        return "\n".join(lines) if lines else "No tool result available."

    # ---------------------------------------------------------------------
    # Tools execution
    # ---------------------------------------------------------------------

    def _execute_tool_calls(
        self,
        correlation_id: str,
        user_id: str,
        session_id: str,
        tool_calls: Sequence[ToolCall],
        session_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls via ToolRuntime (timeouts + tracing),
        persist tool results to ST, and return messages to feed back into the LLM.

        IMPORTANT:
        For Ollama compatibility, we do NOT return role="tool" messages.
        We return plain text as role="assistant".
        """
        ctx = ToolContext(
            user_id=user_id,
            session_id=session_id,
            metadata={
                "workspace": self.settings.workspace,
                "env": self.settings.env,
            },
        )

        rendered_lines: List[str] = []
        session_state = session_state or {}

        for tc in tool_calls:
            name = tc.name
            args = tc.arguments or {}

            # Get tool spec (for validation) + execute through ToolRuntime (timeout + tracing)
            try:
                tool = self.skills.get(name)  # Tool(spec=ToolSpec, handler=...)
            except KeyError:
                payload = {"ok": False, "error": f"Tool not found: {name}"}
                tool_text = json.dumps(payload, ensure_ascii=False)

                # Persist as tool turn
                self._persist_turn(
                    user_id=user_id,
                    session_id=session_id,
                    role="tool",
                    content=tool_text,
                    tool_name=name,
                    tool_args_json=json.dumps(args, ensure_ascii=False),
                )

                rendered_lines.append(f"[TOOL RESULT] {name} args={args} => {tool_text}")
                continue

            # Defense in depth: enforce policy at execution time too.
            if not self.planner.is_tool_allowed(tool, session_state=session_state):
                payload = {"ok": False, "error": f"Tool blocked by policy: {name}"}
                tool_text = json.dumps(payload, ensure_ascii=False)

                self._persist_turn(
                    user_id=user_id,
                    session_id=session_id,
                    role="tool",
                    content=tool_text,
                    tool_name=name,
                    tool_args_json=json.dumps(args, ensure_ascii=False),
                )

                rendered_lines.append(f"[TOOL RESULT] {name} args={args} => {tool_text}")
                continue

            if self.planner.requires_confirmation(tool, session_state=session_state):
                payload = {
                    "ok": False,
                    "error": (
                        f"Tool '{name}' requires explicit confirmation before execution."
                    ),
                }
                tool_text = json.dumps(payload, ensure_ascii=False)

                self._persist_turn(
                    user_id=user_id,
                    session_id=session_id,
                    role="tool",
                    content=tool_text,
                    tool_name=name,
                    tool_args_json=json.dumps(args, ensure_ascii=False),
                )

                rendered_lines.append(f"[TOOL RESULT] {name} args={args} => {tool_text}")
                continue

            run = self.tool_runtime.run(
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                tool_spec=tool.spec,
                args=args,
                context=ctx,
                timeout_s=self.planner.timeout_for(tool, session_state=session_state),
            )

            if run.ok:
                payload = {"ok": True, "result": run.result}
            else:
                payload = {"ok": False, "error": run.error, "error_type": run.error_type}

            tool_text = json.dumps(payload, ensure_ascii=False)

            # Persist tool output as a ST "tool" turn
            self._persist_turn(
                user_id=user_id,
                session_id=session_id,
                role="tool",
                content=tool_text,
                tool_name=name,
                tool_args_json=json.dumps(args, ensure_ascii=False),
                tool_result_json=json.dumps(run.result, ensure_ascii=False) if run.ok else None,
            )

            # Feed tool result back to the model as plain text (works for Ollama & others)
            rendered_lines.append(f"[TOOL RESULT] {name} args={args} => {tool_text}")

        tool_feedback = "\n".join(rendered_lines)

        # Return as assistant text instead of role=tool (Ollama-safe)
        return [{"role": "assistant", "content": tool_feedback}]

    def _policy_state_for_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Extension point for per-user/session policy overrides.
        Example supported keys:
        - deny_scopes
        - deny_tools
        - deny_tags
        - deny_risk
        - allow_tags
        - confirmed_tools
        - tool_timeouts_s
        """
        return {}


    def _tool_call_to_provider_shape(self, tc: ToolCall) -> Dict[str, Any]:
        """
        Create a generic tool_call shape. Some providers ignore it.
        Keeping this helps providers that expect tool_call IDs.
        """
        return {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
            },
        }

    # ---------------------------------------------------------------------
    # Short-term persistence (ST)
    # ---------------------------------------------------------------------

    def _persist_turn(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_args_json: Optional[str] = None,
        tool_result_json: Optional[str] = None,
    ) -> None:
        """
        Insert one turn into chat_history.

        IMPORTANT:
        - Uses a transaction to safely compute next turn_id per session.
        - Assumes chat_history schema exists.
        """
        ts = int(time.time())

        with self.db:  # transaction
            row = self.db.execute(
                "SELECT COALESCE(MAX(turn_id), 0) + 1 FROM chat_history WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_turn_id = int(row[0])

            self.db.execute(
                """
                INSERT INTO chat_history (
                    user_id, session_id, turn_id, ts, role, content,
                    tool_name, tool_args_json, tool_result_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    next_turn_id,
                    ts,
                    role,
                    content,
                    tool_name,
                    tool_args_json,
                    tool_result_json,
                ),
            )

    def _read_recent_turns(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """
        Read the last N turns for this session, ordered oldest->newest.
        """
        rows = self.db.execute(
            """
            SELECT role, content, tool_name, tool_args_json, tool_result_json
            FROM chat_history
            WHERE session_id = ?
            ORDER BY turn_id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

        # Reverse to get chronological order
        out = []
        for r in reversed(rows):
            out.append(
                {
                    "role": r["role"],
                    "content": r["content"],
                    "tool_name": r["tool_name"],
                    "tool_args_json": r["tool_args_json"],
                    "tool_result_json": r["tool_result_json"],
                }
            )
        return out

    # ---------------------------------------------------------------------
    # MT/LT hooks (placeholders for next steps)
    # ---------------------------------------------------------------------

    def _maybe_create_episode(
        self,
        correlation_id: str,
        user_id: str,
        session_id: str,
    ) -> None:
        maybe_create_episode(
            db=self.db,
            llm=self.llm,
            embeddings=self.embeddings,
            tracer=self.tracer,
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
        )

    def _maybe_distill_profile(self, user_id: str, session_id: str) -> None:
        """
        Hook: extract stable LT memories (preferences/facts/procedures) and upsert.

        Implement once you have:
        - profile_memory table
        - upsert logic (fingerprint, supersede by key)
        - retrieval logic (fts + vec) for future context building

        For now: no-op.
        """
        return

    def _cleanup(self, user_id: str, session_id: str) -> None:
        """
        Hook: cleanup policies (MT TTL purge, ST archive/purge, etc.).
        For now: no-op.
        """
        return

    def _retrieve_lt_block(self, user_id: str, session_id: str) -> str:
        """
        Hook: retrieve relevant long-term memory as a compact string.
        For now: empty (no LT yet).
        """
        return ""

    def _retrieve_mt_block(self, user_id: str, session_id: str) -> str:
        """
        Hook: retrieve relevant episode summaries as a compact string.
        For now: empty (no MT yet).
        """
        return ""
