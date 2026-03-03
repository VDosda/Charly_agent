from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.config.settings import Settings
from agent.providers.llm.base import LLMProvider, LLMResult, ToolCall, ToolSpec
from agent.providers.embeddings.base import EmbeddingProvider
from agent.skills.base import ToolContext, ToolExecutionError
from agent.skills.registry import ToolRegistry


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

    def handle_message(self, user_id: str, session_id: str, message: str) -> str:
        """
        Main entrypoint:
        - Persist user message
        - Build context
        - Call LLM
        - If tool calls: execute, persist tool results, call LLM again
        - Persist assistant final answer
        - Run MT/LT hooks + cleanup
        """
        # 1) Persist user message (ST)
        self._persist_turn(user_id, session_id, role="user", content=message)

        # 2) Build messages for LLM
        messages = self._build_llm_messages(user_id, session_id)

        # 3) Tool specs for this run
        tool_specs = self.skills.list_specs() if self.llm.supports_tools() else None

        # 4) Tool-calling loop
        final_text = ""
        tool_iterations = 0
        last_tool_msgs: List[Dict[str, Any]] = []

        while True:
            result = self.llm.generate(
                messages=messages,
                tools=tool_specs,
                tool_choice="auto",
            )

            # If model produced text, keep it (might be partial if tool calls follow)
            if result.text:
                final_text = result.text

            # No tool calls -> finalize
            if not result.tool_calls:
                if self._uses_flat_tool_transcript() and not final_text and last_tool_msgs:
                    final_text = self._flat_tool_fallback_text(last_tool_msgs)
                break

            tool_iterations += 1
            if tool_iterations > self.limits.max_tool_iterations:
                final_text = (
                    "ERROR: tool loop exceeded max iterations. "
                    "Refusing to continue for safety."
                )
                break

            # Execute each tool call and append tool results to the messages
            tool_msgs = self._execute_tool_calls(
                user_id=user_id,
                session_id=session_id,
                tool_calls=result.tool_calls,
            )
            last_tool_msgs = tool_msgs

            # Append the assistant message that triggered tool calls
            # (Some providers return empty text with tool_calls; still add a marker)
            if self._uses_flat_tool_transcript():
                # Ollama chat parsing is stricter on role/fields; keep only assistant text.
                messages.append(
                    {
                        "role": "assistant",
                        "content": result.text or "",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": result.text or "",
                        "tool_calls": [self._tool_call_to_provider_shape(tc) for tc in result.tool_calls],
                    }
                )

            # Append tool results to messages.
            if self._uses_flat_tool_transcript():
                for tm in tool_msgs:
                    tool_name = tm.get("name") or "tool"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"[TOOL RESULT] {tool_name} => {tm.get('content', '')}",
                        }
                    )
                # Ensure next generation is a response turn, not a continuation of assistant logs.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Use only the tool results above and answer "
                            "the latest user request in one concise sentence."
                        ),
                    }
                )
            else:
                # Provider-agnostic "tool" role (OpenAI-compatible)
                messages.extend(tool_msgs)

        # 5) Persist final assistant message
        self._persist_turn(user_id, session_id, role="assistant", content=final_text)

        # 6) MT/LT hooks (safe no-ops until implemented)
        self._maybe_create_episode(user_id, session_id)
        self._maybe_distill_profile(user_id, session_id)

        # 7) Cleanup (TTL, compression purge policies, etc.)
        self._cleanup(user_id, session_id)

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
        user_id: str,
        session_id: str,
        tool_calls: Sequence[ToolCall],
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls, persist tool results to ST, and return tool messages.
        """
        tool_messages: List[Dict[str, Any]] = []

        ctx = ToolContext(
            user_id=user_id,
            session_id=session_id,
            metadata={
                "workspace": self.settings.workspace,
                "env": self.settings.env,
            },
        )

        for tc in tool_calls:
            name = tc.name
            args = tc.arguments or {}

            try:
                result = self.skills.execute(name=name, args=args, context=ctx)
                payload = {"ok": True, "result": result}
            except ToolExecutionError as e:
                payload = {"ok": False, "error": str(e)}
            except Exception as e:
                # Never raise raw exceptions to the LLM; normalize them
                payload = {"ok": False, "error": f"Unhandled tool error: {type(e).__name__}: {e}"}

            # Persist tool output as a ST "tool" turn
            tool_text = json.dumps(payload, ensure_ascii=False)
            self._persist_turn(
                user_id=user_id,
                session_id=session_id,
                role="tool",
                content=tool_text,
                tool_name=name,
                tool_args_json=json.dumps(args, ensure_ascii=False),
            )

            # Add tool message back to LLM conversation
            tool_messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": tool_text,
                }
            )

        return tool_messages

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

    def _maybe_create_episode(self, user_id: str, session_id: str) -> None:
        """
        Hook: create a medium-term episode summary every N turns and mark ST turns compressed.

        Implement once you have:
        - episode_memory table
        - compressed flag in chat_history (optional)
        - distillation prompt to summarize a range of turns

        For now: no-op.
        """
        return

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
