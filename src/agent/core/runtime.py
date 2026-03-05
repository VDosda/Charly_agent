from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from agent.config.settings import Settings
from agent.memory.retrieve_smart import (
    retrieve_lt_smart,
    retrieve_mt_smart,
)
from agent.memory.cleanup import cleanup_st
from agent.memory.context import build_memory_context_blocks
from agent.memory.models import MemoryBundle
from agent.memory.scoring import rerank_items, rerank_episodes
from agent.providers.llm.base import LLMProvider, ToolCall
from agent.providers.embeddings.base import EmbeddingProvider
from agent.skills.base import ToolContext
from agent.skills.registry import ToolRegistry
from agent.core.tracing import JSONTracer, new_correlation_id
from agent.core.tool_runtime import ToolRuntime
from agent.core.planner import Planner, ToolPolicy
from agent.memory.distill_mt import maybe_create_episode
from agent.memory.distill_lt import LTConfig, maybe_distill_profile_from_st_window, retry_pending_lt_embeddings


@dataclass(frozen=True)
class MemoryJob:
    correlation_id: str
    user_id: str
    session_id: str
    user_turn_id: int
    assistant_turn_id: int
    enqueued_at_ms: int
    enqueued_perf_ts: float



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
        self.tracer = JSONTracer(enabled=settings.trace_stdout)
        self.tool_runtime = ToolRuntime(
            registry=self.skills,
            tracer=self.tracer,
            default_timeout_s=15.0,
            max_workers=4,
        )
        self.planner = Planner(policy=ToolPolicy.from_settings(settings))
        self.lt_cfg = LTConfig(
            min_importance=float(self.settings.memory.lt_importance_threshold),
            min_confidence=float(self.settings.memory.lt_confidence_threshold),
        )
        self._memory_loop = asyncio.new_event_loop()
        self._memory_queue: Optional[asyncio.Queue[Optional[MemoryJob]]] = None
        self._memory_session_locks: Dict[tuple[str, str], asyncio.Lock] = {}
        self._memory_worker_ready = threading.Event()
        self._memory_worker_thread = threading.Thread(
            target=self._memory_worker_main,
            name="agent-memory-worker",
            daemon=True,
        )
        self._memory_worker_thread.start()
        if not self._memory_worker_ready.wait(timeout=2.0):
            self.tracer.emit(
                event="memory.worker.init",
                level="warning",
                correlation_id="runtime-init",
                payload={"status": "not_ready"},
            )

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
                enqueue memory job (background)
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
        user_turn_id = self._persist_turn(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=message,
        )

        # 2 Build context for LLM
        messages = self._build_llm_messages(user_id, session_id)
        bundle = self._build_memory_bundle(
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            user_message=message,
        )
        mem_blocks = build_memory_context_blocks(bundle)
        messages[1:1] = mem_blocks
        lt_chars, mt_chars = self._memory_block_sizes(mem_blocks)
        self.tracer.emit(
            event="memory.inject",
            level="debug",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "lt_chars": lt_chars,
                "mt_chars": mt_chars,
                "lt_items": len(bundle.lt_items),
                "mt_episodes": len(bundle.mt_episodes),
            },
        )
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
        llm_main_start = time.perf_counter()

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

        llm_main_latency_ms = int((time.perf_counter() - llm_main_start) * 1000)
        self.tracer.emit(
            event="llm.main.latency",
            level="debug",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={"llm_main_latency": llm_main_latency_ms},
        )

        # 5 Persist assistant answer
        assistant_turn_id = self._persist_turn(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=final_text,
        )

        memory_enqueued = self._enqueue_memory_job(
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            user_turn_id=user_turn_id,
            assistant_turn_id=assistant_turn_id,
        )

        # 6 Tracing end (memory continues in background worker)
        self.tracer.emit(
            event="request.end",
            level="info",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "response_len": len(final_text),
                "llm_main_latency": llm_main_latency_ms,
                "memory_job_enqueued": bool(memory_enqueued),
            },
        )

        return final_text


    # ---------------------------------------------------------------------
    # Prompt / Context building
    # ---------------------------------------------------------------------

    def _build_llm_messages(
        self,
        user_id: str,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Build the LLM message list from:
        - System instructions (stable)
        - Recent short-term turns (ST)
        """
        system_text = self._system_prompt()

        st_turns = self._read_recent_turns(
            session_id=session_id,
            limit=self.limits.max_history_turns,
        )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_text}]

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
    ) -> int:
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
        return next_turn_id

    def _read_recent_turns(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """
        Read the last N turns for this session, ordered oldest->newest.
        """
        try:
            rows = self.db.execute(
                """
                SELECT role, content, tool_name, tool_args_json, tool_result_json
                FROM chat_history
                WHERE session_id = ?
                  AND archived = 0
                ORDER BY turn_id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Backward-compat fallback when archived column is not present yet.
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
    # MT/LT hooks (executed in background memory worker)
    # ---------------------------------------------------------------------

    def _maybe_distill_lt_from_st(self, correlation_id: str, user_id: str, session_id: str) -> int:
        return maybe_distill_profile_from_st_window(
            db=self.db,
            llm=self.llm,
            embeddings=self.embeddings,
            tracer=self.tracer,
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            cfg=self.lt_cfg,
        )

    def _maybe_create_episode(self, correlation_id: str, user_id: str, session_id: str) -> Dict[str, Any]:
        episode_id = maybe_create_episode(
            db=self.db,
            llm=self.llm,
            embeddings=self.embeddings,
            tracer=self.tracer,
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
        )
        retry_stats = retry_pending_lt_embeddings(
            db=self.db,
            embeddings=self.embeddings,
            tracer=self.tracer,
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            limit=6,
            force=False,
        )
        return {
            "episode_id": episode_id,
            "embedding_retry_count": int((retry_stats or {}).get("processed", 0)),
            "embedding_retry_succeeded": int((retry_stats or {}).get("succeeded", 0)),
        }

    def _cleanup(self, user_id: str, session_id: str) -> None:
        cleanup_st(self.db, session_id=session_id)

    def _build_memory_bundle(
        self,
        *,
        correlation_id: str,
        user_id: str,
        session_id: str,
        user_message: str,
    ) -> MemoryBundle:
        items_with_dist = []
        eps_with_dist = []

        try:
            items_with_dist = retrieve_lt_smart(
                self.db,
                self.embeddings,
                user_id=user_id,
                user_message=user_message,
                limit=6,
            )
        except Exception as e:
            self.tracer.emit(
                event="memory.retrieve.lt.error",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"error": f"{type(e).__name__}: {e}"},
            )

        try:
            eps_with_dist = retrieve_mt_smart(
                self.db,
                self.embeddings,
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                limit=3,
            )
        except Exception as e:
            self.tracer.emit(
                event="memory.retrieve.mt.error",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"error": f"{type(e).__name__}: {e}"},
            )

        return MemoryBundle(
            lt_items=rerank_items(items_with_dist, limit=6),
            mt_episodes=rerank_episodes(eps_with_dist, limit=3),
        )

    def _memory_block_sizes(self, mem_blocks: Sequence[Dict[str, str]]) -> tuple[int, int]:
        lt_chars = 0
        mt_chars = 0

        for block in mem_blocks:
            content = block.get("content", "")
            if content.startswith("LONG-TERM MEMORY"):
                lt_chars = len(content)
            elif content.startswith("MEDIUM-TERM EPISODES"):
                mt_chars = len(content)

        return lt_chars, mt_chars

    # ---------------------------------------------------------------------
    # Background memory worker
    # ---------------------------------------------------------------------

    def _memory_worker_main(self) -> None:
        asyncio.set_event_loop(self._memory_loop)
        self._memory_queue = asyncio.Queue()
        self._memory_loop.create_task(self._memory_worker_loop())
        self._memory_worker_ready.set()
        self.tracer.emit(
            event="memory.worker.init",
            level="info",
            correlation_id="runtime-init",
            payload={"status": "ready"},
        )
        self._memory_loop.run_forever()

    async def _memory_worker_loop(self) -> None:
        if self._memory_queue is None:
            return

        while True:
            job = await self._memory_queue.get()
            if job is None:
                self._memory_queue.task_done()
                break

            queue_delay_ms = int((time.perf_counter() - job.enqueued_perf_ts) * 1000)
            self.tracer.emit(
                event="memory.job.start",
                level="debug",
                correlation_id=job.correlation_id,
                user_id=job.user_id,
                session_id=job.session_id,
                payload={
                    "memory_job_queue_delay": queue_delay_ms,
                    "user_turn_id": job.user_turn_id,
                    "assistant_turn_id": job.assistant_turn_id,
                },
            )

            started = time.perf_counter()
            lt_items_written = 0
            mt_episode_created = False
            embedding_retry_count = 0

            lock = self._memory_session_lock(job.user_id, job.session_id)
            async with lock:
                try:
                    lt_items_written = int(
                        self._maybe_distill_lt_from_st(
                            correlation_id=job.correlation_id,
                            user_id=job.user_id,
                            session_id=job.session_id,
                        )
                    )
                except Exception as e:
                    self.tracer.emit(
                        event="memory.job.step.error",
                        level="warning",
                        correlation_id=job.correlation_id,
                        user_id=job.user_id,
                        session_id=job.session_id,
                        payload={"step": "lt_distill", "error": f"{type(e).__name__}: {e}"},
                    )

                try:
                    mt_stats = self._maybe_create_episode(
                        correlation_id=job.correlation_id,
                        user_id=job.user_id,
                        session_id=job.session_id,
                    )
                    mt_episode_created = bool((mt_stats or {}).get("episode_id"))
                    embedding_retry_count = int((mt_stats or {}).get("embedding_retry_count", 0))
                except Exception as e:
                    self.tracer.emit(
                        event="memory.job.step.error",
                        level="warning",
                        correlation_id=job.correlation_id,
                        user_id=job.user_id,
                        session_id=job.session_id,
                        payload={"step": "mt_distill_and_retry", "error": f"{type(e).__name__}: {e}"},
                    )

                try:
                    self._cleanup(job.user_id, job.session_id)
                except Exception as e:
                    self.tracer.emit(
                        event="memory.job.step.error",
                        level="warning",
                        correlation_id=job.correlation_id,
                        user_id=job.user_id,
                        session_id=job.session_id,
                        payload={"step": "st_cleanup", "error": f"{type(e).__name__}: {e}"},
                    )

            memory_job_duration_ms = int((time.perf_counter() - started) * 1000)
            self.tracer.emit(
                event="memory.job.end",
                level="info",
                correlation_id=job.correlation_id,
                user_id=job.user_id,
                session_id=job.session_id,
                payload={
                    "memory_job_queue_delay": queue_delay_ms,
                    "memory_job_duration": memory_job_duration_ms,
                    "lt_items_written": int(lt_items_written),
                    "mt_episode_created": bool(mt_episode_created),
                    "embedding_retry_count": int(embedding_retry_count),
                },
            )
            self._memory_queue.task_done()

    def _memory_session_lock(self, user_id: str, session_id: str) -> asyncio.Lock:
        key = (user_id, session_id)
        lock = self._memory_session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._memory_session_locks[key] = lock
        return lock

    def _enqueue_memory_job(
        self,
        *,
        correlation_id: str,
        user_id: str,
        session_id: str,
        user_turn_id: int,
        assistant_turn_id: int,
    ) -> bool:
        if not self._memory_worker_ready.is_set() or self._memory_queue is None:
            self.tracer.emit(
                event="memory.job.enqueue",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"status": "worker_not_ready"},
            )
            return False

        job = MemoryJob(
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            user_turn_id=int(user_turn_id),
            assistant_turn_id=int(assistant_turn_id),
            enqueued_at_ms=int(time.time() * 1000),
            enqueued_perf_ts=time.perf_counter(),
        )
        self._memory_loop.call_soon_threadsafe(self._memory_queue.put_nowait, job)
        self.tracer.emit(
            event="memory.job.enqueue",
            level="debug",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "status": "queued",
                "user_turn_id": int(user_turn_id),
                "assistant_turn_id": int(assistant_turn_id),
            },
        )
        return True

    def wait_memory_idle(self, timeout_s: float = 5.0) -> bool:
        if not self._memory_worker_ready.is_set() or self._memory_queue is None:
            return True
        fut = asyncio.run_coroutine_threadsafe(self._memory_queue.join(), self._memory_loop)
        try:
            fut.result(timeout=float(timeout_s))
            return True
        except Exception:
            return False
