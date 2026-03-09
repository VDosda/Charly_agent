from __future__ import annotations

import sqlite3
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    MemorySettings,
    Settings,
)
from agent.core.runtime import AgentRuntime
from agent.providers.embeddings.base import EmbeddingResult
from agent.providers.llm.base import LLMResult, ToolCall, ToolSpec
from agent.skills.registry import Tool, ToolRegistry


MIGRATIONS = [
    "0001_init.sql",
    "0002_mt_episodes.sql",
    "0003_lt_memory_items.sql",
]


class FastLLM:
    def generate(self, messages, tools=None, tool_choice=None):
        return LLMResult(text="ok-main-response", tool_calls=[])

    def supports_tools(self) -> bool:
        return False


class FastEmbeddings:
    def embed(self, texts):
        vectors = [[0.11, 0.22, 0.33, 0.44] for _ in texts]
        return EmbeddingResult(vectors=vectors, dimensions=4, model="test-embed")

    def supports_batch(self) -> bool:
        return True


class EchoToolHandler:
    def execute(self, args, context):
        return {"echo": args.get("value", "")}


class ToolThenSilentLLM:
    def __init__(self) -> None:
        self._calls = 0

    def supports_tools(self) -> bool:
        return True

    def generate(self, messages, tools=None, tool_choice=None):
        self._calls += 1
        if self._calls == 1:
            return LLMResult(
                text="",
                tool_calls=[ToolCall(id="tc-1", name="echo_tool", arguments={"value": "ping"})],
            )
        return LLMResult(text="", tool_calls=[])


class ToolThenTextLLM:
    def __init__(self) -> None:
        self._calls = 0

    def supports_tools(self) -> bool:
        return True

    def generate(self, messages, tools=None, tool_choice=None):
        self._calls += 1
        if self._calls == 1:
            return LLMResult(
                text="",
                tool_calls=[ToolCall(id="tc-1", name="echo_tool", arguments={"value": "ping"})],
            )
        return LLMResult(text="final tool answer", tool_calls=[])


def _migrations_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "agent" / "db" / "migrations"


def make_test_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    migrations_dir = _migrations_dir()
    for filename in MIGRATIONS:
        sql = (migrations_dir / filename).read_text(encoding="utf-8")
        conn.executescript(sql)
    return conn


def make_settings() -> Settings:
    return Settings(
        env="test",
        debug=True,
        db=DatabaseSettings(path=":memory:", vec_extension="none"),
        llm=LLMSettings(provider="ollama", model="test-model"),
        embeddings=EmbeddingSettings(provider="local", model="test-embed", dimensions=4),
        memory=MemorySettings(st_active_turns=20),
        workspace=".",
    )


class RuntimeMemoryBackgroundTests(unittest.TestCase):
    def test_lt_thresholds_from_settings_are_forwarded_to_distiller(self):
        settings = make_settings()
        settings.memory.lt_importance_threshold = 0.91
        settings.memory.lt_confidence_threshold = 0.83
        runtime = AgentRuntime(
            db=make_test_db(),
            llm=FastLLM(),
            embeddings=FastEmbeddings(),
            skills=ToolRegistry(),
            settings=settings,
        )

        captured = {}

        def fake_distill(*, cfg, **kwargs):
            captured["min_importance"] = cfg.min_importance
            captured["min_confidence"] = cfg.min_confidence
            return 0

        with patch("agent.core.runtime.maybe_distill_profile_from_st_window", side_effect=fake_distill):
            runtime._maybe_distill_lt_from_st(
                correlation_id="c-thresholds",
                user_id="u-thresholds",
                session_id="s-thresholds",
            )

        self.assertEqual(captured["min_importance"], 0.91)
        self.assertEqual(captured["min_confidence"], 0.83)

    def test_handle_message_returns_before_memory_job_finishes(self):
        runtime = AgentRuntime(
            db=make_test_db(),
            llm=FastLLM(),
            embeddings=FastEmbeddings(),
            skills=ToolRegistry(),
            settings=make_settings(),
        )

        finished = threading.Event()

        def slow_lt(*, correlation_id, user_id, session_id):
            time.sleep(0.30)
            finished.set()
            return 0

        runtime._maybe_distill_lt_from_st = slow_lt  # type: ignore[method-assign]
        runtime._maybe_create_episode = (  # type: ignore[method-assign]
            lambda *, correlation_id, user_id, session_id: {"episode_id": None, "embedding_retry_count": 0}
        )
        runtime._cleanup = lambda user_id, session_id: None  # type: ignore[method-assign]

        t0 = time.perf_counter()
        resp = runtime.handle_message(user_id="u-bg-1", session_id="s-bg-1", message="hello")
        elapsed = time.perf_counter() - t0

        self.assertEqual(resp, "ok-main-response")
        self.assertLess(elapsed, 0.25, "response should return before slow memory processing ends")
        self.assertTrue(finished.wait(2.0), "background memory job should eventually complete")

    def test_memory_exception_does_not_break_response(self):
        runtime = AgentRuntime(
            db=make_test_db(),
            llm=FastLLM(),
            embeddings=FastEmbeddings(),
            skills=ToolRegistry(),
            settings=make_settings(),
        )

        mt_called = threading.Event()

        def failing_lt(*, correlation_id, user_id, session_id):
            raise RuntimeError("simulated lt failure")

        def mt_ok(*, correlation_id, user_id, session_id):
            mt_called.set()
            return {"episode_id": None, "embedding_retry_count": 0}

        runtime._maybe_distill_lt_from_st = failing_lt  # type: ignore[method-assign]
        runtime._maybe_create_episode = mt_ok  # type: ignore[method-assign]
        runtime._cleanup = lambda user_id, session_id: None  # type: ignore[method-assign]

        resp = runtime.handle_message(user_id="u-bg-2", session_id="s-bg-2", message="bonjour")
        self.assertEqual(resp, "ok-main-response")

        self.assertTrue(runtime.wait_memory_idle(timeout_s=2.0))
        self.assertTrue(mt_called.is_set(), "memory worker should continue after LT exception")

    def test_tool_roundtrip_with_empty_final_text_uses_tool_fallback(self):
        registry = ToolRegistry()
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="echo_tool",
                    description="Echo value",
                    json_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                ),
                handler=EchoToolHandler(),
            )
        )
        runtime = AgentRuntime(
            db=make_test_db(),
            llm=ToolThenSilentLLM(),
            embeddings=FastEmbeddings(),
            skills=registry,
            settings=make_settings(),
        )
        runtime._maybe_distill_lt_from_st = lambda *, correlation_id, user_id, session_id: 0  # type: ignore[method-assign]
        runtime._maybe_create_episode = (  # type: ignore[method-assign]
            lambda *, correlation_id, user_id, session_id: {"episode_id": None, "embedding_retry_count": 0}
        )
        runtime._cleanup = lambda user_id, session_id: None  # type: ignore[method-assign]

        resp = runtime.handle_message(user_id="u-tools-1", session_id="s-tools-1", message="use tool")

        self.assertIn("[TOOL RESULT]", resp)
        self.assertIn("echo_tool", resp)
        self.assertIn("ping", resp)

    def test_tool_roundtrip_prefers_model_final_text_when_present(self):
        registry = ToolRegistry()
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="echo_tool",
                    description="Echo value",
                    json_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                ),
                handler=EchoToolHandler(),
            )
        )
        runtime = AgentRuntime(
            db=make_test_db(),
            llm=ToolThenTextLLM(),
            embeddings=FastEmbeddings(),
            skills=registry,
            settings=make_settings(),
        )
        runtime._maybe_distill_lt_from_st = lambda *, correlation_id, user_id, session_id: 0  # type: ignore[method-assign]
        runtime._maybe_create_episode = (  # type: ignore[method-assign]
            lambda *, correlation_id, user_id, session_id: {"episode_id": None, "embedding_retry_count": 0}
        )
        runtime._cleanup = lambda user_id, session_id: None  # type: ignore[method-assign]

        resp = runtime.handle_message(user_id="u-tools-2", session_id="s-tools-2", message="use tool")

        self.assertEqual(resp, "final tool answer")


if __name__ == "__main__":
    unittest.main()
