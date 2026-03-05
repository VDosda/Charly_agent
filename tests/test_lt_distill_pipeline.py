from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest.mock import patch

from agent.memory.distill_lt import (
    LTConfig,
    maybe_distill_profile_from_episode,
    maybe_distill_profile_from_st_window,
    retry_pending_lt_embeddings,
)
from agent.memory.distill_mt import maybe_create_episode
from agent.memory.store_lt import upsert_memory_item
from agent.memory.store_mt import insert_episode
from agent.providers.embeddings.base import EmbeddingResult
from agent.providers.llm.base import LLMResult


MIGRATIONS = [
    "0001_init.sql",
    "0002_mt_episodes.sql",
    "0003_lt_memory_items.sql",
]


class CapturingTracer:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(
        self,
        event: str,
        level: str,
        correlation_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "event": event,
                "level": level,
                "correlation_id": correlation_id,
                "user_id": user_id,
                "session_id": session_id,
                "payload": payload or {},
            }
        )


class StaticLLM:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: List[Sequence[Dict[str, Any]]] = []

    def generate(self, messages, tools=None, tool_choice=None):
        self.calls.append(messages)
        text = self.responses.pop(0) if self.responses else "[]"
        return LLMResult(text=text, tool_calls=[])

    def supports_tools(self) -> bool:
        return False


class StableEmbeddings:
    def embed(self, texts):
        vectors = []
        for t in texts:
            base = float((len(t) % 7) + 1)
            vectors.append([base, base + 0.1, base + 0.2, base + 0.3])
        return EmbeddingResult(vectors=vectors, dimensions=4, model="test-embed")

    def supports_batch(self) -> bool:
        return True


class FlakyEmbeddings:
    def __init__(self, fail_first: bool = True) -> None:
        self.fail_first = fail_first

    def embed(self, texts):
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("embedding provider down")
        vectors = [[0.11, 0.22, 0.33, 0.44] for _ in texts]
        return EmbeddingResult(vectors=vectors, dimensions=4, model="test-embed")

    def supports_batch(self) -> bool:
        return True


def _migrations_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "agent" / "db" / "migrations"


def make_test_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrations_dir = _migrations_dir()
    for filename in MIGRATIONS:
        sql = (migrations_dir / filename).read_text(encoding="utf-8")
        conn.executescript(sql)
    return conn


def seed_turns(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    session_id: str,
    start_turn: int,
    count: int,
    prefix: str,
) -> List[int]:
    ids: List[int] = []
    for i in range(count):
        turn_id = start_turn + i
        role = "tool" if i % 4 == 3 else "user"
        tool_name = "http.get" if role == "tool" else None
        content = f"{prefix} fact {i + 1}: durable setting {i + 1}"
        conn.execute(
            """
            INSERT INTO chat_history(
              user_id, session_id, turn_id, ts, role, content,
              tool_name, tool_args_json, tool_result_json, archived
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0)
            """,
            (user_id, session_id, turn_id, 1_700_000_000 + turn_id, role, content, tool_name),
        )
        ids.append(turn_id)
    conn.commit()
    return ids


def seed_episode(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    session_id: str,
    start_turn: int,
    end_turn: int,
    source_turn_ids: List[int],
    summary: str,
) -> int:
    return insert_episode(
        conn,
        user_id=user_id,
        session_id=session_id,
        start_turn_id=start_turn,
        end_turn_id=end_turn,
        ts=1_700_000_000 + end_turn,
        summary=summary,
        topics_json=json.dumps(["memory", "profile"]),
        facts_json=json.dumps([{"k": "scope", "v": "lt"}]),
        open_tasks_json=json.dumps([]),
        importance=0.7,
        confidence=0.8,
        embedding_model=None,
        embedding_dims=None,
        embedding_blob=None,
        source_turn_ids=source_turn_ids,
    )


class LTMemoryPipelineTests(unittest.TestCase):
    def test_upsert_update_preserves_previous_version(self):
        conn = make_test_db()

        first = upsert_memory_item(
            conn,
            user_id="u_ver_1",
            kind="goal",
            mem_key="user_goal",
            value="learn machine learning",
            confidence=0.8,
            importance=0.8,
            source_session_id="s_ver_1",
            source_note="seed",
            evidence_span="I want to learn ML",
            source_turn_ids_json=json.dumps([1]),
            embedding_status="ready",
        )
        self.assertEqual(first.action, "inserted")

        second = upsert_memory_item(
            conn,
            user_id="u_ver_1",
            kind="goal",
            mem_key="user_goal",
            value="build LLM agent",
            confidence=0.92,
            importance=0.9,
            source_session_id="s_ver_2",
            source_episode_id=7,
            source_note="update",
            evidence_span="Now I want to build an agent",
            source_turn_ids_json=json.dumps([2]),
            embedding_status="ready",
        )
        self.assertEqual(second.action, "updated")
        self.assertEqual(second.item_id, first.item_id)

        current = conn.execute(
            "SELECT value FROM memory_items WHERE id = ?",
            (first.item_id,),
        ).fetchone()
        self.assertEqual(current["value"], "build LLM agent")

        versions = conn.execute(
            """
            SELECT memory_item_id, value, change_reason
            FROM memory_item_versions
            WHERE memory_item_id = ?
            ORDER BY id ASC
            """,
            (first.item_id,),
        ).fetchall()
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["memory_item_id"], first.item_id)
        self.assertEqual(versions[0]["value"], "learn machine learning")
        self.assertEqual(versions[0]["change_reason"], "upsert_by_mem_key")

    def test_semantic_dedupe_merges_different_keys(self):
        conn = make_test_db()
        tracer = CapturingTracer()
        user = "u_sem_1"
        session = "s_sem_1"
        cfg = LTConfig(
            min_importance=0.1,
            min_confidence=0.1,
            min_st_turns=1,
            max_st_turns=80,
            semantic_dedupe_min_similarity=0.9,
            semantic_dedupe_top_k=3,
        )

        conn.execute(
            """
            INSERT INTO chat_history(
              user_id, session_id, turn_id, ts, role, content,
              tool_name, tool_args_json, tool_result_json, archived
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 0)
            """,
            (user, session, 1, 1_700_200_001, "user", "I use Grafana for monitoring."),
        )
        conn.commit()

        existing = upsert_memory_item(
            conn,
            user_id=user,
            kind="procedure",
            mem_key="user_uses_grafana",
            value="user uses grafana",
            confidence=0.9,
            importance=0.9,
            source_session_id=session,
            source_note="seed",
            evidence_span="I use Grafana for monitoring.",
            source_turn_ids_json=json.dumps([1]),
            embedding_status="ready",
        )

        llm = StaticLLM(
            [
                json.dumps(
                    [
                        {
                            "kind": "procedure",
                            "key": "monitoring_stack",
                            "value": "grafana monitoring",
                            "confidence": 0.95,
                            "importance": 0.9,
                            "evidence_span": "I use Grafana for monitoring.",
                            "source_turn_ids": [1],
                        }
                    ]
                )
            ]
        )

        with patch("agent.memory.distill_lt.query_topk_item_ids", return_value=[(existing.item_id, 0.01)]):
            upserted = maybe_distill_profile_from_st_window(
                db=conn,
                llm=llm,
                embeddings=StableEmbeddings(),
                tracer=tracer,
                correlation_id="c-sem-1",
                user_id=user,
                session_id=session,
                cfg=cfg,
                include_latest_mt_context=False,
            )

        self.assertEqual(upserted, 1)
        count = conn.execute("SELECT COUNT(*) FROM memory_items WHERE user_id = ?", (user,)).fetchone()[0]
        self.assertEqual(count, 1, "semantic duplicate should update existing row, not insert new one")

        row = conn.execute(
            """
            SELECT mem_key, value
            FROM memory_items
            WHERE id = ?
            """,
            (existing.item_id,),
        ).fetchone()
        self.assertEqual(row["mem_key"], "user_uses_grafana")
        self.assertEqual(row["value"], "grafana monitoring")

        merge_events = [e for e in tracer.events if e["event"] == "lt.item.semantic_merge"]
        self.assertGreaterEqual(len(merge_events), 1)

    def test_direct_lt_one_turn_durable_writes_without_mt_threshold(self):
        conn = make_test_db()
        tracer = CapturingTracer()
        user = "u_direct_1"
        session = "s_direct_1"
        cfg = LTConfig(min_importance=0.5, min_confidence=0.6, min_st_turns=1, max_st_turns=80)

        conn.execute(
            """
            INSERT INTO chat_history(
              user_id, session_id, turn_id, ts, role, content,
              tool_name, tool_args_json, tool_result_json, archived
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 0)
            """,
            (
                user,
                session,
                1,
                1_700_100_001,
                "user",
                "Je préfère que les réponses soient en français.",
            ),
        )
        conn.commit()

        llm = StaticLLM(
            [
                json.dumps(
                    [
                        {
                            "kind": "preference",
                            "key": "language",
                            "value": "fr",
                            "confidence": 0.92,
                            "importance": 0.88,
                            "evidence_span": "Je préfère que les réponses soient en français.",
                            "source_turn_ids": [1],
                        }
                    ]
                )
            ]
        )

        upserted = maybe_distill_profile_from_st_window(
            db=conn,
            llm=llm,
            embeddings=StableEmbeddings(),
            tracer=tracer,
            correlation_id="c-direct-1",
            user_id=user,
            session_id=session,
            cfg=cfg,
        )
        self.assertEqual(upserted, 1)

        row = conn.execute(
            """
            SELECT source_episode_id, source_note, evidence_span, source_turn_ids_json
            FROM memory_items
            WHERE user_id = ? AND kind = 'preference' AND mem_key = 'language'
            """,
            (user,),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row["source_episode_id"])
        self.assertEqual(row["source_note"], "distilled_direct_from_st_window")
        self.assertTrue((row["evidence_span"] or "").strip())
        self.assertTrue((row["source_turn_ids_json"] or "").strip())

    def test_direct_lt_small_talk_does_not_write(self):
        conn = make_test_db()
        tracer = CapturingTracer()
        user = "u_direct_2"
        session = "s_direct_2"
        cfg = LTConfig(min_importance=0.5, min_confidence=0.6, min_st_turns=1, max_st_turns=80)

        conn.execute(
            """
            INSERT INTO chat_history(
              user_id, session_id, turn_id, ts, role, content,
              tool_name, tool_args_json, tool_result_json, archived
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 0)
            """,
            (user, session, 1, 1_700_100_002, "user", "Salut, ça va ?"),
        )
        conn.execute(
            """
            INSERT INTO chat_history(
              user_id, session_id, turn_id, ts, role, content,
              tool_name, tool_args_json, tool_result_json, archived
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 0)
            """,
            (user, session, 2, 1_700_100_003, "assistant", "Oui, super. Et toi ?"),
        )
        conn.commit()

        llm = StaticLLM(["[]"])
        upserted = maybe_distill_profile_from_st_window(
            db=conn,
            llm=llm,
            embeddings=StableEmbeddings(),
            tracer=tracer,
            correlation_id="c-direct-2",
            user_id=user,
            session_id=session,
            cfg=cfg,
        )
        self.assertEqual(upserted, 0)
        count = conn.execute("SELECT COUNT(*) FROM memory_items WHERE user_id = ?", (user,)).fetchone()[0]
        self.assertEqual(count, 0)

        skip_events = [e for e in tracer.events if e["event"] == "lt.distill.skip"]
        self.assertTrue(any((e["payload"] or {}).get("reason") == "no_candidates" for e in skip_events))

    def test_mt_after_20_turns_and_direct_lt_coexist(self):
        conn = make_test_db()
        tracer = CapturingTracer()
        user = "u_direct_3"
        session = "s_direct_3"
        cfg = LTConfig(min_importance=0.5, min_confidence=0.6, min_st_turns=1, max_st_turns=80)

        seed_turns(conn, user_id=user, session_id=session, start_turn=1, count=20, prefix="mix")

        mt_llm = StaticLLM(
            [
                json.dumps(
                    {
                        "summary": "Episode summary",
                        "topics": ["memory"],
                        "facts": [{"k": "mode", "v": "test"}],
                        "open_tasks": [],
                        "importance": 0.8,
                        "confidence": 0.9,
                    }
                )
            ]
        )
        episode_id = maybe_create_episode(
            db=conn,
            llm=mt_llm,
            embeddings=StableEmbeddings(),
            tracer=tracer,
            correlation_id="c-mt-20",
            user_id=user,
            session_id=session,
        )
        self.assertIsNotNone(episode_id, "MT should be created once 20 turns are present")

        lt_llm = StaticLLM(
            [
                json.dumps(
                    [
                        {
                            "kind": "constraint",
                            "key": "stack",
                            "value": "sqlite",
                            "confidence": 0.91,
                            "importance": 0.84,
                            "evidence_span": "durable setting 18",
                            "source_turn_ids": [18],
                        }
                    ]
                )
            ]
        )
        lt_upserted = maybe_distill_profile_from_st_window(
            db=conn,
            llm=lt_llm,
            embeddings=StableEmbeddings(),
            tracer=tracer,
            correlation_id="c-lt-after-20",
            user_id=user,
            session_id=session,
            cfg=cfg,
        )
        self.assertEqual(lt_upserted, 1)

        ep_count = conn.execute("SELECT COUNT(*) FROM episodes WHERE session_id = ?", (session,)).fetchone()[0]
        self.assertEqual(ep_count, 1)
        lt_count = conn.execute("SELECT COUNT(*) FROM memory_items WHERE user_id = ?", (user,)).fetchone()[0]
        self.assertGreaterEqual(lt_count, 1)

    def test_recall_and_mem_key_stability(self):
        conn = make_test_db()
        tracer = CapturingTracer()
        user = "u1"
        session = "s1"
        cfg = LTConfig(min_importance=0.1, min_confidence=0.1, min_st_turns=1, max_st_turns=80)

        turns_1 = seed_turns(conn, user_id=user, session_id=session, start_turn=1, count=10, prefix="run1")
        ep1 = seed_episode(
            conn,
            user_id=user,
            session_id=session,
            start_turn=1,
            end_turn=10,
            source_turn_ids=turns_1,
            summary="first summary",
        )
        items_1 = [
            {
                "kind": "preference",
                "key": f"pref_{i}",
                "value": f"value_{i}",
                "confidence": 0.9,
                "importance": 0.9,
                "evidence_span": f"durable setting {i}",
                "source_turn_ids": [i],
            }
            for i in range(1, 11)
        ]
        llm = StaticLLM([json.dumps(items_1), json.dumps(items_1)])
        emb = StableEmbeddings()

        upserted_1 = maybe_distill_profile_from_episode(
            db=conn,
            llm=llm,
            embeddings=emb,
            tracer=tracer,
            correlation_id="c1",
            user_id=user,
            session_id=session,
            episode_id=ep1,
            cfg=cfg,
        )
        self.assertEqual(upserted_1, 10)
        count_1 = conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
        self.assertEqual(count_1, 10)

        turns_2 = seed_turns(conn, user_id=user, session_id=session, start_turn=11, count=10, prefix="run2")
        ep2 = seed_episode(
            conn,
            user_id=user,
            session_id=session,
            start_turn=11,
            end_turn=20,
            source_turn_ids=turns_2,
            summary="second summary",
        )
        items_2 = [
            {
                "kind": "preference",
                "key": f"pref_{i}",
                "value": f"value_{i}_updated",
                "confidence": 0.92,
                "importance": 0.95,
                "evidence_span": f"durable setting {i}",
                "source_turn_ids": [10 + i],
            }
            for i in range(1, 11)
        ]
        llm.responses = [json.dumps(items_2)]
        upserted_2 = maybe_distill_profile_from_episode(
            db=conn,
            llm=llm,
            embeddings=emb,
            tracer=tracer,
            correlation_id="c2",
            user_id=user,
            session_id=session,
            episode_id=ep2,
            cfg=cfg,
        )
        self.assertEqual(upserted_2, 10)
        count_2 = conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
        self.assertEqual(count_2, 10, "upsert by mem_key should keep stable cardinality")

        row = conn.execute(
            "SELECT value FROM memory_items WHERE user_id = ? AND kind = ? AND mem_key = ?",
            (user, "preference", "pref_3"),
        ).fetchone()
        self.assertEqual(row["value"], "value_3_updated")

    def test_anti_hallucination_requires_evidence(self):
        conn = make_test_db()
        tracer = CapturingTracer()
        user = "u2"
        session = "s2"
        cfg = LTConfig(min_importance=0.1, min_confidence=0.1, min_st_turns=1, max_st_turns=80)

        turns = seed_turns(conn, user_id=user, session_id=session, start_turn=1, count=4, prefix="proof")
        ep = seed_episode(
            conn,
            user_id=user,
            session_id=session,
            start_turn=1,
            end_turn=4,
            source_turn_ids=turns,
            summary="summary",
        )

        llm = StaticLLM(
            [
                json.dumps(
                    [
                        {
                            "kind": "constraint",
                            "key": "stack",
                            "value": "Use sqlite",
                            "confidence": 0.9,
                            "importance": 0.9,
                            "evidence_span": "",
                            "source_turn_ids": [1],
                        },
                        {
                            "kind": "preference",
                            "key": "lang",
                            "value": "French",
                            "confidence": 0.9,
                            "importance": 0.9,
                            "evidence_span": "durable setting 2",
                            "source_turn_ids": [],
                        },
                        {
                            "kind": "preference",
                            "key": "timezone",
                            "value": "Europe/Paris",
                            "confidence": 0.9,
                            "importance": 0.9,
                            "evidence_span": "durable setting 3",
                            "source_turn_ids": [3],
                        },
                    ]
                )
            ]
        )

        upserted = maybe_distill_profile_from_episode(
            db=conn,
            llm=llm,
            embeddings=StableEmbeddings(),
            tracer=tracer,
            correlation_id="c3",
            user_id=user,
            session_id=session,
            episode_id=ep,
            cfg=cfg,
        )
        self.assertEqual(upserted, 1)

        rows = conn.execute(
            "SELECT evidence_span, source_turn_ids_json FROM memory_items ORDER BY id ASC"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertTrue((rows[0]["evidence_span"] or "").strip())
        self.assertTrue((rows[0]["source_turn_ids_json"] or "").strip())

        skip_events = [e for e in tracer.events if e["event"] == "lt.item.skip"]
        self.assertGreaterEqual(len(skip_events), 2)

    def test_embedding_failure_non_blocking_with_retry(self):
        conn = make_test_db()
        tracer = CapturingTracer()
        user = "u3"
        session = "s3"
        cfg = LTConfig(min_importance=0.1, min_confidence=0.1, min_st_turns=1, max_st_turns=80)

        turns = seed_turns(conn, user_id=user, session_id=session, start_turn=1, count=3, prefix="embed")
        ep = seed_episode(
            conn,
            user_id=user,
            session_id=session,
            start_turn=1,
            end_turn=3,
            source_turn_ids=turns,
            summary="summary",
        )

        llm = StaticLLM(
            [
                json.dumps(
                    [
                        {
                            "kind": "goal",
                            "key": "g1",
                            "value": "Ship release",
                            "confidence": 0.95,
                            "importance": 0.9,
                            "evidence_span": "durable setting 1",
                            "source_turn_ids": [1],
                        }
                    ]
                )
            ]
        )
        emb = FlakyEmbeddings(fail_first=True)

        upserted = maybe_distill_profile_from_episode(
            db=conn,
            llm=llm,
            embeddings=emb,
            tracer=tracer,
            correlation_id="c4",
            user_id=user,
            session_id=session,
            episode_id=ep,
            cfg=cfg,
        )
        self.assertEqual(upserted, 1)

        row = conn.execute(
            """
            SELECT embedding_status, last_embedding_error, embedding_retry_count
            FROM memory_items
            WHERE user_id = ? AND kind = 'goal' AND mem_key = 'g1'
            """,
            (user,),
        ).fetchone()
        self.assertEqual(row["embedding_status"], "pending")
        self.assertTrue((row["last_embedding_error"] or "").strip())
        self.assertEqual(int(row["embedding_retry_count"]), 0)

        retry_stats = retry_pending_lt_embeddings(
            db=conn,
            embeddings=emb,
            tracer=tracer,
            correlation_id="c4-retry",
            user_id=user,
            session_id=session,
            limit=5,
            force=True,
        )
        self.assertEqual(retry_stats["succeeded"], 1)

        row2 = conn.execute(
            """
            SELECT embedding_status, last_embedding_error, embedding_dims, embedding_blob
            FROM memory_items
            WHERE user_id = ? AND kind = 'goal' AND mem_key = 'g1'
            """,
            (user,),
        ).fetchone()
        self.assertEqual(row2["embedding_status"], "ready")
        self.assertIsNone(row2["last_embedding_error"])
        self.assertEqual(int(row2["embedding_dims"]), 4)
        self.assertIsNotNone(row2["embedding_blob"])


if __name__ == "__main__":
    unittest.main()
