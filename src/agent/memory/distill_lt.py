from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import sqlite3

from agent.core.tracing import JSONTracer
from agent.memory.memory_manager import (
    MemoryIntent,
    MemoryManager,
    MemoryManagerConfig,
)
from agent.memory.store_lt import (
    list_embedding_retry_candidates,
    mark_embedding_retry_failure,
    mark_embedding_retry_success,
)
from agent.memory.vector_store import query_topk_item_ids, upsert_item_vec
from agent.providers.embeddings.base import EmbeddingProvider
from agent.providers.embeddings.utils import pack_f32
from agent.providers.llm.base import LLMProvider


@dataclass(frozen=True)
class LTConfig:
    max_items: int = 12
    min_importance: float = 0.50
    min_confidence: float = 0.60
    min_st_turns: int = 40
    max_st_turns: int = 80
    max_chars_per_turn: int = 1600
    semantic_dedupe_top_k: int = 6
    semantic_dedupe_min_similarity: float = 0.92


def maybe_distill_profile_from_st_window(
    *,
    db: sqlite3.Connection,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    tracer: JSONTracer,
    correlation_id: str,
    user_id: str,
    session_id: str,
    cfg: Optional[LTConfig] = None,
    include_latest_mt_context: bool = True,
) -> int:
    """
    Direct LT path: distill from ST window without requiring an MT episode.
    Returns number of upserted LT items.
    """
    cfg = cfg or LTConfig()
    latest_mt = _read_latest_mt_payload(db, session_id=session_id) if include_latest_mt_context else None
    last_episode_end_turn = int(latest_mt.get("end_turn_id") or 0) if latest_mt else 0

    st_turns = _read_st_window_since_last_episode(
        db,
        session_id=session_id,
        last_episode_end_turn=last_episode_end_turn,
        min_turns=cfg.min_st_turns,
        max_turns=cfg.max_st_turns,
    )
    mt_payload: Dict[str, Any] = {}
    if latest_mt:
        mt_payload = {
            "summary": latest_mt.get("summary"),
            "topics": _safe_load_json(latest_mt.get("topics_json")),
            "facts": _safe_load_json(latest_mt.get("facts_json")),
            "open_tasks": _safe_load_json(latest_mt.get("open_tasks_json")),
        }

    return _distill_lt_from_st_window(
        db=db,
        llm=llm,
        embeddings=embeddings,
        tracer=tracer,
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        cfg=cfg,
        st_turns=st_turns,
        mt_payload=mt_payload,
        source_episode_id=None,
        source_note="distilled_direct_from_st_window",
        mode="direct",
        episode_id_for_trace=None,
        last_episode_end_turn=last_episode_end_turn,
    )


def maybe_distill_profile_from_episode(
    *,
    db: sqlite3.Connection,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    tracer: JSONTracer,
    correlation_id: str,
    user_id: str,
    session_id: str,
    episode_id: int,
    cfg: Optional[LTConfig] = None,
) -> int:
    """
    Extract LT memory items from ST window (source of truth) + optional MT summary.
    Returns number of upserted items.
    """
    cfg = cfg or LTConfig()

    ep = db.execute(
        """
        SELECT
          id, session_id, start_turn_id, end_turn_id,
          summary, topics_json, facts_json, open_tasks_json
        FROM episodes
        WHERE id = ?
        """,
        (episode_id,),
    ).fetchone()
    if not ep:
        return 0

    st_turns = _read_st_window(
        db,
        session_id=session_id,
        episode_id=int(ep["id"]),
        start_turn_id=int(ep["start_turn_id"]),
        end_turn_id=int(ep["end_turn_id"]),
        min_turns=cfg.min_st_turns,
        max_turns=cfg.max_st_turns,
    )

    mt_payload = {
        "summary": ep["summary"],
        "topics": _safe_load_json(ep["topics_json"]),
        "facts": _safe_load_json(ep["facts_json"]),
        "open_tasks": _safe_load_json(ep["open_tasks_json"]),
    }

    return _distill_lt_from_st_window(
        db=db,
        llm=llm,
        embeddings=embeddings,
        tracer=tracer,
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        cfg=cfg,
        st_turns=st_turns,
        mt_payload=mt_payload,
        source_episode_id=episode_id,
        source_note="distilled_from_episode_st_window",
        mode="from_episode",
        episode_id_for_trace=episode_id,
        last_episode_end_turn=int(ep["end_turn_id"]),
    )


def _distill_lt_from_st_window(
    *,
    db: sqlite3.Connection,
    llm: LLMProvider,
    embeddings: EmbeddingProvider,
    tracer: JSONTracer,
    correlation_id: str,
    user_id: str,
    session_id: str,
    cfg: LTConfig,
    st_turns: Sequence[Dict[str, Any]],
    mt_payload: Optional[Dict[str, Any]],
    source_episode_id: Optional[int],
    source_note: str,
    mode: str,
    episode_id_for_trace: Optional[int],
    last_episode_end_turn: int,
) -> int:
    tracer.emit(
        event="lt.distill.start",
        level="info",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={
            "mode": mode,
            "episode_id": episode_id_for_trace,
            "st_turns": len(st_turns),
            "st_min_turns": cfg.min_st_turns,
            "st_max_turns": cfg.max_st_turns,
            "last_episode_end_turn": int(last_episode_end_turn),
        },
    )

    if not st_turns:
        tracer.emit(
            event="lt.distill.skip",
            level="warning",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "mode": mode,
                "episode_id": episode_id_for_trace,
                "reason": "empty_st_window",
            },
        )
        return 0

    st_turn_ids = {int(t["turn_id"]) for t in st_turns}
    prompts = _lt_messages(
        st_turns=st_turns,
        mt_payload=mt_payload or {},
        max_items=cfg.max_items,
        min_confidence=cfg.min_confidence,
        max_chars_per_turn=cfg.max_chars_per_turn,
    )

    tracer.emit(
        event="lt.llm.prompt",
        level="debug",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={
            "mode": mode,
            "episode_id": episode_id_for_trace,
            "max_items": cfg.max_items,
            "min_importance": cfg.min_importance,
            "min_confidence": cfg.min_confidence,
            "system_prompt": _truncate_for_trace(prompts[0]["content"]),
            "system_prompt_chars": len(prompts[0]["content"] or ""),
            "system_prompt_truncated": len(prompts[0]["content"] or "") > 4000,
            "user_prompt": _truncate_for_trace(prompts[1]["content"]),
            "user_prompt_chars": len(prompts[1]["content"] or ""),
            "user_prompt_truncated": len(prompts[1]["content"] or "") > 4000,
        },
    )

    res = llm.generate(
        messages=prompts,
        tools=None,
        tool_choice="none",
    )
    intents = _parse_intents_json(res.text)
    tracer.emit(
        event="lt.llm.response",
        level="debug",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={
            "mode": mode,
            "episode_id": episode_id_for_trace,
            "raw_response_chars": len((res.text or "").strip()),
            "raw_response_preview": _truncate_for_trace((res.text or "").strip()),
            "raw_response_preview_truncated": len((res.text or "").strip()) > 4000,
            "parsed_items": len(intents),
        },
    )

    if not intents:
        tracer.emit(
            event="lt.distill.skip",
            level="debug",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "mode": mode,
                "episode_id": episode_id_for_trace,
                "reason": "no_candidates",
            },
        )
        return 0

    manager = MemoryManager(
        db=db,
        embeddings=embeddings,
        tracer=tracer,
        cfg=MemoryManagerConfig(
            min_importance=cfg.min_importance,
            min_confidence=cfg.min_confidence,
            semantic_dedupe_top_k=cfg.semantic_dedupe_top_k,
            semantic_dedupe_min_similarity=cfg.semantic_dedupe_min_similarity,
        ),
        # Keep this dependency injection explicit so tests can patch
        # `agent.memory.distill_lt.query_topk_item_ids`.
        semantic_query_fn=query_topk_item_ids,
    )

    upserted = 0
    inserted = 0
    updated = 0
    archived = 0
    skipped = 0

    for intent in intents[: cfg.max_items]:
        key_hint = str((intent.attributes or {}).get("key_hint") or "").strip() or None

        tracer.emit(
            event="lt.item.candidate",
            level="debug",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "mode": mode,
                "episode_id": episode_id_for_trace,
                "action": intent.action,
                "entity_type": intent.entity_type,
                "entity_name": intent.entity_name,
                "kind": intent.entity_type,
                "mem_key_hint": key_hint,
                "description_chars": len(intent.description or ""),
                "confidence": intent.confidence,
                "importance": intent.importance,
                "evidence_chars": len(intent.evidence_span or ""),
                "source_turn_ids": intent.source_turn_ids,
            },
        )

        result = manager.process_intent(
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            intent=intent,
            st_turn_ids=st_turn_ids,
            source_episode_id=source_episode_id,
            source_note=source_note,
            mode=mode,
            episode_id_for_trace=episode_id_for_trace,
        )

        if not result.applied:
            skipped += 1
            tracer.emit(
                event="lt.item.skip",
                level="debug",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={
                    "mode": mode,
                    "episode_id": episode_id_for_trace,
                    "reason": result.skip_reason,
                    "action": intent.action,
                    "kind": result.kind or intent.entity_type,
                    "mem_key_hint": key_hint,
                    "importance": intent.importance,
                    "confidence": intent.confidence,
                    "evidence_chars": len(intent.evidence_span or ""),
                    "source_turn_ids": intent.source_turn_ids,
                },
            )
            continue

        if result.action == "inserted":
            inserted += 1
        elif result.action == "updated":
            updated += 1
        elif result.action == "archived":
            archived += 1

        tracer.emit(
            event="lt.item.upsert",
            level="info",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "mode": mode,
                "episode_id": episode_id_for_trace,
                "item_id": result.item_id,
                "action": result.action,
                "kind": result.kind or intent.entity_type,
                "mem_key": result.mem_key,
                "confidence": intent.confidence,
                "importance": intent.importance,
                "source_turn_ids": intent.source_turn_ids,
                "embedding_status": result.embedding_status,
            },
        )
        upserted += 1

    tracer.emit(
        event="lt.distill.end",
        level="info",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={
            "mode": mode,
            "episode_id": episode_id_for_trace,
            "upserted": upserted,
            "inserted": inserted,
            "updated": updated,
            "archived": archived,
            "skipped": skipped,
            "considered": min(len(intents), cfg.max_items),
            "max_items": cfg.max_items,
            "min_importance": cfg.min_importance,
            "min_confidence": cfg.min_confidence,
            "st_turns": len(st_turns),
        },
    )

    return upserted


def retry_pending_lt_embeddings(
    *,
    db: sqlite3.Connection,
    embeddings: EmbeddingProvider,
    tracer: JSONTracer,
    correlation_id: str,
    user_id: Optional[str],
    session_id: Optional[str],
    limit: int = 8,
    force: bool = False,
) -> Dict[str, int]:
    now = int(time.time())
    candidates = list_embedding_retry_candidates(
        db,
        now_ts=now,
        limit=limit,
        user_id=user_id,
        due_only=not force,
    )

    tracer.emit(
        event="lt.embedding.retry.start",
        level="debug",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={"candidates": len(candidates), "limit": int(limit), "force": bool(force)},
    )

    succeeded = 0
    failed = 0
    skipped = 0

    for row in candidates:
        item_id = int(row["id"])
        value = str(row.get("value") or "").strip()
        kind = str(row.get("kind") or "").strip()
        mem_key = str(row.get("mem_key") or "").strip()
        retry_count = int(row.get("embedding_retry_count") or 0)

        if not value:
            skipped += 1
            continue

        try:
            emb = embeddings.embed([f"{kind}: {mem_key} {value}".strip()])
            vec = emb.vectors[0] if emb.vectors else []
            if not vec:
                raise RuntimeError("Embedding provider returned an empty vector")

            emb_blob = pack_f32(vec)
            mark_embedding_retry_success(
                db,
                item_id=item_id,
                embedding_model=emb.model,
                embedding_dims=emb.dimensions,
                embedding_blob=emb_blob,
                ts_now=now,
            )
            try:
                upsert_item_vec(db, item_id, vec)
            except Exception as e:
                tracer.emit(
                    event="lt.vec.upsert.error",
                    level="warning",
                    correlation_id=correlation_id,
                    user_id=row.get("user_id"),
                    session_id=session_id,
                    payload={"item_id": item_id, "error": f"{type(e).__name__}: {e}"},
                )
            succeeded += 1
            tracer.emit(
                event="lt.embedding.retry.item",
                level="info",
                correlation_id=correlation_id,
                user_id=row.get("user_id"),
                session_id=session_id,
                payload={"item_id": item_id, "ok": True, "retry_count": retry_count},
            )
        except Exception as e:
            failed += 1
            error = f"{type(e).__name__}: {e}"
            next_retry = now + _retry_delay_seconds(retry_count + 1)
            mark_embedding_retry_failure(
                db,
                item_id=item_id,
                error=error,
                ts_now=now,
                next_retry_ts=next_retry,
            )
            tracer.emit(
                event="lt.embedding.retry.item",
                level="warning",
                correlation_id=correlation_id,
                user_id=row.get("user_id"),
                session_id=session_id,
                payload={
                    "item_id": item_id,
                    "ok": False,
                    "retry_count": retry_count + 1,
                    "error": error,
                    "next_retry_ts": next_retry,
                },
            )

    tracer.emit(
        event="lt.embedding.retry.end",
        level="debug",
        correlation_id=correlation_id,
        user_id=user_id,
        session_id=session_id,
        payload={
            "processed": len(candidates),
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
        },
    )

    return {
        "processed": len(candidates),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
    }


def _lt_messages(
    *,
    st_turns: Sequence[Dict[str, Any]],
    mt_payload: Dict[str, Any],
    max_items: int,
    min_confidence: float,
    max_chars_per_turn: int,
) -> List[Dict[str, Any]]:
    st_ids = [int(t["turn_id"]) for t in st_turns]
    st_transcript = _render_st_transcript(st_turns, max_chars_per_turn=max_chars_per_turn)
    system = (
        "You extract LONG-TERM memory intents for an agent.\n"
        "SOURCE OF TRUTH: ST transcript below.\n"
        "MT summary is optional context only (prioritization, dedupe).\n"
        "Return STRICT JSON only. No markdown.\n"
        "Return an array with schema:\n"
        "{\n"
        '  "action": "create"|"update"|"archive"|"none",\n'
        '  "entity_type": "identity"|"preference"|"constraint"|"goal"|"procedure"|"project"|"other",\n'
        '  "entity_name": string,\n'
        '  "description": string,\n'
        '  "attributes": object,\n'
        '  "confidence": number,\n'
        '  "importance": number,\n'
        '  "evidence_span": string,\n'
        '  "source_turn_ids": [integer]\n'
        "}\n"
        f"Max items: {max_items}.\n"
        f"Minimum confidence guidance: {min_confidence:.2f}.\n"
        "Rules:\n"
        "- Only durable facts/preferences/constraints/procedures/projects.\n"
        "- Evidence is mandatory for every item.\n"
        "- source_turn_ids MUST reference turn ids present in ST transcript.\n"
        "- If no evidence in ST for an intent, do not output it.\n"
        "- The LLM must not define final mem_key. Use attributes only as hints.\n"
        "- `entity_name` and `attributes.canonical_value_en` MUST be canonical English.\n"
        "- For project lifecycle updates, include `attributes.status` in English (active/abandoned/etc.).\n"
        "- For archive action, describe what should be archived in description/attributes.\n"
        "- If no durable memory intent exists, return an empty array.\n"
    )
    user = (
        "ST transcript (ground truth):\n"
        f"{st_transcript}\n\n"
        f"Allowed source_turn_ids: {json.dumps(st_ids, ensure_ascii=False)}\n\n"
        "Optional MT context:\n"
        f"{json.dumps(mt_payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _read_st_window(
    db: sqlite3.Connection,
    *,
    session_id: str,
    episode_id: int,
    start_turn_id: int,
    end_turn_id: int,
    min_turns: int,
    max_turns: int,
) -> List[Dict[str, Any]]:
    rows = db.execute(
        """
        SELECT ch.turn_id, ch.role, ch.content, ch.tool_name
        FROM episode_sources es
        JOIN chat_history ch
          ON ch.session_id = es.session_id
         AND ch.turn_id = es.turn_id
        WHERE es.episode_id = ?
          AND es.session_id = ?
        ORDER BY ch.turn_id ASC
        """,
        (int(episode_id), session_id),
    ).fetchall()

    if not rows:
        rows = db.execute(
            """
            SELECT turn_id, role, content, tool_name
            FROM chat_history
            WHERE session_id = ?
              AND turn_id BETWEEN ? AND ?
            ORDER BY turn_id ASC
            """,
            (session_id, int(start_turn_id), int(end_turn_id)),
        ).fetchall()

    turns = [
        {
            "turn_id": int(r["turn_id"]),
            "role": str(r["role"] or ""),
            "content": str(r["content"] or ""),
            "tool_name": r["tool_name"],
        }
        for r in rows
    ]

    if turns and len(turns) < int(min_turns):
        earliest = int(turns[0]["turn_id"])
        extra_cap = max(0, int(max_turns) - len(turns))
        extra_need = min(int(min_turns) - len(turns), extra_cap)
        if earliest > 1 and extra_need > 0:
            extra_start = max(1, earliest - extra_need)
            extra_rows = db.execute(
                """
                SELECT turn_id, role, content, tool_name
                FROM chat_history
                WHERE session_id = ?
                  AND turn_id BETWEEN ? AND ?
                ORDER BY turn_id ASC
                """,
                (session_id, int(extra_start), int(earliest - 1)),
            ).fetchall()
            extra_turns = [
                {
                    "turn_id": int(r["turn_id"]),
                    "role": str(r["role"] or ""),
                    "content": str(r["content"] or ""),
                    "tool_name": r["tool_name"],
                }
                for r in extra_rows
            ]
            turns = extra_turns + turns

    if len(turns) > int(max_turns):
        turns = turns[-int(max_turns) :]

    return turns


def _read_st_window_since_last_episode(
    db: sqlite3.Connection,
    *,
    session_id: str,
    last_episode_end_turn: int,
    min_turns: int,
    max_turns: int,
) -> List[Dict[str, Any]]:
    rows = db.execute(
        """
        SELECT turn_id, role, content, tool_name
        FROM chat_history
        WHERE session_id = ?
          AND turn_id > ?
        ORDER BY turn_id ASC
        """,
        (session_id, int(last_episode_end_turn)),
    ).fetchall()

    turns = [
        {
            "turn_id": int(r["turn_id"]),
            "role": str(r["role"] or ""),
            "content": str(r["content"] or ""),
            "tool_name": r["tool_name"],
        }
        for r in rows
    ]

    if not turns:
        return []

    if len(turns) > int(max_turns):
        turns = turns[-int(max_turns) :]

    if len(turns) < int(min_turns):
        earliest = int(turns[0]["turn_id"])
        extra_cap = max(0, int(max_turns) - len(turns))
        extra_need = min(int(min_turns) - len(turns), extra_cap)
        if earliest > 1 and extra_need > 0:
            extra_start = max(1, earliest - extra_need)
            extra_rows = db.execute(
                """
                SELECT turn_id, role, content, tool_name
                FROM chat_history
                WHERE session_id = ?
                  AND turn_id BETWEEN ? AND ?
                ORDER BY turn_id ASC
                """,
                (session_id, int(extra_start), int(earliest - 1)),
            ).fetchall()
            extra_turns = [
                {
                    "turn_id": int(r["turn_id"]),
                    "role": str(r["role"] or ""),
                    "content": str(r["content"] or ""),
                    "tool_name": r["tool_name"],
                }
                for r in extra_rows
            ]
            turns = extra_turns + turns

    if len(turns) > int(max_turns):
        turns = turns[-int(max_turns) :]

    return turns


def _read_latest_mt_payload(db: sqlite3.Connection, *, session_id: str) -> Optional[Dict[str, Any]]:
    row = db.execute(
        """
        SELECT
          id, end_turn_id, summary, topics_json, facts_json, open_tasks_json
        FROM episodes
        WHERE session_id = ?
        ORDER BY end_turn_id DESC, id DESC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()

    if not row:
        return None

    return {
        "id": int(row["id"]),
        "end_turn_id": int(row["end_turn_id"] or 0),
        "summary": row["summary"],
        "topics_json": row["topics_json"],
        "facts_json": row["facts_json"],
        "open_tasks_json": row["open_tasks_json"],
    }


def _render_st_transcript(turns: Sequence[Dict[str, Any]], *, max_chars_per_turn: int) -> str:
    lines: List[str] = []
    for t in turns:
        turn_id = int(t.get("turn_id") or 0)
        role = str(t.get("role") or "").strip().lower() or "unknown"
        content = str(t.get("content") or "").strip()
        if len(content) > max_chars_per_turn:
            content = content[: max_chars_per_turn - 1] + "…"
        if role == "tool":
            tool_name = str(t.get("tool_name") or "tool")
            lines.append(f"{turn_id:04d} TOOL {tool_name}: {content}")
        else:
            lines.append(f"{turn_id:04d} {role.upper()}: {content}")
    return "\n".join(lines)


def _parse_intents_json(text: str) -> List[MemoryIntent]:
    """
    Parse intents from strict JSON and coerce each object into MemoryIntent.

    Backward compatibility:
    - legacy item schema {"kind","key","value",...} is mapped to
      action="create" intent with attributes.key_hint/value.
    """
    if not text:
        return []
    s = text.strip()
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []

    intents: List[MemoryIntent] = []
    for raw in obj:
        intent = _coerce_memory_intent(raw)
        if intent is None:
            continue
        intents.append(intent)
    return intents


def _coerce_memory_intent(raw: Any) -> Optional[MemoryIntent]:
    if not isinstance(raw, dict):
        return None

    action = str(raw.get("action") or "").strip().lower()
    entity_type = str(raw.get("entity_type") or "").strip().lower()
    entity_name = str(raw.get("entity_name") or "").strip()
    description = str(raw.get("description") or "").strip()
    attributes = raw.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}

    # Legacy schema fallback.
    if not action:
        action = "create"
    if not entity_type:
        entity_type = str(raw.get("kind") or "").strip().lower()
    if not entity_name:
        entity_name = (
            str(attributes.get("entity_name") or "").strip()
            or str(attributes.get("name") or "").strip()
            or str(raw.get("key") or "").strip()
        )
    if not description:
        description = str(raw.get("value") or "").strip()

    if raw.get("key") is not None and "key_hint" not in attributes:
        key_hint = str(raw.get("key") or "").strip()
        if key_hint:
            attributes["key_hint"] = key_hint
    if raw.get("value") is not None and "value" not in attributes:
        value_hint = str(raw.get("value") or "").strip()
        if value_hint:
            attributes["value"] = value_hint

    confidence = _clamp01(_to_float(raw.get("confidence"), 0.6))
    importance = _clamp01(_to_float(raw.get("importance"), 0.5))
    evidence_span = str(raw.get("evidence_span") or "").strip()
    source_turn_ids = _coerce_int_list(raw.get("source_turn_ids"))

    return MemoryIntent(
        action=action,
        entity_type=entity_type,
        entity_name=entity_name,
        description=description,
        attributes=attributes,
        evidence_span=evidence_span,
        source_turn_ids=source_turn_ids,
        confidence=confidence,
        importance=importance,
    )


def _safe_load_json(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _coerce_int_list(value: Any) -> List[int]:
    if isinstance(value, list):
        out: List[int] = []
        for v in value:
            try:
                out.append(int(v))
            except Exception:
                continue
        # Keep order, dedup
        seen = set()
        uniq: List[int] = []
        for v in out:
            if v in seen:
                continue
            seen.add(v)
            uniq.append(v)
        return uniq
    return []


def _retry_delay_seconds(retry_count: int) -> int:
    base = 30
    capped = min(3600, base * (2 ** max(0, int(retry_count))))
    return int(capped)


def _truncate_for_trace(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
