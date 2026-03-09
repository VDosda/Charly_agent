from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import sqlite3

from agent.infrastructure.tracing import JSONTracer
from agent.domain.memory.scoring import distance_to_similarity
from agent.infrastructure.memory.store_lt import (
    list_embedding_retry_candidates,
    mark_embedding_retry_failure,
    mark_embedding_retry_success,
    upsert_memory_item,
)
from agent.infrastructure.memory.vector_store import query_topk_item_ids, upsert_item_vec
from agent.infrastructure.providers.embeddings.base import EmbeddingProvider
from agent.infrastructure.providers.embeddings.utils import pack_f32
from agent.infrastructure.providers.llm.base import LLMProvider


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
    existing_index = _read_existing_memory_index(db, user_id=user_id)
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
    items = _parse_items_json(res.text)
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
            "parsed_items": len(items),
        },
    )

    if not items:
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

    upserted = 0
    inserted = 0
    updated = 0
    skipped = 0

    for it in items[: cfg.max_items]:
        kind = str(it.get("kind") or "").strip()
        mem_key = it.get("key")
        mem_key = str(mem_key).strip() if mem_key is not None else None
        if mem_key == "":
            mem_key = None
        value = str(it.get("value") or "").strip()
        confidence = _clamp01(_to_float(it.get("confidence"), 0.6))
        importance = _clamp01(_to_float(it.get("importance"), 0.5))
        evidence_span = str(it.get("evidence_span") or "").strip()
        source_turn_ids = _coerce_int_list(it.get("source_turn_ids"))
        valid_source_turn_ids = [tid for tid in source_turn_ids if tid in st_turn_ids]

        tracer.emit(
            event="lt.item.candidate",
            level="debug",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "mode": mode,
                "episode_id": episode_id_for_trace,
                "kind": kind,
                "mem_key": mem_key,
                "value_chars": len(value),
                "confidence": confidence,
                "importance": importance,
                "evidence_chars": len(evidence_span),
                "source_turn_ids": valid_source_turn_ids,
            },
        )

        skip_reason = _item_skip_reason(
            kind=kind,
            value=value,
            importance=importance,
            min_importance=cfg.min_importance,
            confidence=confidence,
            min_confidence=cfg.min_confidence,
            evidence_span=evidence_span,
            valid_source_turn_ids=valid_source_turn_ids,
        )
        if skip_reason:
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
                    "reason": skip_reason,
                    "kind": kind,
                    "mem_key": mem_key,
                    "importance": importance,
                    "confidence": confidence,
                    "evidence_chars": len(evidence_span),
                    "source_turn_ids": valid_source_turn_ids,
                },
            )
            continue

        source_turn_ids_json = json.dumps(valid_source_turn_ids, ensure_ascii=False)
        emb_status = "ready"
        emb_error = None
        emb_retry_count = 0
        emb_last_attempt_ts = int(time.time())
        emb_next_retry_ts = None

        vec: List[float] = []
        emb_blob = None
        emb_model = None
        emb_dims = None
        try:
            emb = embeddings.embed(
                [f"{kind}: {mem_key or ''} {value}\nEvidence: {evidence_span}".strip()]
            )
            vec = emb.vectors[0] if emb.vectors else []
            emb_model = emb.model
            emb_dims = emb.dimensions
            emb_blob = pack_f32(vec) if vec else None
            if not vec:
                emb_status = "pending"
                emb_error = "Embedding provider returned an empty vector"
                emb_next_retry_ts = emb_last_attempt_ts + _retry_delay_seconds(0)
        except Exception as e:
            emb_status = "pending"
            emb_error = f"{type(e).__name__}: {e}"
            emb_next_retry_ts = emb_last_attempt_ts + _retry_delay_seconds(0)
            tracer.emit(
                event="lt.embed.error",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"mode": mode, "episode_id": episode_id_for_trace, "error": emb_error},
            )

        semantic_match: Optional[Dict[str, Any]] = None
        mem_key_for_upsert = mem_key
        target_item_id: Optional[int] = None

        if vec:
            semantic_match = _find_semantic_duplicate(
                db=db,
                user_id=user_id,
                kind=kind,
                mem_key=mem_key,
                value=value,
                vec=vec,
                cfg=cfg,
                existing_index=existing_index,
            )

        if semantic_match:
            match_item_id = int(semantic_match["item_id"])
            match_mem_key = semantic_match.get("mem_key")
            match_similarity = float(semantic_match.get("similarity", 0.0))
            match_distance = float(semantic_match.get("distance", 0.0))

            # Prefer canonical existing mem_key when available.
            if match_mem_key is not None and str(match_mem_key).strip():
                mem_key_for_upsert = str(match_mem_key).strip()
            else:
                target_item_id = match_item_id

            tracer.emit(
                event="lt.item.semantic_merge",
                level="debug",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={
                    "mode": mode,
                    "episode_id": episode_id_for_trace,
                    "match_item_id": match_item_id,
                    "match_mem_key": match_mem_key,
                    "match_similarity": match_similarity,
                    "match_distance": match_distance,
                    "candidate_mem_key": mem_key,
                    "resolved_mem_key": mem_key_for_upsert,
                    "target_item_id": target_item_id,
                },
            )

        upsert_result = upsert_memory_item(
            db,
            user_id=user_id,
            kind=kind,
            mem_key=mem_key_for_upsert,
            value=value,
            confidence=confidence,
            importance=importance,
            source_session_id=session_id,
            source_episode_id=source_episode_id,
            source_note=source_note,
            evidence_span=evidence_span,
            source_turn_ids_json=source_turn_ids_json,
            embedding_model=emb_model,
            embedding_dims=emb_dims,
            embedding_blob=emb_blob,
            embedding_status=emb_status,
            last_embedding_error=emb_error,
            embedding_retry_count=emb_retry_count,
            embedding_last_attempt_ts=emb_last_attempt_ts,
            embedding_next_retry_ts=emb_next_retry_ts,
            target_item_id=target_item_id,
        )

        if upsert_result.action == "inserted":
            inserted += 1
        elif upsert_result.action == "updated":
            updated += 1

        tracer.emit(
            event="lt.item.upsert",
            level="info",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={
                "mode": mode,
                "episode_id": episode_id_for_trace,
                "item_id": upsert_result.item_id,
                "action": upsert_result.action,
                "kind": kind,
                "mem_key": mem_key,
                "confidence": confidence,
                "importance": importance,
                "source_turn_ids": valid_source_turn_ids,
                "embedding_status": emb_status,
            },
        )

        if vec:
            try:
                upsert_item_vec(db, upsert_result.item_id, vec)
                tracer.emit(
                    event="lt.item.vec.upsert",
                    level="debug",
                    correlation_id=correlation_id,
                    user_id=user_id,
                    session_id=session_id,
                    payload={
                        "mode": mode,
                        "episode_id": episode_id_for_trace,
                        "item_id": upsert_result.item_id,
                        "embedding_dims": int(len(vec)),
                    },
                )
            except Exception as e:
                tracer.emit(
                    event="lt.vec.upsert.error",
                    level="warning",
                    correlation_id=correlation_id,
                    user_id=user_id,
                    session_id=session_id,
                    payload={"item_id": upsert_result.item_id, "error": f"{type(e).__name__}: {e}"},
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
            "skipped": skipped,
            "considered": min(len(items), cfg.max_items),
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
        "You extract LONG-TERM memory items for an agent.\n"
        "SOURCE OF TRUTH: ST transcript below.\n"
        "MT summary is optional context only for prioritization and deduplication.\n"
        "Return STRICT JSON only. No markdown.\n"
        "Return a JSON array with items using this schema:\n"
        "{\n"
        '  "kind": "identity"|"preference"|"constraint"|"goal"|"procedure"|"project",\n'
        '  "key": string|null,\n'
        '  "value": string,\n'
        '  "confidence": number,\n'
        '  "importance": number,\n'
        '  "evidence_span": string,\n'
        '  "source_turn_ids": [integer]\n'
        "}\n"
        f"Max items: {max_items}.\n"
        f"Minimum confidence guidance: {min_confidence:.2f}.\n"
        "Definition of long-term memory:\n"
        "- Extract only information likely to remain useful across future sessions.\n"
        "- Valid items are stable identity facts, durable preferences, persistent constraints,\n"
        "  recurring goals, ongoing projects, and reusable procedures.\n"
        "- Do NOT extract temporary task state, one-off requests, short-lived plans,\n"
        "  transient debugging context, speculative statements, or discussion topics.\n"
        "Rules:\n"
        "- Evidence is mandatory for every item.\n"
        "- source_turn_ids MUST reference turn ids present in ST transcript.\n"
        "- If no evidence in ST supports an item, do not output it.\n"
        "- Prefer explicitly stated user facts over inferred facts.\n"
        "- If multiple candidate items are duplicates or near-duplicates, keep only one canonical item.\n"
        "- If multiple turns conflict, keep only the most explicit and most recent supported item.\n"
        "- Never output contradictory items.\n"
        "- Keep values canonical, concise, and standalone.\n"
        "- Avoid pronouns and context-dependent wording.\n"
        "- Write values in English.\n"
        "- key must be a stable snake_case identifier in English, or null if uncertain.\n"
        "Scoring guidance:\n"
        "- confidence reflects how directly and unambiguously the ST supports the item.\n"
        "- importance reflects how useful the item is for future sessions.\n"
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


def _read_existing_memory_index(db: sqlite3.Connection, *, user_id: str) -> Dict[str, Any]:
    rows = db.execute(
        """
        SELECT id, kind, mem_key
        FROM memory_items
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchall()

    by_id: Dict[int, Dict[str, Any]] = {}
    ids_by_kind: Dict[str, List[int]] = {}
    id_by_kind_key: Dict[tuple[str, str], int] = {}

    for r in rows:
        item_id = int(r["id"])
        kind = str(r["kind"] or "").strip()
        mem_key = r["mem_key"]
        mem_key_norm = str(mem_key).strip() if mem_key is not None else None
        if mem_key_norm == "":
            mem_key_norm = None

        by_id[item_id] = {"id": item_id, "kind": kind, "mem_key": mem_key_norm}
        ids_by_kind.setdefault(kind, []).append(item_id)
        if mem_key_norm is not None:
            id_by_kind_key[(kind, mem_key_norm)] = item_id

    return {
        "by_id": by_id,
        "ids_by_kind": ids_by_kind,
        "id_by_kind_key": id_by_kind_key,
    }


def _find_semantic_duplicate(
    *,
    db: sqlite3.Connection,
    user_id: str,
    kind: str,
    mem_key: Optional[str],
    value: str,
    vec: Sequence[float],
    cfg: LTConfig,
    existing_index: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not vec:
        return None

    mem_key_norm = str(mem_key).strip() if mem_key is not None else None
    if mem_key_norm == "":
        mem_key_norm = None

    # Exact key already handled by deterministic key-upsert.
    if mem_key_norm is not None and (kind, mem_key_norm) in existing_index.get("id_by_kind_key", {}):
        return None

    allowed_ids = list(existing_index.get("ids_by_kind", {}).get(kind, []))
    if not allowed_ids:
        return None

    try:
        neighbors = query_topk_item_ids(
            db,
            list(vec),
            k=max(1, int(cfg.semantic_dedupe_top_k)),
            allowed_ids=allowed_ids,
        )
    except Exception:
        # If vec table is unavailable, semantic dedupe is skipped gracefully.
        return None

    by_id = existing_index.get("by_id", {})
    for item_id, distance in neighbors:
        item = by_id.get(int(item_id))
        if not item:
            continue

        similarity = distance_to_similarity(float(distance))
        if similarity < float(cfg.semantic_dedupe_min_similarity):
            continue

        existing_mem_key = item.get("mem_key")
        if mem_key_norm is not None and existing_mem_key is not None and mem_key_norm == existing_mem_key:
            # This case should already be captured by key-upsert path.
            continue

        return {
            "item_id": int(item_id),
            "mem_key": existing_mem_key,
            "similarity": float(similarity),
            "distance": float(distance),
            "candidate_value_chars": len(value),
        }

    return None


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


def _item_skip_reason(
    *,
    kind: str,
    value: str,
    importance: float,
    min_importance: float,
    confidence: float,
    min_confidence: float,
    evidence_span: str,
    valid_source_turn_ids: Sequence[int],
) -> Optional[str]:
    if not kind or not value:
        return "missing_kind_or_value"
    if not evidence_span:
        return "missing_evidence_span"
    if not valid_source_turn_ids:
        return "missing_source_turn_ids"
    if importance < min_importance:
        return "importance_below_threshold"
    if confidence < min_confidence:
        return "confidence_below_threshold"
    return None


def _parse_items_json(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    s = text.strip()
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


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
