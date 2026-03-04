from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Sequence, Tuple

from agent.memory.models import Episode, MemoryItem
from agent.memory.vector_store import embed_query, query_topk_episode_ids, query_topk_item_ids
from agent.providers.embeddings.base import EmbeddingProvider


def retrieve_mt_smart(
    db: sqlite3.Connection,
    embeddings: EmbeddingProvider,
    *,
    user_id: str,
    session_id: str,
    user_message: str,
    limit: int = 3,
) -> List[Tuple[Episode, float]]:
    qv = embed_query(embeddings, user_message)
    if not qv:
        return []

    scoped_episode_ids = _read_scoped_episode_ids(db, user_id=user_id, session_id=session_id)
    if not scoped_episode_ids:
        return []

    k = max(int(limit) * 3, 8)
    candidates = query_topk_episode_ids(db, qv, k=k, allowed_ids=scoped_episode_ids)
    if not candidates:
        return []

    ids = [episode_id for episode_id, _ in candidates]
    rows = db.execute(
        f"""
        SELECT
            id, user_id, session_id, start_turn_id, end_turn_id, ts, summary,
            topics_json, facts_json, open_tasks_json, importance, confidence
        FROM episodes
        WHERE user_id = ? AND session_id = ?
          AND id IN ({",".join("?" for _ in ids)})
        """,
        (user_id, session_id, *ids),
    ).fetchall()

    by_id = {int(r["id"]): r for r in rows}

    out: List[Tuple[Episode, float]] = []
    for episode_id, distance in candidates:
        row = by_id.get(int(episode_id))
        if row is None:
            continue

        out.append(
            (
                Episode(
                    id=int(row["id"]),
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    start_turn_id=int(row["start_turn_id"]),
                    end_turn_id=int(row["end_turn_id"]),
                    ts=int(row["ts"]),
                    summary=row["summary"],
                    topics=_coerce_str_list(_safe_json(row["topics_json"])),
                    facts=_coerce_dict_list(_safe_json(row["facts_json"])),
                    open_tasks=_coerce_dict_list(_safe_json(row["open_tasks_json"])),
                    importance=float(row["importance"]),
                    confidence=float(row["confidence"]),
                ),
                float(distance),
            )
        )

    return out


def retrieve_lt_smart(
    db: sqlite3.Connection,
    embeddings: EmbeddingProvider,
    *,
    user_id: str,
    user_message: str,
    limit: int = 6,
) -> List[Tuple[MemoryItem, float]]:
    qv = embed_query(embeddings, user_message)
    if not qv:
        return []

    scoped_item_ids = _read_scoped_item_ids(db, user_id=user_id)
    if not scoped_item_ids:
        return []

    k = max(int(limit) * 3, 14)
    candidates = query_topk_item_ids(db, qv, k=k, allowed_ids=scoped_item_ids)
    if not candidates:
        return []

    ids = [item_id for item_id, _ in candidates]
    rows = db.execute(
        f"""
        SELECT
            id, user_id, kind, mem_key, value,
            ts_created, ts_updated, last_seen_ts,
            importance, confidence,
            source_session_id, source_episode_id, source_note
        FROM memory_items
        WHERE user_id = ?
          AND id IN ({",".join("?" for _ in ids)})
        """,
        (user_id, *ids),
    ).fetchall()

    by_id = {int(r["id"]): r for r in rows}

    out: List[Tuple[MemoryItem, float]] = []
    for item_id, distance in candidates:
        row = by_id.get(int(item_id))
        if row is None:
            continue

        out.append(
            (
                MemoryItem(
                    id=int(row["id"]),
                    user_id=row["user_id"],
                    kind=row["kind"],
                    mem_key=row["mem_key"],
                    value=row["value"],
                    ts_created=int(row["ts_created"]),
                    ts_updated=int(row["ts_updated"]),
                    last_seen_ts=int(row["last_seen_ts"]),
                    importance=float(row["importance"]),
                    confidence=float(row["confidence"]),
                    source_session_id=row["source_session_id"],
                    source_episode_id=int(row["source_episode_id"]) if row["source_episode_id"] is not None else None,
                    source_note=row["source_note"],
                ),
                float(distance),
            )
        )

    return out


def _safe_json(text: Any) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _coerce_str_list(value: Any) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(v) for v in value]


def _coerce_dict_list(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    out: List[Dict[str, str]] = []
    for v in value:
        if not isinstance(v, dict):
            continue
        entry: Dict[str, str] = {}
        for k, vv in v.items():
            entry[str(k)] = str(vv)
        out.append(entry)
    return out


def _read_scoped_episode_ids(
    db: sqlite3.Connection,
    *,
    user_id: str,
    session_id: str,
) -> List[int]:
    rows = db.execute(
        """
        SELECT id
        FROM episodes
        WHERE user_id = ? AND session_id = ?
        """,
        (user_id, session_id),
    ).fetchall()
    return [int(r["id"]) for r in rows]


def _read_scoped_item_ids(
    db: sqlite3.Connection,
    *,
    user_id: str,
) -> List[int]:
    rows = db.execute(
        """
        SELECT id
        FROM memory_items
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchall()
    return [int(r["id"]) for r in rows]
