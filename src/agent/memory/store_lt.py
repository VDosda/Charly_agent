from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple


def upsert_memory_item(
    db: sqlite3.Connection,
    *,
    user_id: str,
    kind: str,
    mem_key: Optional[str],
    value: str,
    confidence: float,
    importance: float,
    source_session_id: Optional[str] = None,
    source_episode_id: Optional[int] = None,
    source_note: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dims: Optional[int] = None,
    embedding_blob: Optional[bytes] = None,
) -> int:
    """
    Upsert by (user_id, kind, mem_key) when mem_key is not null.
    Otherwise insert a new row.
    Returns item id.
    """
    now = int(time.time())

    with db:
        if mem_key is not None:
            row = db.execute(
                """
                SELECT id FROM memory_items
                WHERE user_id = ? AND kind = ? AND mem_key = ?
                """,
                (user_id, kind, mem_key),
            ).fetchone()

            if row:
                item_id = int(row["id"])
                db.execute(
                    """
                    UPDATE memory_items
                    SET value = ?,
                        ts_updated = ?,
                        last_seen_ts = ?,
                        confidence = ?,
                        importance = ?,
                        source_session_id = COALESCE(?, source_session_id),
                        source_episode_id = COALESCE(?, source_episode_id),
                        source_note = COALESCE(?, source_note),
                        embedding_model = COALESCE(?, embedding_model),
                        embedding_dims = COALESCE(?, embedding_dims),
                        embedding_blob = COALESCE(?, embedding_blob)
                    WHERE id = ?
                    """,
                    (
                        value,
                        now,
                        now,
                        float(confidence),
                        float(importance),
                        source_session_id,
                        source_episode_id,
                        source_note,
                        embedding_model,
                        embedding_dims,
                        embedding_blob,
                        item_id,
                    ),
                )
                return item_id

        cur = db.execute(
            """
            INSERT INTO memory_items (
                user_id, kind, mem_key, value,
                source_session_id, source_episode_id, source_note,
                ts_created, ts_updated, last_seen_ts,
                confidence, importance,
                embedding_model, embedding_dims, embedding_blob
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                kind,
                mem_key,
                value,
                source_session_id,
                source_episode_id,
                source_note,
                now,
                now,
                now,
                float(confidence),
                float(importance),
                embedding_model,
                embedding_dims,
                embedding_blob,
            ),
        )
        return int(cur.lastrowid)


def read_top_memory_items(
    db: sqlite3.Connection,
    *,
    user_id: str,
    limit: int = 20,
    min_importance: float = 0.0,
) -> List[Dict[str, Any]]:
    rows = db.execute(
        """
        SELECT id, kind, mem_key, value, confidence, importance, ts_updated
        FROM memory_items
        WHERE user_id = ? AND importance >= ?
        ORDER BY importance DESC, ts_updated DESC
        LIMIT ?
        """,
        (user_id, float(min_importance), int(limit)),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": int(r["id"]),
                "kind": r["kind"],
                "mem_key": r["mem_key"],
                "value": r["value"],
                "confidence": float(r["confidence"]),
                "importance": float(r["importance"]),
                "ts_updated": int(r["ts_updated"]),
            }
        )
    return out