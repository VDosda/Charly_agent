from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class UpsertMemoryItemResult:
    item_id: int
    action: str  # "inserted" | "updated"


def _snapshot_memory_item_before_update(
    db: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    versioned_ts: int,
    change_reason: str,
) -> None:
    """
    Persist a snapshot of the current row before UPDATE.
    If version table is not available yet, skip without failing the write path.
    """
    try:
        db.execute(
            """
            INSERT INTO memory_item_versions (
                memory_item_id, user_id, kind, mem_key, value,
                confidence, importance,
                source_session_id, source_episode_id, source_note,
                evidence_span, source_turn_ids_json,
                embedding_model, embedding_dims, embedding_status, last_embedding_error,
                embedding_retry_count, embedding_last_attempt_ts, embedding_next_retry_ts,
                ts_created, ts_updated, last_seen_ts,
                versioned_ts, change_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["id"]),
                row["user_id"],
                row["kind"],
                row["mem_key"],
                row["value"],
                row["confidence"],
                row["importance"],
                row["source_session_id"],
                row["source_episode_id"],
                row["source_note"],
                row["evidence_span"],
                row["source_turn_ids_json"],
                row["embedding_model"],
                row["embedding_dims"],
                row["embedding_status"],
                row["last_embedding_error"],
                row["embedding_retry_count"],
                row["embedding_last_attempt_ts"],
                row["embedding_next_retry_ts"],
                row["ts_created"],
                row["ts_updated"],
                row["last_seen_ts"],
                int(versioned_ts),
                change_reason,
            ),
        )
    except sqlite3.OperationalError as e:
        # Backward compatibility if version table is not present.
        lowered = str(e).lower()
        if "no such table" in lowered and "memory_item_versions" in lowered:
            return
        raise


def _update_memory_item(
    db: sqlite3.Connection,
    *,
    item_id: int,
    now: int,
    value: str,
    confidence: float,
    importance: float,
    source_session_id: Optional[str],
    source_episode_id: Optional[int],
    source_note: Optional[str],
    evidence_span: Optional[str],
    source_turn_ids_json: Optional[str],
    embedding_model: Optional[str],
    embedding_dims: Optional[int],
    embedding_blob: Optional[bytes],
    embedding_status: Optional[str],
    last_embedding_error: Optional[str],
    embedding_retry_count: Optional[int],
    embedding_last_attempt_ts: Optional[int],
    embedding_next_retry_ts: Optional[int],
) -> None:
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
            evidence_span = ?,
            source_turn_ids_json = ?,
            embedding_model = ?,
            embedding_dims = ?,
            embedding_blob = ?,
            embedding_status = COALESCE(?, embedding_status),
            last_embedding_error = ?,
            embedding_retry_count = COALESCE(?, embedding_retry_count),
            embedding_last_attempt_ts = ?,
            embedding_next_retry_ts = ?
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
            evidence_span,
            source_turn_ids_json,
            embedding_model,
            embedding_dims,
            embedding_blob,
            embedding_status,
            last_embedding_error,
            embedding_retry_count,
            embedding_last_attempt_ts,
            embedding_next_retry_ts,
            int(item_id),
        ),
    )


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
    evidence_span: Optional[str] = None,
    source_turn_ids_json: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dims: Optional[int] = None,
    embedding_blob: Optional[bytes] = None,
    embedding_status: Optional[str] = None,
    last_embedding_error: Optional[str] = None,
    embedding_retry_count: Optional[int] = None,
    embedding_last_attempt_ts: Optional[int] = None,
    embedding_next_retry_ts: Optional[int] = None,
    target_item_id: Optional[int] = None,
) -> UpsertMemoryItemResult:
    """
    If target_item_id is provided, update that item first (semantic merge path).
    Else upsert by (user_id, kind, mem_key) when mem_key is not null.
    Otherwise insert a new row.
    Returns item id + action (inserted/updated).
    """
    now = int(time.time())

    with db:
        if target_item_id is not None:
            row = db.execute(
                """
                SELECT
                  id, user_id, kind, mem_key, value,
                  source_session_id, source_episode_id, source_note,
                  ts_created, ts_updated, last_seen_ts,
                  confidence, importance,
                  embedding_model, embedding_dims, embedding_blob,
                  evidence_span, source_turn_ids_json,
                  embedding_status, last_embedding_error,
                  embedding_retry_count, embedding_last_attempt_ts, embedding_next_retry_ts
                FROM memory_items
                WHERE id = ? AND user_id = ?
                """,
                (int(target_item_id), user_id),
            ).fetchone()
            if row:
                item_id = int(row["id"])
                _snapshot_memory_item_before_update(
                    db,
                    row=row,
                    versioned_ts=now,
                    change_reason="upsert_target_item_id",
                )
                _update_memory_item(
                    db,
                    item_id=item_id,
                    now=now,
                    value=value,
                    confidence=confidence,
                    importance=importance,
                    source_session_id=source_session_id,
                    source_episode_id=source_episode_id,
                    source_note=source_note,
                    evidence_span=evidence_span,
                    source_turn_ids_json=source_turn_ids_json,
                    embedding_model=embedding_model,
                    embedding_dims=embedding_dims,
                    embedding_blob=embedding_blob,
                    embedding_status=embedding_status,
                    last_embedding_error=last_embedding_error,
                    embedding_retry_count=embedding_retry_count,
                    embedding_last_attempt_ts=embedding_last_attempt_ts,
                    embedding_next_retry_ts=embedding_next_retry_ts,
                )
                return UpsertMemoryItemResult(item_id=item_id, action="updated")

        if mem_key is not None:
            row = db.execute(
                """
                SELECT
                  id, user_id, kind, mem_key, value,
                  source_session_id, source_episode_id, source_note,
                  ts_created, ts_updated, last_seen_ts,
                  confidence, importance,
                  embedding_model, embedding_dims, embedding_blob,
                  evidence_span, source_turn_ids_json,
                  embedding_status, last_embedding_error,
                  embedding_retry_count, embedding_last_attempt_ts, embedding_next_retry_ts
                FROM memory_items
                WHERE user_id = ? AND kind = ? AND mem_key = ?
                """,
                (user_id, kind, mem_key),
            ).fetchone()

            if row:
                item_id = int(row["id"])
                _snapshot_memory_item_before_update(
                    db,
                    row=row,
                    versioned_ts=now,
                    change_reason="upsert_by_mem_key",
                )
                _update_memory_item(
                    db,
                    item_id=item_id,
                    now=now,
                    value=value,
                    confidence=confidence,
                    importance=importance,
                    source_session_id=source_session_id,
                    source_episode_id=source_episode_id,
                    source_note=source_note,
                    evidence_span=evidence_span,
                    source_turn_ids_json=source_turn_ids_json,
                    embedding_model=embedding_model,
                    embedding_dims=embedding_dims,
                    embedding_blob=embedding_blob,
                    embedding_status=embedding_status,
                    last_embedding_error=last_embedding_error,
                    embedding_retry_count=embedding_retry_count,
                    embedding_last_attempt_ts=embedding_last_attempt_ts,
                    embedding_next_retry_ts=embedding_next_retry_ts,
                )
                return UpsertMemoryItemResult(item_id=item_id, action="updated")

        cur = db.execute(
            """
            INSERT INTO memory_items (
                user_id, kind, mem_key, value,
                source_session_id, source_episode_id, source_note,
                evidence_span, source_turn_ids_json,
                ts_created, ts_updated, last_seen_ts,
                confidence, importance,
                embedding_model, embedding_dims, embedding_blob,
                embedding_status, last_embedding_error,
                embedding_retry_count, embedding_last_attempt_ts, embedding_next_retry_ts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                kind,
                mem_key,
                value,
                source_session_id,
                source_episode_id,
                source_note,
                evidence_span,
                source_turn_ids_json,
                now,
                now,
                now,
                float(confidence),
                float(importance),
                embedding_model,
                embedding_dims,
                embedding_blob,
                embedding_status or "pending",
                last_embedding_error,
                int(embedding_retry_count or 0),
                embedding_last_attempt_ts,
                embedding_next_retry_ts,
            ),
        )
        return UpsertMemoryItemResult(item_id=int(cur.lastrowid), action="inserted")


def list_embedding_retry_candidates(
    db: sqlite3.Connection,
    *,
    now_ts: int,
    limit: int = 20,
    user_id: Optional[str] = None,
    due_only: bool = True,
) -> List[Dict[str, Any]]:
    where = ["embedding_status IN ('pending', 'failed')"]
    params: List[Any] = []
    if user_id:
        where.append("user_id = ?")
        params.append(user_id)
    if due_only:
        where.append("(embedding_next_retry_ts IS NULL OR embedding_next_retry_ts <= ?)")
        params.append(int(now_ts))
    where_sql = " AND ".join(where)

    rows = db.execute(
        f"""
        SELECT
          id, user_id, kind, mem_key, value,
          embedding_retry_count, embedding_status
        FROM memory_items
        WHERE {where_sql}
        ORDER BY COALESCE(embedding_next_retry_ts, 0) ASC, id ASC
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()

    return [
        {
            "id": int(r["id"]),
            "user_id": r["user_id"],
            "kind": r["kind"],
            "mem_key": r["mem_key"],
            "value": r["value"],
            "embedding_retry_count": int(r["embedding_retry_count"] or 0),
            "embedding_status": r["embedding_status"],
        }
        for r in rows
    ]


def mark_embedding_retry_success(
    db: sqlite3.Connection,
    *,
    item_id: int,
    embedding_model: str,
    embedding_dims: int,
    embedding_blob: bytes,
    ts_now: int,
) -> None:
    with db:
        db.execute(
            """
            UPDATE memory_items
            SET embedding_model = ?,
                embedding_dims = ?,
                embedding_blob = ?,
                embedding_status = 'ready',
                last_embedding_error = NULL,
                embedding_last_attempt_ts = ?,
                embedding_next_retry_ts = NULL
            WHERE id = ?
            """,
            (embedding_model, int(embedding_dims), embedding_blob, int(ts_now), int(item_id)),
        )


def mark_embedding_retry_failure(
    db: sqlite3.Connection,
    *,
    item_id: int,
    error: str,
    ts_now: int,
    next_retry_ts: int,
) -> None:
    with db:
        db.execute(
            """
            UPDATE memory_items
            SET embedding_status = 'failed',
                last_embedding_error = ?,
                embedding_retry_count = COALESCE(embedding_retry_count, 0) + 1,
                embedding_last_attempt_ts = ?,
                embedding_next_retry_ts = ?
            WHERE id = ?
            """,
            (error, int(ts_now), int(next_retry_ts), int(item_id)),
        )


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
