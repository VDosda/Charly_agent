from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class UpsertMemoryItemResult:
    item_id: int
    action: str  # "inserted" | "updated"


@dataclass(frozen=True)
class ArchiveMemoryItemResult:
    item_id: int
    archived: bool


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except Exception:
        return default


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
                archived, archived_at, archived_reason,
                ts_created, ts_updated, last_seen_ts,
                versioned_ts, change_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _row_get(row, "archived"),
                _row_get(row, "archived_at"),
                _row_get(row, "archived_reason"),
                row["ts_created"],
                row["ts_updated"],
                row["last_seen_ts"],
                int(versioned_ts),
                change_reason,
            ),
        )
    except sqlite3.OperationalError as e:
        # Backward compatibility:
        # - if version table is missing, skip snapshots entirely;
        # - if archive columns are not present yet in versions table, use legacy INSERT.
        lowered = str(e).lower()
        if "no such table" in lowered and "memory_item_versions" in lowered:
            return
        if "table memory_item_versions has no column named archived" not in lowered:
            raise
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
            archived = 0,
            archived_at = NULL,
            archived_reason = NULL,
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
                  embedding_retry_count, embedding_last_attempt_ts, embedding_next_retry_ts,
                  archived, archived_at, archived_reason
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
                  embedding_retry_count, embedding_last_attempt_ts, embedding_next_retry_ts,
                  archived, archived_at, archived_reason
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
    where.append("archived = 0")
    if due_only:
        where.append("(embedding_next_retry_ts IS NULL OR embedding_next_retry_ts <= ?)")
        params.append(int(now_ts))
    where_sql = " AND ".join(where)

    try:
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
    except sqlite3.OperationalError:
        # Backward-compat fallback when archived column is not present yet.
        where = [w for w in where if w != "archived = 0"]
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


def _mode_clause(mode: str) -> Tuple[str, Tuple[Any, ...]]:
    normalized = (mode or "active").strip().lower()
    if normalized == "active":
        return "archived = 0", ()
    if normalized == "archive":
        return "archived = 1", ()
    if normalized == "all":
        return "1=1", ()
    raise ValueError(f"Unsupported memory mode: {mode}")


def list_memory_items(
    db: sqlite3.Connection,
    *,
    user_id: str,
    mode: str = "active",
    kind: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Read memory rows by lifecycle mode.
    - mode="active": only usable memories for normal LLM context.
    - mode="archive": explicitly archived memories for historical lookup.
    - mode="all": both active + archived.
    """
    mode_sql, mode_params = _mode_clause(mode)
    where = [mode_sql, "user_id = ?"]
    params: List[Any] = [*mode_params, user_id]
    if kind:
        where.append("kind = ?")
        params.append(kind)
    where_sql = " AND ".join(where)

    try:
        rows = db.execute(
            f"""
            SELECT
              id, user_id, kind, mem_key, value,
              confidence, importance, ts_created, ts_updated, last_seen_ts,
              source_session_id, source_episode_id, source_note,
              evidence_span, source_turn_ids_json,
              archived, archived_at, archived_reason
            FROM memory_items
            WHERE {where_sql}
            ORDER BY importance DESC, ts_updated DESC
            LIMIT ?
            """,
            (*params, int(limit)),
        ).fetchall()
    except sqlite3.OperationalError:
        if mode == "archive":
            return []
        legacy_where = ["user_id = ?"]
        legacy_params: List[Any] = [user_id]
        if kind:
            legacy_where.append("kind = ?")
            legacy_params.append(kind)
        legacy_where_sql = " AND ".join(legacy_where)
        rows = db.execute(
            f"""
            SELECT
              id, user_id, kind, mem_key, value,
              confidence, importance, ts_created, ts_updated, last_seen_ts,
              source_session_id, source_episode_id, source_note,
              evidence_span, source_turn_ids_json
            FROM memory_items
            WHERE {legacy_where_sql}
            ORDER BY importance DESC, ts_updated DESC
            LIMIT ?
            """,
            (*legacy_params, int(limit)),
        ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "user_id": r["user_id"],
                "kind": r["kind"],
                "mem_key": r["mem_key"],
                "value": r["value"],
                "confidence": float(r["confidence"]),
                "importance": float(r["importance"]),
                "ts_created": int(r["ts_created"]),
                "ts_updated": int(r["ts_updated"]),
                "last_seen_ts": int(r["last_seen_ts"]),
                "source_session_id": r["source_session_id"],
                "source_episode_id": int(r["source_episode_id"]) if r["source_episode_id"] is not None else None,
                "source_note": r["source_note"],
                "evidence_span": r["evidence_span"],
                "source_turn_ids_json": r["source_turn_ids_json"],
                "archived": 0,
                "archived_at": None,
                "archived_reason": None,
            }
            for r in rows
        ]

    return [
        {
            "id": int(r["id"]),
            "user_id": r["user_id"],
            "kind": r["kind"],
            "mem_key": r["mem_key"],
            "value": r["value"],
            "confidence": float(r["confidence"]),
            "importance": float(r["importance"]),
            "ts_created": int(r["ts_created"]),
            "ts_updated": int(r["ts_updated"]),
            "last_seen_ts": int(r["last_seen_ts"]),
            "source_session_id": r["source_session_id"],
            "source_episode_id": int(r["source_episode_id"]) if r["source_episode_id"] is not None else None,
            "source_note": r["source_note"],
            "evidence_span": r["evidence_span"],
            "source_turn_ids_json": r["source_turn_ids_json"],
            "archived": int(r["archived"] or 0),
            "archived_at": int(r["archived_at"]) if r["archived_at"] is not None else None,
            "archived_reason": r["archived_reason"],
        }
        for r in rows
    ]


def archive_memory_item_by_id(
    db: sqlite3.Connection,
    *,
    user_id: str,
    item_id: int,
    reason: str,
    source_session_id: Optional[str] = None,
    source_episode_id: Optional[int] = None,
    source_note: Optional[str] = None,
) -> ArchiveMemoryItemResult:
    """
    Logical archive only: memory is kept for history/audit but excluded from active retrieval.
    """
    now = int(time.time())
    with db:
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
              embedding_retry_count, embedding_last_attempt_ts, embedding_next_retry_ts,
              archived, archived_at, archived_reason
            FROM memory_items
            WHERE id = ? AND user_id = ?
            """,
            (int(item_id), user_id),
        ).fetchone()
        if not row:
            return ArchiveMemoryItemResult(item_id=int(item_id), archived=False)

        if int(_row_get(row, "archived", 0) or 0) == 1:
            return ArchiveMemoryItemResult(item_id=int(item_id), archived=False)

        _snapshot_memory_item_before_update(
            db,
            row=row,
            versioned_ts=now,
            change_reason="archive_memory_item",
        )
        db.execute(
            """
            UPDATE memory_items
            SET archived = 1,
                archived_at = ?,
                archived_reason = ?,
                ts_updated = ?,
                last_seen_ts = ?,
                source_session_id = COALESCE(?, source_session_id),
                source_episode_id = COALESCE(?, source_episode_id),
                source_note = COALESCE(?, source_note)
            WHERE id = ? AND user_id = ?
            """,
            (
                now,
                reason,
                now,
                now,
                source_session_id,
                source_episode_id,
                source_note,
                int(item_id),
                user_id,
            ),
        )
        return ArchiveMemoryItemResult(item_id=int(item_id), archived=True)


def archive_memory_item(
    db: sqlite3.Connection,
    *,
    user_id: str,
    mem_key: str,
    kind: Optional[str] = None,
    reason: str,
    source_session_id: Optional[str] = None,
    source_episode_id: Optional[int] = None,
    source_note: Optional[str] = None,
) -> ArchiveMemoryItemResult:
    where = ["user_id = ?", "mem_key = ?", "archived = 0"]
    params: List[Any] = [user_id, mem_key]
    if kind:
        where.append("kind = ?")
        params.append(kind)
    where_sql = " AND ".join(where)
    row = db.execute(
        f"""
        SELECT id
        FROM memory_items
        WHERE {where_sql}
        ORDER BY ts_updated DESC, id DESC
        LIMIT 1
        """,
        tuple(params),
    ).fetchone()
    if not row:
        return ArchiveMemoryItemResult(item_id=0, archived=False)

    return archive_memory_item_by_id(
        db,
        user_id=user_id,
        item_id=int(row["id"]),
        reason=reason,
        source_session_id=source_session_id,
        source_episode_id=source_episode_id,
        source_note=source_note,
    )


def read_top_memory_items(
    db: sqlite3.Connection,
    *,
    user_id: str,
    limit: int = 20,
    min_importance: float = 0.0,
    mode: str = "active",
) -> List[Dict[str, Any]]:
    mode_sql, mode_params = _mode_clause(mode)
    try:
        rows = db.execute(
            f"""
            SELECT id, kind, mem_key, value, confidence, importance, ts_updated
            FROM memory_items
            WHERE user_id = ? AND importance >= ? AND {mode_sql}
            ORDER BY importance DESC, ts_updated DESC
            LIMIT ?
            """,
            (user_id, float(min_importance), *mode_params, int(limit)),
        ).fetchall()
    except sqlite3.OperationalError:
        if mode == "archive":
            return []
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
