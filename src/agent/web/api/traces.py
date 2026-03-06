from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from agent.db.conn import get_db  # adapt to your project: returns sqlite3.Connection

router = APIRouter(tags=["traces"])


@router.get("/correlations")
def list_correlations(
    limit: int = Query(50, ge=1, le=500),
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns recent correlation_ids (requests) with start/end timestamps and counts.
    """
    where = []
    params: List[Any] = []

    if user_id:
        where.append("user_id = ?")
        params.append(user_id)
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    rows = db.execute(
        f"""
        SELECT
          correlation_id,
          MIN(ts_ms) AS ts_start,
          MAX(ts_ms) AS ts_end,
          COUNT(*) AS events
        FROM trace_events
        {where_sql}
        GROUP BY correlation_id
        ORDER BY ts_end DESC
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()

    out = []
    for r in rows:
        out.append(
            {
                "correlation_id": r["correlation_id"],
                "ts_start": int(r["ts_start"]),
                "ts_end": int(r["ts_end"]),
                "events": int(r["events"]),
            }
        )
    return {"items": out}


@router.get("/correlation/{correlation_id}")
def get_trace_by_correlation(
    correlation_id: str,
    db: sqlite3.Connection = Depends(get_db),
) -> Dict[str, Any]:
    rows = db.execute(
        """
        SELECT ts_ms, level, event, correlation_id, user_id, session_id, payload_json
        FROM trace_events
        WHERE correlation_id = ?
        ORDER BY ts_ms ASC
        """,
        (correlation_id,),
    ).fetchall()

    items = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
        except Exception:
            payload = {"_raw": r["payload_json"]}

        items.append(
            {
                "ts_ms": int(r["ts_ms"]),
                "level": r["level"],
                "event": r["event"],
                "correlation_id": r["correlation_id"],
                "user_id": r["user_id"],
                "session_id": r["session_id"],
                "payload": payload,
            }
        )
    return {"items": items}


@router.get("/logs")
def list_logs(
    limit: int = Query(200, ge=1, le=1000),
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns trace events as plain log lines ordered from newest to oldest.
    """
    where = []
    params: List[Any] = []

    if user_id:
        where.append("user_id = ?")
        params.append(user_id)
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
    if correlation_id:
        where.append("correlation_id = ?")
        params.append(correlation_id)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    rows = db.execute(
        f"""
        SELECT ts_ms, level, event, correlation_id, user_id, session_id, payload_json
        FROM trace_events
        {where_sql}
        ORDER BY ts_ms DESC
        LIMIT ?
        """,
        (*params, int(limit)),
    ).fetchall()

    items: List[Dict[str, Any]] = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
        except Exception:
            payload = {"_raw": r["payload_json"]}

        items.append(
            {
                "ts_ms": int(r["ts_ms"]),
                "level": r["level"],
                "event": r["event"],
                "correlation_id": r["correlation_id"],
                "user_id": r["user_id"],
                "session_id": r["session_id"],
                "payload": payload,
            }
        )

    return {"items": items}
