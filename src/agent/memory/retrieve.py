from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple


def retrieve_mt_episodes(
    db: sqlite3.Connection,
    *,
    user_id: str,
    session_id: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    rows = db.execute(
        """
        SELECT id, summary, importance, confidence, ts, start_turn_id, end_turn_id
        FROM episodes
        WHERE user_id = ? AND session_id = ?
        ORDER BY ts DESC
        LIMIT ?
        """,
        (user_id, session_id, int(limit)),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": int(r["id"]),
                "summary": r["summary"],
                "importance": float(r["importance"]),
                "confidence": float(r["confidence"]),
                "ts": int(r["ts"]),
                "start_turn_id": int(r["start_turn_id"]),
                "end_turn_id": int(r["end_turn_id"]),
            }
        )
    return out


def retrieve_lt_items(
    db: sqlite3.Connection,
    *,
    user_id: str,
    limit: int = 8,
    min_importance: float = 0.5,
    mode: str = "active",
) -> List[Dict[str, Any]]:
    where_mode = _mode_sql(mode)
    try:
        rows = db.execute(
            f"""
            SELECT id, kind, mem_key, value, confidence, importance, ts_updated
            FROM memory_items
            WHERE user_id = ?
              AND importance >= ?
              AND {where_mode}
            ORDER BY importance DESC, ts_updated DESC
            LIMIT ?
            """,
            (user_id, float(min_importance), int(limit)),
        ).fetchall()
    except sqlite3.OperationalError:
        # Backward-compat when archived column is not present.
        if mode == "archive":
            return []
        rows = db.execute(
            """
            SELECT id, kind, mem_key, value, confidence, importance, ts_updated
            FROM memory_items
            WHERE user_id = ?
              AND importance >= ?
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


def _mode_sql(mode: str) -> str:
    normalized = (mode or "active").strip().lower()
    if normalized == "active":
        return "archived = 0"
    if normalized == "archive":
        return "archived = 1"
    if normalized == "all":
        return "1=1"
    raise ValueError(f"Unsupported LT retrieval mode: {mode}")


def render_lt_block(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    lines: List[str] = []
    for it in items:
        key = f" ({it['mem_key']})" if it.get("mem_key") else ""
        lines.append(
            f"- [{it['kind']}{key}] {it['value']} "
            f"(imp={it['importance']:.2f}, conf={it['confidence']:.2f})"
        )
    return "\n".join(lines)


def render_mt_block(episodes: List[Dict[str, Any]]) -> str:
    if not episodes:
        return ""
    lines: List[str] = []
    for ep in episodes:
        lines.append(
            f"- {ep['summary']} (imp={ep['importance']:.2f}, conf={ep['confidence']:.2f})"
        )
    return "\n".join(lines)
