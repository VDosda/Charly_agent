from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple


def get_last_episode_end_turn_id(db: sqlite3.Connection, session_id: str) -> int:
    row = db.execute(
        """
        SELECT COALESCE(MAX(end_turn_id), 0)
        FROM episodes
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return int(row[0] or 0)


def read_turns_range(
    db: sqlite3.Connection,
    session_id: str,
    start_turn_id: int,
    end_turn_id: int,
) -> List[Dict[str, Any]]:
    rows = db.execute(
        """
        SELECT turn_id, role, content, tool_name
        FROM chat_history
        WHERE session_id = ?
          AND turn_id BETWEEN ? AND ?
        ORDER BY turn_id ASC
        """,
        (session_id, start_turn_id, end_turn_id),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "turn_id": int(r["turn_id"]),
                "role": r["role"],
                "content": r["content"],
                "tool_name": r["tool_name"],
            }
        )
    return out


def count_turns_after(
    db: sqlite3.Connection,
    session_id: str,
    after_turn_id: int,
) -> int:
    row = db.execute(
        """
        SELECT COUNT(*)
        FROM chat_history
        WHERE session_id = ?
          AND turn_id > ?
        """,
        (session_id, after_turn_id),
    ).fetchone()
    return int(row[0] or 0)


def get_max_turn_id(db: sqlite3.Connection, session_id: str) -> int:
    row = db.execute(
        """
        SELECT COALESCE(MAX(turn_id), 0)
        FROM chat_history
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return int(row[0] or 0)


def insert_episode(
    db: sqlite3.Connection,
    *,
    user_id: str,
    session_id: str,
    start_turn_id: int,
    end_turn_id: int,
    ts: int,
    summary: str,
    topics_json: Optional[str],
    facts_json: Optional[str],
    open_tasks_json: Optional[str],
    importance: float,
    confidence: float,
    embedding_model: Optional[str],
    embedding_dims: Optional[int],
    embedding_blob: Optional[bytes],
    source_turn_ids: List[int],
) -> int:
    """
    Insert episode + its source mapping in a single transaction.
    Returns the new episode_id.
    """
    with db:
        cur = db.execute(
            """
            INSERT INTO episodes (
                user_id, session_id,
                start_turn_id, end_turn_id,
                ts, summary,
                topics_json, facts_json, open_tasks_json,
                importance, confidence,
                embedding_model, embedding_dims, embedding_blob
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_id,
                start_turn_id,
                end_turn_id,
                ts,
                summary,
                topics_json,
                facts_json,
                open_tasks_json,
                float(importance),
                float(confidence),
                embedding_model,
                embedding_dims,
                embedding_blob,
            ),
        )
        episode_id = int(cur.lastrowid)

        for tid in source_turn_ids:
            db.execute(
                """
                INSERT OR IGNORE INTO episode_sources (episode_id, session_id, turn_id)
                VALUES (?, ?, ?)
                """,
                (episode_id, session_id, int(tid)),
            )

    return episode_id