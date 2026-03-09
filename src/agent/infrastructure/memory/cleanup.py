from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class CleanupConfig:
    # keep last N turns not archived for each session
    keep_last_turns: int = 120
    # hard delete archived rows older than this many turns (optional)
    hard_delete_archived: bool = False


def cleanup_st(
    db: sqlite3.Connection,
    *,
    session_id: str,
    cfg: CleanupConfig | None = None,
) -> None:
    cfg = cfg or CleanupConfig()

    # Determine current max turn_id
    row = db.execute(
        "SELECT COALESCE(MAX(turn_id), 0) FROM chat_history WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    max_turn = int(row[0] or 0)

    cutoff = max_turn - cfg.keep_last_turns
    if cutoff <= 0:
        return

    with db:
        db.execute(
            """
            UPDATE chat_history
            SET archived = 1
            WHERE session_id = ?
              AND turn_id <= ?
            """,
            (session_id, cutoff),
        )

        if cfg.hard_delete_archived:
            db.execute(
                """
                DELETE FROM chat_history
                WHERE session_id = ?
                  AND archived = 1
                """,
                (session_id,),
            )