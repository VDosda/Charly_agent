from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional

from agent.core.tracing import JSONTracer, TraceEvent, _now_epoch_ms


class DBTracer:
    """
    Writes TraceEvent to SQLite.
    Intended to be used alongside JSONTracer (stdout).
    """

    def __init__(self, db: sqlite3.Connection, enabled: bool = True) -> None:
        self.db = db
        self.enabled = enabled

    def emit(
        self,
        event: str,
        level: str,
        correlation_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        ts_ms = _now_epoch_ms()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))

        # Keep writes cheap and safe
        try:
            with self.db:
                self.db.execute(
                    """
                    INSERT INTO trace_events(ts_ms, level, event, correlation_id, user_id, session_id, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ts_ms, level, event, correlation_id, user_id, session_id, payload_json),
                )
        except Exception:
            # Never crash the agent because tracing failed
            return


class MultiTracer:
    """
    Fan-out tracer: emit to multiple sinks (stdout JSON + SQLite).
    """

    def __init__(self, *tracers) -> None:
        self.tracers = [t for t in tracers if t is not None]

    def emit(self, *args, **kwargs) -> None:
        for t in self.tracers:
            try:
                t.emit(*args, **kwargs)
            except Exception:
                continue