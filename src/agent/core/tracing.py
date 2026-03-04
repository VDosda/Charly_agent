from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


def new_correlation_id() -> str:
    return str(uuid.uuid4())


def _now_epoch_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TraceEvent:
    """
    Structured trace event.

    event: short machine-friendly name (e.g. "llm.request", "tool.result")
    level: "debug" | "info" | "warning" | "error"
    payload: arbitrary JSON-serializable dict
    """
    event: str
    level: str
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    ts_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms or _now_epoch_ms(),
            "level": self.level,
            "event": self.event,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "payload": self.payload or {},
        }


class JSONTracer:
    """
    Simple tracer that writes JSON lines to stdout.

    In production, this is easy to ship to:
    - journald/systemd
    - docker logs
    - fluentbit / loki / elk
    """

    def __init__(self, enabled: bool = True, stream=None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stdout

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

        ev = TraceEvent(
            event=event,
            level=level,
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload=payload or {},
            ts_ms=_now_epoch_ms(),
        )

        line = json.dumps(ev.to_dict(), ensure_ascii=False, separators=(",", ":"))
        self.stream.write(line + "\n")
        self.stream.flush()