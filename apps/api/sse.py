from __future__ import annotations

import json
from typing import Any, Literal


SSEEventName = Literal["start", "delta", "end", "error"]
ALLOWED_SSE_EVENTS: set[str] = {"start", "delta", "end", "error"}


def encode_sse_event(event: SSEEventName, data: Any) -> str:
    if event not in ALLOWED_SSE_EVENTS:
        raise ValueError(f"Unsupported SSE event: {event}")

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"
