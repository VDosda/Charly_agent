from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from agent.providers.llm.base import ToolCall, ToolSpec


def tool_specs_to_openai(tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.json_schema,
            },
        }
        for t in tools
    ]


def normalize_tool_choice(tool_choice: Optional[str]) -> Optional[Any]:
    """
    Provider-agnostic tool_choice normalization.

    Common inputs:
    - None => provider default
    - "auto"
    - "none"
    - "<tool_name>"

    Each provider has its own schema; we map in provider implementations.
    """
    return tool_choice


def safe_json_loads(s: Any) -> Dict[str, Any]:
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # If provider gives malformed JSON, keep it as raw text.
            return {"_raw": s}
    return {"_raw": str(s)}


def new_tool_call_id() -> str:
    return str(uuid.uuid4())


def openai_tool_calls_to_normalized(tool_calls: Any) -> List[ToolCall]:
    """
    Normalize OpenAI-style tool calls (Responses or Chat Completions variants)
    to our ToolCall objects.

    We accept a broad shape to reduce brittleness across SDK versions.
    """
    out: List[ToolCall] = []
    if not tool_calls:
        return out

    for tc in tool_calls:
        # Possible keys: id, function{name, arguments}, etc.
        tc_id = getattr(tc, "id", None) or tc.get("id") if isinstance(tc, dict) else None
        fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
        name = getattr(fn, "name", None) if fn is not None else None
        if isinstance(fn, dict):
            name = fn.get("name", name)
        args = getattr(fn, "arguments", None) if fn is not None else None
        if isinstance(fn, dict):
            args = fn.get("arguments", args)

        out.append(
            ToolCall(
                id=tc_id or new_tool_call_id(),
                name=name or "unknown_tool",
                arguments=safe_json_loads(args),
            )
        )
    return out