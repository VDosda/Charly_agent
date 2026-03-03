from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence
from urllib import error, request

from agent.providers.llm.base import LLMProvider, LLMResult, ToolCall, ToolSpec
from agent.providers.llm._tooling import new_tool_call_id, safe_json_loads

# Notes:
# - Ollama's tool calling behavior depends on the model and Ollama version.
# - This implementation uses HTTP directly to avoid extra dependencies.
# - If your model doesn't support tools, tool_calls will be empty.


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def supports_tools(self) -> bool:
        # Some Ollama models support tools; we return True but runtime should handle empty tool_calls.
        return True

    def generate(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResult:
        """
        Calls Ollama /api/chat.

        Expected request shape (Ollama):
        {
          "model": "...",
          "messages": [{"role":"user","content":"..."}],
          "stream": false,
          "options": {"temperature": 0.2, "num_predict": ...},
          "tools": [...]   # if supported
        }
        """
        url = f"{self.base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                # Ollama uses num_predict as max tokens for generation
                "num_predict": self.max_tokens,
            },
        }

        # Best-effort tool schema mapping (Ollama expects a function-like spec in some versions)
        if tools:
            payload["tools"] = [
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

        # tool_choice is not consistently supported; ignore safely
        _ = tool_choice

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                detail = ""
            if detail:
                raise RuntimeError(
                    f"Failed calling Ollama at {url}: HTTP {e.code} {e.reason}. Response body: {detail}"
                ) from e
            raise RuntimeError(f"Failed calling Ollama at {url}: HTTP {e.code} {e.reason}") from e
        except Exception as e:
            raise RuntimeError(f"Failed calling Ollama at {url}: {e}") from e

        obj = json.loads(raw)

        # Typical response:
        # {"message": {"role": "assistant", "content": "...", "tool_calls":[...]}, ...}
        msg = obj.get("message") or {}
        text = msg.get("content") or ""

        tool_calls: List[ToolCall] = []
        for tc in (msg.get("tool_calls") or []):
            # shape varies; best effort
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or "unknown_tool"
            args = fn.get("arguments") or tc.get("arguments") or {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id") or new_tool_call_id(),
                    name=name,
                    arguments=safe_json_loads(args),
                )
            )

        return LLMResult(text=text, tool_calls=tool_calls, usage=None, raw=obj)
