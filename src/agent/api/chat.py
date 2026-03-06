from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib import error, request

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agent.providers.llm._tooling import (
    new_tool_call_id,
    normalize_tool_choice,
    safe_json_loads,
    tool_specs_to_openai,
)
from agent.providers.llm.base import LLMResult, ToolCall, ToolSpec
from agent.api.dependencies import RuntimeContext, get_runtime_context_dep
from agent.api.schemas import ChatStreamRequest
from agent.api.sse import encode_sse_event


router = APIRouter(prefix="/api/chat", tags=["chat"])


class StreamingLLMProxy:
    """
    Thin adapter used only by the API layer.
    It preserves runtime behavior while enabling token deltas when the provider supports streaming.
    """

    def __init__(self, llm: Any, on_delta: Callable[[str], None]) -> None:
        self._llm = llm
        self._on_delta = on_delta

    def supports_tools(self) -> bool:
        return bool(self._llm.supports_tools())

    def generate(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResult:
        stream_generate = getattr(self._llm, "stream_generate", None)
        if callable(stream_generate):
            result = stream_generate(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                on_delta=self._emit_delta,
            )
            if isinstance(result, LLMResult):
                return result
            raise TypeError("stream_generate must return LLMResult")

        module_name = self._llm.__class__.__module__

        if module_name.endswith(".providers.llm.openai"):
            return self._generate_openai(messages=messages, tools=tools, tool_choice=tool_choice)
        if module_name.endswith(".providers.llm.anthropic"):
            return self._generate_anthropic(messages=messages, tools=tools, tool_choice=tool_choice)
        if module_name.endswith(".providers.llm.ollama"):
            return self._generate_ollama(messages=messages, tools=tools, tool_choice=tool_choice)

        # Provider has no known streaming API; keep behavior unchanged.
        return self._llm.generate(messages=messages, tools=tools, tool_choice=tool_choice)

    def _emit_delta(self, text: str) -> None:
        if text:
            self._on_delta(text)

    def _generate_openai(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[ToolSpec]],
        tool_choice: Optional[str],
    ) -> LLMResult:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("OpenAI SDK not installed for streaming mode.") from exc

        client = OpenAI(
            api_key=getattr(self._llm, "api_key", None),
            base_url=getattr(self._llm, "base_url", None),
        )
        oai_tools = tool_specs_to_openai(tools)
        normalized = normalize_tool_choice(tool_choice)

        oai_tool_choice: Any = None
        if normalized in (None, "auto"):
            oai_tool_choice = "auto"
        elif normalized == "none":
            oai_tool_choice = "none"
        else:
            oai_tool_choice = {"type": "function", "function": {"name": str(normalized)}}

        stream = client.chat.completions.create(
            model=getattr(self._llm, "model"),
            messages=list(messages),
            temperature=getattr(self._llm, "temperature", 0.2),
            max_tokens=getattr(self._llm, "max_tokens", 2048),
            tools=oai_tools,
            tool_choice=oai_tool_choice,
            stream=True,
        )

        text_parts: List[str] = []
        pending_tool_calls: Dict[int, Dict[str, str]] = {}

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if isinstance(content, str):
                    self._emit_delta(content)
                    text_parts.append(content)
                else:
                    for part in content:
                        token = getattr(part, "text", None) or ""
                        if token:
                            self._emit_delta(token)
                            text_parts.append(token)

            for tc in (getattr(delta, "tool_calls", None) or []):
                idx = int(getattr(tc, "index", 0) or 0)
                current = pending_tool_calls.setdefault(
                    idx,
                    {"id": "", "name": "", "arguments": ""},
                )

                tc_id = getattr(tc, "id", None)
                if tc_id:
                    current["id"] = str(tc_id)

                fn = getattr(tc, "function", None)
                if fn is None:
                    continue

                fn_name = getattr(fn, "name", None)
                if fn_name:
                    current["name"] = str(fn_name)

                fn_args = getattr(fn, "arguments", None)
                if fn_args:
                    current["arguments"] += str(fn_args)

        tool_calls: List[ToolCall] = []
        for idx in sorted(pending_tool_calls):
            raw = pending_tool_calls[idx]
            tool_calls.append(
                ToolCall(
                    id=raw["id"] or new_tool_call_id(),
                    name=raw["name"] or "unknown_tool",
                    arguments=safe_json_loads(raw["arguments"] or "{}"),
                )
            )

        return LLMResult(text="".join(text_parts), tool_calls=tool_calls, usage=None, raw=None)

    def _generate_anthropic(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[ToolSpec]],
        tool_choice: Optional[str],
    ) -> LLMResult:
        try:
            import anthropic  # type: ignore
        except Exception as exc:
            raise RuntimeError("Anthropic SDK not installed for streaming mode.") from exc

        client = anthropic.Anthropic(
            api_key=getattr(self._llm, "api_key", None),
            base_url=getattr(self._llm, "base_url", None),
        )

        system_text = ""
        anthropic_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_text += (msg.get("content") or "") + "\n"
            else:
                anthropic_messages.append({"role": role, "content": msg.get("content")})

        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.json_schema,
                }
                for t in tools
            ]

        anthropic_tool_choice: Any = None
        if tool_choice in (None, "auto"):
            anthropic_tool_choice = {"type": "auto"}
        elif tool_choice == "none":
            anthropic_tool_choice = {"type": "none"}
        else:
            anthropic_tool_choice = {"type": "tool", "name": str(tool_choice)}

        text_parts: List[str] = []

        with client.messages.stream(
            model=getattr(self._llm, "model"),
            system=system_text.strip() or None,
            messages=anthropic_messages,
            temperature=getattr(self._llm, "temperature", 0.2),
            max_tokens=getattr(self._llm, "max_tokens", 2048),
            tools=anthropic_tools,
            tool_choice=anthropic_tool_choice,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    self._emit_delta(text)
                    text_parts.append(text)
            final_message = stream.get_final_message()

        tool_calls: List[ToolCall] = []
        for block in final_message.content:
            btype = getattr(block, "type", None) or block.get("type")
            if btype != "tool_use":
                continue

            name = getattr(block, "name", None) or block.get("name") or "unknown_tool"
            tc_input = getattr(block, "input", None) or block.get("input") or {}
            tc_id = getattr(block, "id", None) or block.get("id") or new_tool_call_id()
            tool_calls.append(ToolCall(id=tc_id, name=name, arguments=tc_input))

        return LLMResult(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            usage=None,
            raw=final_message,
        )

    def _generate_ollama(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[ToolSpec]],
        tool_choice: Optional[str],
    ) -> LLMResult:
        _ = tool_choice
        base_url = str(getattr(self._llm, "base_url", "http://localhost:11434")).rstrip("/")
        url = f"{base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": getattr(self._llm, "model"),
            "messages": list(messages),
            "stream": True,
            "options": {
                "temperature": getattr(self._llm, "temperature", 0.2),
                "num_predict": getattr(self._llm, "max_tokens", 2048),
            },
        }

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

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"})

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        try:
            with request.urlopen(req, timeout=180) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    chunk = json.loads(line)
                    message = chunk.get("message") or {}
                    content = message.get("content") or ""

                    if content:
                        self._emit_delta(content)
                        text_parts.append(content)

                    raw_tool_calls = message.get("tool_calls") or []
                    if raw_tool_calls:
                        tool_calls = self._parse_ollama_tool_calls(raw_tool_calls)
        except error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                detail = ""
            if detail:
                raise RuntimeError(
                    f"Failed calling Ollama stream at {url}: HTTP {exc.code} {exc.reason}. "
                    f"Response body: {detail}"
                ) from exc
            raise RuntimeError(
                f"Failed calling Ollama stream at {url}: HTTP {exc.code} {exc.reason}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed calling Ollama stream at {url}: {exc}") from exc

        return LLMResult(text="".join(text_parts), tool_calls=tool_calls, usage=None, raw=None)

    def _parse_ollama_tool_calls(self, raw_tool_calls: Sequence[Dict[str, Any]]) -> List[ToolCall]:
        parsed: List[ToolCall] = []
        for tc in raw_tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or "unknown_tool"
            args = fn.get("arguments") or tc.get("arguments") or {}
            parsed.append(
                ToolCall(
                    id=tc.get("id") or new_tool_call_id(),
                    name=name,
                    arguments=safe_json_loads(args),
                )
            )
        return parsed


def _stream_headers() -> Dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


@router.post("/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    context: RuntimeContext = Depends(get_runtime_context_dep),
) -> StreamingResponse:
    async def event_stream():
        yield encode_sse_event(
            "start",
            {
                "user_id": payload.user_id,
                "session_id": payload.session_id,
            },
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Tuple[str, Dict[str, Any]]] = asyncio.Queue()

        def emit(event: str, data: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        def run_runtime() -> None:
            runtime = context.runtime

            with context.lock:
                original_llm = runtime.llm
                runtime.llm = StreamingLLMProxy(
                    llm=original_llm,
                    on_delta=lambda token: emit("delta", {"text": token}),
                )
                try:
                    final_text = runtime.handle_message(
                        user_id=payload.user_id,
                        session_id=payload.session_id,
                        message=payload.message,
                    )
                except Exception as exc:
                    emit("error", {"message": str(exc)})
                else:
                    emit("end", {"text": final_text})
                finally:
                    runtime.llm = original_llm

        thread = threading.Thread(target=run_runtime, name="agent-api-chat-stream", daemon=True)
        thread.start()

        while True:
            event_name, event_payload = await queue.get()
            yield encode_sse_event(event_name, event_payload)
            if event_name in {"end", "error"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_stream_headers(),
    )
