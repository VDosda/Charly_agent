from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from agent.providers.llm.base import LLMProvider, LLMResult, ToolCall, ToolSpec, Usage
from agent.providers.llm._tooling import new_tool_call_id

# Anthropic "tools" are supported in their Messages API.
# We keep a normalized output.


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def supports_tools(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResult:
        """
        Anthropic expects:
        - system: separate field
        - messages: user/assistant
        - tools: list of {name, description, input_schema}
        Response content can include tool_use blocks.
        """
        try:
            import anthropic  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Anthropic SDK not installed. Install 'anthropic' to use AnthropicProvider."
            ) from e

        client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)

        system_text = ""
        anth_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                system_text += (m.get("content") or "") + "\n"
            else:
                anth_messages.append({"role": role, "content": m.get("content")})

        anth_tools = None
        if tools:
            anth_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.json_schema,
                }
                for t in tools
            ]

        # tool_choice mapping:
        # - "auto" => let model decide
        # - "none" => no tool use
        # - "<name>" => force tool name
        anth_tool_choice: Any = None
        if tool_choice in (None, "auto"):
            anth_tool_choice = {"type": "auto"}
        elif tool_choice == "none":
            anth_tool_choice = {"type": "none"}
        else:
            anth_tool_choice = {"type": "tool", "name": str(tool_choice)}

        resp = client.messages.create(
            model=self.model,
            system=system_text.strip() or None,
            messages=anth_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=anth_tools,
            tool_choice=anth_tool_choice,
        )

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        # Anthropic content blocks: text, tool_use, etc.
        for block in resp.content:
            btype = getattr(block, "type", None) or block.get("type")
            if btype == "text":
                text_parts.append(getattr(block, "text", None) or block.get("text") or "")
            elif btype == "tool_use":
                name = getattr(block, "name", None) or block.get("name") or "unknown_tool"
                inp = getattr(block, "input", None) or block.get("input") or {}
                tc_id = getattr(block, "id", None) or block.get("id") or new_tool_call_id()
                tool_calls.append(ToolCall(id=tc_id, name=name, arguments=inp))

        usage = None
        # Some SDKs expose usage differently; keep best effort.
        if getattr(resp, "usage", None):
            u = resp.usage
            usage = Usage(
                prompt_tokens=int(getattr(u, "input_tokens", 0) or 0),
                completion_tokens=int(getattr(u, "output_tokens", 0) or 0),
                total_tokens=int((getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0)),
            )

        return LLMResult(text="".join(text_parts).strip(), tool_calls=tool_calls, usage=usage, raw=resp)