from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from agent.providers.llm.base import LLMProvider, LLMResult, ToolSpec, Usage
from agent.providers.llm._tooling import (
    normalize_tool_choice,
    openai_tool_calls_to_normalized,
    tool_specs_to_openai,
)

# NOTE:
# We use lazy import, project can run without openai installed
# when using other providers.


class OpenAIProvider(LLMProvider):
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
        Uses OpenAI Chat Completions API shape for broad compatibility.
        """
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI SDK not installed. Install 'openai' to use OpenAIProvider."
            ) from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        oai_tools = tool_specs_to_openai(tools)
        tc = normalize_tool_choice(tool_choice)

        # Map tool_choice to OpenAI schema
        oai_tool_choice: Any = None
        if tc is None or tc == "auto":
            oai_tool_choice = "auto"
        elif tc == "none":
            oai_tool_choice = "none"
        else:
            # specific tool name
            oai_tool_choice = {"type": "function", "function": {"name": str(tc)}}

        resp = client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=oai_tools,
            tool_choice=oai_tool_choice,
        )

        choice = resp.choices[0]
        msg = choice.message

        text = msg.content or ""
        tool_calls = openai_tool_calls_to_normalized(getattr(msg, "tool_calls", None))

        usage = None
        if getattr(resp, "usage", None):
            usage = Usage(
                prompt_tokens=int(resp.usage.prompt_tokens or 0),
                completion_tokens=int(resp.usage.completion_tokens or 0),
                total_tokens=int(resp.usage.total_tokens or 0),
            )

        return LLMResult(text=text, tool_calls=tool_calls, usage=usage, raw=resp)