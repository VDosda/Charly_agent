from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class ToolSpec:
    """
    Provider-agnostic tool specification.

    We intentionally keep it close to the common "function tool" pattern:
    - name: tool name
    - description: what it does
    - json_schema: JSON schema for arguments
    """
    name: str
    description: str
    json_schema: Dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """
    Provider-agnostic tool call emitted by the LLM.
    """
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    """
    Provider-agnostic response from the LLM.
    """
    text: str
    tool_calls: List[ToolCall]
    usage: Optional[Usage] = None
    raw: Optional[Any] = None  # Keep original provider response for debugging


class LLMProvider(Protocol):
    """
    Contract for all LLM providers.
    """

    def generate(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResult:
        """
        Generate a completion.

        messages: list of {"role": "...", "content": "..."} plus optional provider-specific fields
        tools: list of tool specs that the provider can use
        tool_choice: provider-agnostic hint, e.g. "auto" | "none" | "<tool_name>"
        """
        ...

    def supports_tools(self) -> bool:
        """
        Indicates whether this provider implementation supports tool calling.
        """
        ...