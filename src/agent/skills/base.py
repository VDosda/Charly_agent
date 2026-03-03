from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from agent.providers.llm.base import ToolSpec


class ToolExecutionError(RuntimeError):
    """
    Raised when a tool execution fails in a controlled way.
    """
    pass


@dataclass
class ToolContext:
    """
    Context passed to tools.

    Allows tools to access runtime data safely.
    """
    user_id: str
    session_id: str
    metadata: Dict[str, Any]


class BaseToolHandler:
    """
    Base class for tool handlers.

    Subclasses must implement execute().
    """

    def execute(self, args: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        raise NotImplementedError