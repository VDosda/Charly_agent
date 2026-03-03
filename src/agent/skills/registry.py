from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from agent.providers.llm.base import ToolSpec


@dataclass(frozen=True)
class Tool:
    """
    Runtime representation of a tool.

    - spec: what we expose to the LLM (name/description/schema)
    - handler: callable that executes the tool with validated args
    """
    spec: ToolSpec
    handler: "ToolHandler"


class ToolHandler:
    """
    Interface-like base for tool handlers.

    A handler must implement:
      execute(args: dict, context: dict) -> dict
    """
    def execute(self, args: dict, context: dict) -> dict:  # pragma: no cover
        raise NotImplementedError


class ToolRegistry:
    """
    Central tool registry.

    Responsibilities:
    - register tools
    - detect name collisions
    - provide ToolSpec list to LLM providers
    - execute tool by name
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name.strip()
        if not name:
            raise ValueError("Tool name cannot be empty")

        if name in self._tools:
            raise ValueError(f"Duplicate tool name registered: '{name}'")

        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: '{name}'")
        return self._tools[name]

    def list_specs(self) -> List[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def list_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def execute(self, name: str, args: dict, context: dict) -> dict:
        tool = self.get(name)
        return tool.handler.execute(args=args, context=context)


# -------------------------
# Loading from config
# -------------------------

def load_enabled_skills(skill_paths: Sequence[str], settings: Settings) -> ToolRegistry:
    """
    Load tools from a list of dotted module paths declared in config.

    Example skill_paths:
      - "builtins.time"
      - "plugins.gmail"

    Each module must expose:
      register_tools(registry: ToolRegistry) -> None
    """
    registry = ToolRegistry()

    for path in skill_paths:
        path = path.strip()
        if not path:
            continue

        module_name = _resolve_skill_module(path)

        mod = importlib.import_module(module_name)

        if not hasattr(mod, "register_tools"):
            raise RuntimeError(
                f"Skill module '{module_name}' must define register_tools(registry)."
            )

        mod.register_tools(registry, settings)

    return registry


def _resolve_skill_module(skill_path: str) -> str:
    """
    Resolve a short skill path into a python module name.

    Convention:
    - "builtins.time"   -> "agent.skills.builtins.time"
    - "plugins.gmail"   -> "agent.skills.plugins.gmail"
    - "agent.skills...." stays as-is (explicit)
    """
    if skill_path.startswith("agent.skills."):
        return skill_path

    if skill_path.startswith("builtins."):
        return "agent.skills." + skill_path

    if skill_path.startswith("plugins."):
        return "agent.skills." + skill_path

    # allow explicit external packages if user wants it
    return skill_path