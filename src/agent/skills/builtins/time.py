from __future__ import annotations

from datetime import datetime, timezone

from agent.providers.llm.base import ToolSpec
from agent.skills.base import BaseToolHandler
from agent.skills.registry import Tool, ToolMeta


class NowTool(BaseToolHandler):

    def execute(self, args, context):

        now = datetime.now(timezone.utc)

        return {
            "iso": now.isoformat(),
            "timestamp": int(now.timestamp()),
            "utc": True,
        }


def register_tools(registry, settings):

    spec = ToolSpec(
        name="time.now",
        description="Get the current UTC time.",
        json_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    registry.register(
        Tool(
            spec=spec,
            handler=NowTool(),
            meta=ToolMeta(
                tags=frozenset({"safe"}),
                risk="safe",
                default_timeout_s=5.0,
                scopes=frozenset(),
                requires_confirmation=False,
            ),
        )
    )
