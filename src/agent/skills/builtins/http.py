from __future__ import annotations

import json
from urllib import request

from agent.providers.llm.base import ToolSpec
from agent.skills.base import BaseToolHandler
from agent.skills.registry import Tool, ToolMeta


class HttpGetTool(BaseToolHandler):

    def execute(self, args, context):

        url = args.get("url")

        if not url:
            raise RuntimeError("url argument required")

        req = request.Request(
            url,
            headers={"User-Agent": "agent/1.0"},
        )

        with request.urlopen(req, timeout=15) as resp:

            content_type = resp.headers.get("Content-Type", "")

            body = resp.read().decode("utf-8", errors="ignore")

            if "application/json" in content_type:
                try:
                    body = json.loads(body)
                except Exception:
                    pass

            return {
                "status": resp.status,
                "url": url,
                "body": body,
            }


def register_tools(registry, settings):

    spec = ToolSpec(
        name="http.get",
        description="Fetch a URL using HTTP GET.",
        json_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string"
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    )

    registry.register(
        Tool(
            spec=spec,
            handler=HttpGetTool(),
            meta=ToolMeta(
                tags=frozenset({"network"}),
                risk="sensitive",
                default_timeout_s=15.0,
                scopes=frozenset({"net:http_get"}),
                requires_confirmation=False,
            ),
        )
    )
