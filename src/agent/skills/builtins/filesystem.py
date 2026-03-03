from __future__ import annotations

from pathlib import Path
from typing import Dict

from agent.providers.llm.base import ToolSpec
from agent.skills.base import BaseToolHandler, ToolExecutionError
from agent.skills.registry import Tool


# Root directory where the agent is allowed to read/write files.
ROOT_DIR: Optional[Path] = None

def _safe_path(path: str) -> Path:
    if ROOT_DIR is None:
        raise ToolExecutionError("Filesystem tool not initialized (missing workspace).")
    target = (ROOT_DIR / path).resolve()
    if not str(target).startswith(str(ROOT_DIR)):
        raise ToolExecutionError("Access outside workspace is not allowed.")
    return target


class ReadFileTool(BaseToolHandler):

    def execute(self, args: Dict, context):

        path = args.get("path")

        if not path:
            raise ToolExecutionError("path argument required")

        file_path = _safe_path(path)

        if not file_path.exists():
            raise ToolExecutionError(f"File not found: {path}")

        if file_path.is_dir():
            raise ToolExecutionError("Cannot read a directory")

        content = file_path.read_text(encoding="utf-8")

        return {
            "path": str(file_path),
            "content": content,
        }


class WriteFileTool(BaseToolHandler):

    def execute(self, args: Dict, context):

        path = args.get("path")
        content = args.get("content")

        if not path:
            raise ToolExecutionError("path argument required")

        if content is None:
            raise ToolExecutionError("content argument required")

        file_path = _safe_path(path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding="utf-8")

        return {
            "path": str(file_path),
            "written": True,
        }


class ListDirTool(BaseToolHandler):

    def execute(self, args: Dict, context):

        path = args.get("path", "")

        dir_path = _safe_path(path)

        if not dir_path.exists():
            raise ToolExecutionError(f"Directory not found: {path}")

        if not dir_path.is_dir():
            raise ToolExecutionError("Path is not a directory")

        entries = []

        for p in dir_path.iterdir():
            entries.append(
                {
                    "name": p.name,
                    "type": "dir" if p.is_dir() else "file",
                }
            )

        return {
            "path": str(dir_path),
            "entries": entries,
        }


def register_tools(registry, settings):

    global ROOT_DIR

    ROOT_DIR = Path(settings.workspace).resolve()
    ROOT_DIR.mkdir(parents=True, exist_ok=True)

    registry.register(
        Tool(
            spec=ToolSpec(
                name="fs.read_file",
                description="Read a text file from the workspace.",
                json_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            handler=ReadFileTool(),
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="fs.write_file",
                description="Write content to a file in the workspace.",
                json_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            ),
            handler=WriteFileTool(),
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="fs.list_dir",
                description="List files and directories in the workspace.",
                json_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "additionalProperties": False,
                },
            ),
            handler=ListDirTool(),
        )
    )
