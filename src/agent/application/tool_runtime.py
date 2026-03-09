from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent.infrastructure.providers.llm.base import ToolSpec
from agent.infrastructure.skills.base import ToolContext, ToolExecutionError
from agent.infrastructure.skills.registry import ToolRegistry
from agent.infrastructure.tracing import JSONTracer


@dataclass(frozen=True)
class ToolRunResult:
    ok: bool
    name: str
    duration_ms: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class ToolRuntime:
    """
    Executes tools with:
    - basic JSON-schema-like validation (lightweight)
    - timeout protection
    - structured tracing
    """

    def __init__(
        self,
        registry: ToolRegistry,
        tracer: JSONTracer,
        default_timeout_s: float = 15.0,
        max_workers: int = 4,
    ) -> None:
        self.registry = registry
        self.tracer = tracer
        self.default_timeout_s = float(default_timeout_s)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def run(
        self,
        *,
        correlation_id: str,
        user_id: str,
        session_id: str,
        tool_spec: ToolSpec,
        args: Dict[str, Any],
        context: ToolContext,
        timeout_s: Optional[float] = None,
    ) -> ToolRunResult:
        """
        Run one tool call safely.
        """
        name = tool_spec.name
        timeout_s = float(timeout_s or self.default_timeout_s)

        # Validate args (minimal, avoids extra deps)
        try:
            self._validate_args(tool_spec, args)
        except Exception as e:
            return ToolRunResult(
                ok=False,
                name=name,
                duration_ms=0,
                error=str(e),
                error_type=type(e).__name__,
            )

        self.tracer.emit(
            event="tool.start",
            level="info",
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            payload={"tool": name, "timeout_s": timeout_s, "args": args},
        )

        start = time.time()

        def _call():
            return self.registry.execute(name=name, args=args, context=context)

        try:
            fut = self.pool.submit(_call)
            out = fut.result(timeout=timeout_s)
            dur_ms = int((time.time() - start) * 1000)

            self.tracer.emit(
                event="tool.result",
                level="info",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"tool": name, "ok": True, "duration_ms": dur_ms},
            )

            return ToolRunResult(ok=True, name=name, duration_ms=dur_ms, result=out)

        except FuturesTimeout:
            dur_ms = int((time.time() - start) * 1000)
            self.tracer.emit(
                event="tool.result",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"tool": name, "ok": False, "duration_ms": dur_ms, "error": "timeout"},
            )
            return ToolRunResult(
                ok=False,
                name=name,
                duration_ms=dur_ms,
                error=f"Tool timeout after {timeout_s}s",
                error_type="TimeoutError",
            )

        except ToolExecutionError as e:
            dur_ms = int((time.time() - start) * 1000)
            self.tracer.emit(
                event="tool.result",
                level="warning",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"tool": name, "ok": False, "duration_ms": dur_ms, "error": str(e)},
            )
            return ToolRunResult(
                ok=False,
                name=name,
                duration_ms=dur_ms,
                error=str(e),
                error_type="ToolExecutionError",
            )

        except Exception as e:
            dur_ms = int((time.time() - start) * 1000)
            tb = traceback.format_exc(limit=5)

            self.tracer.emit(
                event="tool.result",
                level="error",
                correlation_id=correlation_id,
                user_id=user_id,
                session_id=session_id,
                payload={"tool": name, "ok": False, "duration_ms": dur_ms, "error": str(e), "traceback": tb},
            )

            return ToolRunResult(
                ok=False,
                name=name,
                duration_ms=dur_ms,
                error=f"Unhandled tool error: {type(e).__name__}: {e}",
                error_type=type(e).__name__,
            )

    def _validate_args(self, tool_spec: ToolSpec, args: Dict[str, Any]) -> None:
        """
        Lightweight JSON-schema validation (subset):
        - required fields
        - additionalProperties == False
        - basic type checks (string/number/integer/boolean/object/array)

        This avoids adding jsonschema dependency for now.
        """
        schema = tool_spec.json_schema or {}
        if schema.get("type") and schema["type"] != "object":
            raise ValueError(f"Tool schema must be an object, got {schema.get('type')}")

        properties: Dict[str, Any] = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []
        additional = schema.get("additionalProperties", True)

        # required
        for k in required:
            if k not in args:
                raise ValueError(f"Missing required argument '{k}' for tool '{tool_spec.name}'")

        # additionalProperties
        if additional is False:
            for k in args.keys():
                if k not in properties:
                    raise ValueError(f"Unknown argument '{k}' for tool '{tool_spec.name}'")

        # basic type checks
        for k, v in args.items():
            prop = properties.get(k)
            if not prop:
                continue
            expected = prop.get("type")
            if not expected:
                continue
            if not self._type_ok(v, expected):
                raise ValueError(
                    f"Invalid type for '{k}' in tool '{tool_spec.name}': expected {expected}, got {type(v).__name__}"
                )

    def _type_ok(self, v: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(v, str)
        if expected == "number":
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        if expected == "integer":
            return isinstance(v, int) and not isinstance(v, bool)
        if expected == "boolean":
            return isinstance(v, bool)
        if expected == "object":
            return isinstance(v, dict)
        if expected == "array":
            return isinstance(v, list)
        return True  # unknown schema types => don't block