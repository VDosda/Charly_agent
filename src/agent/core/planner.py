from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from agent.providers.llm.base import ToolSpec


@dataclass(frozen=True)
class PlanDecision:
    """
    Planner output consumed by AgentRuntime.

    tool_choice:
      - "auto": allow tool usage
      - "none": disallow tool usage
      - "<tool_name>": force a specific tool (rare, but useful)
    """
    tool_choice: str
    # Optional extra system instructions injected for this turn
    system_overrides: Optional[str] = None
    # Optional "hard stop" message (planner blocks execution and returns this)
    blocked_response: Optional[str] = None


class Planner:
    """
    A lightweight policy layer.

    Goals:
    - Reduce unsafe / wasteful tool usage
    - Add simple routing heuristics
    - Keep it provider-agnostic

    This is NOT a full "plan-and-execute" planner.
    It is a deterministic rules layer that sits above the LLM call.
    """

    def __init__(
        self,
        *,
        allow_tools_by_default: bool = True,
        deny_tools: Optional[List[str]] = None,
        require_confirmation_for: Optional[List[str]] = None,
    ) -> None:
        self.allow_tools_by_default = allow_tools_by_default
        self.deny_tools = set(deny_tools or [])

        # If you later implement a confirmation mechanism, these tools can be gated.
        self.require_confirmation_for = set(require_confirmation_for or [
            "fs.write_file",
        ])

    def decide(
        self,
        *,
        user_message: str,
        available_tools: Optional[Sequence[ToolSpec]],
        session_state: Optional[Dict[str, Any]] = None,
    ) -> PlanDecision:
        """
        Decide whether to allow tools, deny tools, or force a tool.

        session_state can later contain things like:
        - user confirmations
        - rate limiting counters
        - environment flags
        """
        session_state = session_state or {}

        # 0) No tools available anyway
        if not available_tools:
            return PlanDecision(tool_choice="none")

        # 1) If tools are globally disabled, block tools entirely
        if not self.allow_tools_by_default:
            return PlanDecision(tool_choice="none")

        # 2) Safety: if user asks for clearly dangerous filesystem actions (outside scope),
        # block or steer. Note: your filesystem tool already sandboxes paths, but we add policy too.
        if self._looks_like_sensitive_file_request(user_message):
            return PlanDecision(
                tool_choice="none",
                blocked_response=(
                    "I can't help with accessing sensitive system files. "
                    "If you meant files in the workspace sandbox, specify the workspace-relative path."
                ),
            )

        # 3) Heuristic: If user explicitly asks to fetch a URL, allow tools (http.get)
        if self._mentions_url(user_message) and self._tool_exists("http.get", available_tools):
            return PlanDecision(
                tool_choice="auto",
                system_overrides=(
                    "If you need web content, use the http.get tool first and then summarize."
                ),
            )

        # 4) Heuristic: If user asks for current time/date, allow tools (time.now)
        if self._asks_for_time(user_message) and self._tool_exists("time.now", available_tools):
            return PlanDecision(
                tool_choice="auto",
                system_overrides="If the user asks for current time, use time.now."
            )

        # 5) Default: allow tools
        # The tool runtime will validate args + enforce timeouts.
        return PlanDecision(tool_choice="auto")

    # -------------------------
    # Helpers
    # -------------------------

    def filter_tools(self, tools: Sequence[ToolSpec]) -> List[ToolSpec]:
        """
        Apply deny-list filtering to the tool list exposed to the LLM.

        This prevents the model from calling tools that are disabled by policy.
        """
        out = []
        for t in tools:
            if t.name in self.deny_tools:
                continue
            out.append(t)
        return out

    def requires_confirmation(self, tool_name: str) -> bool:
        """
        Whether this tool should require explicit user confirmation
        before being executed.

        You can implement confirmation handling in AgentRuntime later.
        """
        return tool_name in self.require_confirmation_for

    def _tool_exists(self, name: str, tools: Sequence[ToolSpec]) -> bool:
        return any(t.name == name for t in tools)

    def _mentions_url(self, text: str) -> bool:
        return bool(re.search(r"https?://", text, flags=re.IGNORECASE))

    def _asks_for_time(self, text: str) -> bool:
        # Simple heuristic; you can refine later.
        t = text.lower()
        print(f"Checking if user is asking for time/date: '{text}'")
        return any(k in t for k in ["time is it", "current time", "heure", "date", "now"])

    def _looks_like_sensitive_file_request(self, text: str) -> bool:
        """
        Detect obvious attempts to read system secrets.
        This is a defense-in-depth check; your filesystem tool already sandboxes.
        """
        t = text.lower()
        suspicious = [
            "/etc/passwd", "/etc/shadow", "id_rsa", ".ssh", "authorized_keys",
            "kubeconfig", "token", "apikey", "api key", "secret", "credentials",
            "windows\\system32", "sam", "registry hive"
        ]
        return any(s in t for s in suspicious)