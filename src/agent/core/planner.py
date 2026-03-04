from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from agent.providers.llm.base import ToolSpec


@dataclass(frozen=True)
class PlanDecision:
    tool_choice: str
    system_overrides: Optional[str] = None
    blocked_response: Optional[str] = None


@dataclass(frozen=True)
class ToolPolicy:
    """
    Scalable policy: no per-tool hardcoding.
    Tools are classified by name patterns into capability buckets.
    """
    allow_tools_by_default: bool = True

    # Capability toggles
    allow_network: bool = True
    allow_filesystem_read: bool = True
    allow_filesystem_write: bool = False  # safest default

    # Explicit deny list
    deny_tools: Set[str] = frozenset()

    # (Future) tools requiring user confirmation
    require_confirmation_for: Set[str] = frozenset({"fs.write_file"})


class Planner:
    def __init__(self, policy: Optional[ToolPolicy] = None) -> None:
        self.policy = policy or ToolPolicy()

    def decide(
        self,
        *,
        user_message: str,
        available_tools: Optional[Sequence[ToolSpec]],
        session_state: Optional[Dict[str, Any]] = None,
    ) -> PlanDecision:
        session_state = session_state or {}

        if not available_tools:
            return PlanDecision(tool_choice="none")

        if not self.policy.allow_tools_by_default:
            return PlanDecision(tool_choice="none")

        # Defense in depth: block obvious sensitive file exfiltration intent
        if self._looks_like_sensitive_file_request(user_message):
            return PlanDecision(
                tool_choice="none",
                blocked_response=(
                    "I can't help with accessing sensitive system files. "
                    "If you meant files in the workspace sandbox, specify the workspace-relative path."
                ),
            )

        sys_overrides: List[str] = []

        if self._mentions_url(user_message):
            sys_overrides.append(
                "If you need remote content, use http.get first and then summarize."
            )

        if self._asks_for_time(user_message):
            sys_overrides.append("If the user asks for current time, use time.now.")

        return PlanDecision(
            tool_choice="auto",
            system_overrides="\n".join(sys_overrides) if sys_overrides else None,
        )

    def filter_tools(self, tools: Sequence[ToolSpec]) -> List[ToolSpec]:
        out: List[ToolSpec] = []

        for t in tools:
            if t.name in self.policy.deny_tools:
                continue

            caps = self._capabilities_for_tool(t.name)

            if "network" in caps and not self.policy.allow_network:
                continue

            if "fs_read" in caps and not self.policy.allow_filesystem_read:
                continue

            if "fs_write" in caps and not self.policy.allow_filesystem_write:
                continue

            out.append(t)

        return out

    def requires_confirmation(self, tool_name: str) -> bool:
        return tool_name in self.policy.require_confirmation_for

    # -------------------------
    # Capability classification
    # -------------------------

    def _capabilities_for_tool(self, tool_name: str) -> Set[str]:
        """
        Naming convention = scalability.
        Add new tools with prefixes and they auto-classify.
        """
        caps: Set[str] = set()

        if tool_name.startswith("http."):
            caps.add("network")

        if tool_name.startswith("fs."):
            if tool_name == "fs.write_file" or tool_name.startswith("fs.write"):
                caps.add("fs_write")
            else:
                caps.add("fs_read")

        if tool_name.startswith("time."):
            caps.add("safe")

        return caps

    # -------------------------
    # Heuristics
    # -------------------------

    def _mentions_url(self, text: str) -> bool:
        return bool(re.search(r"https?://", text, flags=re.IGNORECASE))

    def _asks_for_time(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in ["time is it", "current time", "heure", "date", "now"])

    def _looks_like_sensitive_file_request(self, text: str) -> bool:
        t = text.lower()
        suspicious = [
            "/etc/passwd", "/etc/shadow", "id_rsa", ".ssh", "authorized_keys",
            "kubeconfig", "token", "apikey", "api key", "secret", "credentials",
            "windows\\system32", "sam", "registry hive",
        ]
        return any(s in t for s in suspicious)