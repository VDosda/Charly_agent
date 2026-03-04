from __future__ import annotations

from dataclasses import dataclass, field
from typing import AbstractSet, Any, Dict, Mapping, Optional, Sequence, Set

from agent.skills.registry import Tool, ToolRisk


@dataclass(frozen=True)
class PlanDecision:
    tool_choice: str
    system_overrides: Optional[str] = None
    blocked_response: Optional[str] = None


@dataclass(frozen=True)
class ToolPolicy:
    """
    Central policy applied to declarative ToolMeta.

    Skills declare metadata.
    Planner applies global governance from settings/session.
    """
    allow_tools_by_default: bool = True

    # Coarse global toggles
    allow_network: bool = True
    allow_filesystem_read: bool = True
    allow_filesystem_write: bool = False  # safest default

    # Fine-grained policy controls
    deny_tools: Set[str] = frozenset()
    allow_tags: Optional[Set[str]] = None
    deny_tags: Set[str] = frozenset()
    deny_scopes: Set[str] = frozenset()
    deny_risk: Set[ToolRisk] = frozenset()
    timeout_overrides_s: Mapping[str, float] = field(default_factory=dict)
    enforce_confirmation: bool = False

    @classmethod
    def from_settings(cls, settings) -> "ToolPolicy":
        cfg = settings.tool_policy
        allow_tags = frozenset(cfg.allow_tags) if cfg.allow_tags else None
        return cls(
            allow_tools_by_default=cfg.allow_tools_by_default,
            allow_network=cfg.allow_network,
            allow_filesystem_read=cfg.allow_filesystem_read,
            allow_filesystem_write=cfg.allow_filesystem_write,
            deny_tools=frozenset(cfg.deny_tools),
            allow_tags=allow_tags,
            deny_tags=frozenset(cfg.deny_tags),
            deny_scopes=frozenset(cfg.deny_scopes),
            deny_risk=frozenset(cfg.deny_risk),
            timeout_overrides_s=dict(cfg.tool_timeouts_s),
            enforce_confirmation=cfg.enforce_confirmation,
        )


class Planner:
    def __init__(self, policy: Optional[ToolPolicy] = None) -> None:
        self.policy = policy or ToolPolicy()

    def decide(
        self,
        *,
        user_message: str,
        available_tools: Optional[Sequence[Tool]],
        session_state: Optional[Dict[str, Any]] = None,
    ) -> PlanDecision:
        session_state = session_state or {}

        if not available_tools:
            return PlanDecision(tool_choice="none")

        if not self.policy.allow_tools_by_default:
            return PlanDecision(tool_choice="none")

        return PlanDecision(tool_choice="auto")

    def filter_tools(
        self,
        tools: Sequence[Tool],
        session_state: Optional[Dict[str, Any]] = None,
    ) -> List[Tool]:
        return [t for t in tools if self.is_tool_allowed(t, session_state=session_state)]

    def is_tool_allowed(self, tool: Tool, session_state: Optional[Dict[str, Any]] = None) -> bool:
        if not self.policy.allow_tools_by_default:
            return False

        session_state = session_state or {}
        deny_tools = self._merge_set(self.policy.deny_tools, session_state, "deny_tools")
        deny_tags = self._merge_set(self.policy.deny_tags, session_state, "deny_tags")
        deny_scopes = self._merge_set(self.policy.deny_scopes, session_state, "deny_scopes")
        deny_risk = self._merge_set(self.policy.deny_risk, session_state, "deny_risk")

        if tool.spec.name in deny_tools:
            return False

        if tool.meta.risk in deny_risk:
            return False

        if deny_tags.intersection(tool.meta.tags):
            return False

        if self._scope_is_denied(tool.meta.scopes, deny_scopes):
            return False

        allow_tags = self._effective_allow_tags(session_state)
        if allow_tags is not None and not allow_tags.intersection(tool.meta.tags):
            return False

        if not self.policy.allow_network:
            if "network" in tool.meta.tags or self._has_scope_prefix(tool.meta.scopes, "net:"):
                return False

        if not self.policy.allow_filesystem_read:
            if self._has_scope_prefix(tool.meta.scopes, "fs:read"):
                return False

        if not self.policy.allow_filesystem_write:
            if self._has_scope_prefix(tool.meta.scopes, "fs:write"):
                return False

        return True

    def requires_confirmation(
        self,
        tool: Tool,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.policy.enforce_confirmation:
            return False

        if not tool.meta.requires_confirmation:
            return False

        session_state = session_state or {}
        confirmed_tools = self._read_set(session_state, "confirmed_tools")
        return tool.spec.name not in confirmed_tools

    def timeout_for(
        self,
        tool: Tool,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        timeout = self.policy.timeout_overrides_s.get(tool.spec.name, tool.meta.default_timeout_s)
        session_timeout_overrides = self._read_timeout_overrides(session_state or {})
        timeout = session_timeout_overrides.get(tool.spec.name, timeout)
        return float(timeout)

    def _effective_allow_tags(self, session_state: Dict[str, Any]) -> Optional[Set[str]]:
        base_allow = set(self.policy.allow_tags) if self.policy.allow_tags is not None else None
        session_allow = self._read_set(session_state, "allow_tags")

        if not session_allow:
            return base_allow

        if base_allow is None:
            return session_allow

        return base_allow.intersection(session_allow)

    def _merge_set(
        self,
        base: Set[str] | Set[ToolRisk],
        session_state: Dict[str, Any],
        key: str,
    ) -> Set[str]:
        merged = {str(v) for v in base}
        merged.update(self._read_set(session_state, key))
        return merged

    def _read_set(self, session_state: Dict[str, Any], key: str) -> Set[str]:
        raw = session_state.get(key)
        if raw is None:
            return set()

        if isinstance(raw, str):
            return {v.strip() for v in raw.split(",") if v.strip()}

        if isinstance(raw, (list, tuple, set, frozenset)):
            out: Set[str] = set()
            for v in raw:
                s = str(v).strip()
                if s:
                    out.add(s)
            return out

        return set()

    def _read_timeout_overrides(self, session_state: Dict[str, Any]) -> Dict[str, float]:
        raw = session_state.get("tool_timeouts_s")
        if not isinstance(raw, dict):
            return {}

        out: Dict[str, float] = {}
        for name, value in raw.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v > 0:
                out[str(name)] = v
        return out

    def _scope_is_denied(self, tool_scopes: AbstractSet[str], denied_scopes: Set[str]) -> bool:
        for denied in denied_scopes:
            if denied.endswith("*"):
                prefix = denied[:-1]
                if any(scope.startswith(prefix) for scope in tool_scopes):
                    return True
                continue
            if denied in tool_scopes:
                return True
        return False

    def _has_scope_prefix(self, scopes: AbstractSet[str], prefix: str) -> bool:
        return any(scope.startswith(prefix) for scope in scopes)
