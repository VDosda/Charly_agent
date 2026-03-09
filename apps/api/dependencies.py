from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache

from agent.bootstrap.runtime_factory import create_runtime
from agent.application.runtime import AgentRuntime


@dataclass(frozen=True)
class RuntimeContext:
    runtime: AgentRuntime
    lock: threading.Lock


@lru_cache(maxsize=1)
def get_runtime_context() -> RuntimeContext:
    return RuntimeContext(runtime=create_runtime(), lock=threading.Lock())


def get_runtime_context_dep() -> RuntimeContext:
    return get_runtime_context()
