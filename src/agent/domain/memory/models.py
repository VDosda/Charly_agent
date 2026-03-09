from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class MemoryKind(str, Enum):
    identity = "identity"
    preference = "preference"
    constraint = "constraint"
    goal = "goal"
    procedure = "procedure"
    project = "project"
    other = "other"


@dataclass(frozen=True)
class Turn:
    """Short-Term memory turn (from chat_history)."""
    turn_id: int
    ts: int
    role: str  # user|assistant|tool|system
    content: str
    tool_name: Optional[str] = None


@dataclass(frozen=True)
class Episode:
    """Medium-Term memory episode (from episodes)."""
    id: int
    user_id: str
    session_id: str
    start_turn_id: int
    end_turn_id: int
    ts: int
    summary: str

    topics: List[str] = field(default_factory=list)
    facts: List[Dict[str, str]] = field(default_factory=list)
    open_tasks: List[Dict[str, str]] = field(default_factory=list)

    importance: float = 0.5
    confidence: float = 0.6


@dataclass(frozen=True)
class MemoryItem:
    """Long-Term memory item (from memory_items)."""
    id: int
    user_id: str
    kind: str  # prefer to keep as string for DB compatibility
    mem_key: Optional[str]
    value: str

    ts_created: int
    ts_updated: int
    last_seen_ts: int

    importance: float = 0.5
    confidence: float = 0.6

    source_session_id: Optional[str] = None
    source_episode_id: Optional[int] = None
    source_note: Optional[str] = None


@dataclass(frozen=True)
class RetrievalEvidence:
    """
    Optional debug/explainability info.
    Keep lightweight; do not inject raw secrets.
    """
    vector_distance: Optional[float] = None
    vector_similarity: Optional[float] = None
    recency_score: Optional[float] = None
    importance_score: Optional[float] = None
    confidence_score: Optional[float] = None


@dataclass(frozen=True)
class RetrievedEpisode:
    episode: Episode
    score: float
    evidence: RetrievalEvidence = field(default_factory=RetrievalEvidence)


@dataclass(frozen=True)
class RetrievedMemoryItem:
    item: MemoryItem
    score: float
    evidence: RetrievalEvidence = field(default_factory=RetrievalEvidence)


@dataclass(frozen=True)
class MemoryBundle:
    """
    What retrieval returns to the prompt assembly layer.
    """
    lt_items: List[RetrievedMemoryItem] = field(default_factory=list)
    mt_episodes: List[RetrievedEpisode] = field(default_factory=list)