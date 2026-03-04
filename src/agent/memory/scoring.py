from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar

from agent.memory.models import (
    MemoryItem,
    Episode,
    RetrievedEpisode,
    RetrievedMemoryItem,
    RetrievalEvidence,
)

T = TypeVar("T")


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def distance_to_similarity(dist: float) -> float:
    """
    Converts a distance (lower is better) into a [0..1] similarity-like score.
    Stable, monotonic, no dependency on metric type.
    """
    if dist < 0:
        dist = 0.0
    return 1.0 / (1.0 + float(dist))


def recency_score(ts: int, now: Optional[int] = None, half_life_days: float = 7.0) -> float:
    """
    Exponential-like decay to [0..1].
    half_life_days controls how quickly old memories fade.
    """
    now = int(now if now is not None else time.time())
    age = max(0, now - int(ts))
    half_life_s = max(1.0, half_life_days * 24 * 3600)
    # score = 0.5 when age == half_life
    return float(math.exp(-math.log(2.0) * (age / half_life_s)))


@dataclass(frozen=True)
class HybridWeights:
    """
    Hybrid score = sim*w_sim + importance*w_imp + recency*w_rec + confidence*w_conf
    """
    w_sim: float = 0.75
    w_importance: float = 0.15
    w_recency: float = 0.10
    w_confidence: float = 0.00  # optional


def hybrid_score(
    *,
    similarity: float,
    importance: float,
    recency: float,
    confidence: float,
    w: HybridWeights,
) -> float:
    return (
        w.w_sim * clamp01(similarity)
        + w.w_importance * clamp01(importance)
        + w.w_recency * clamp01(recency)
        + w.w_confidence * clamp01(confidence)
    )


def rerank_episodes(
    episodes_with_distance: Sequence[Tuple[Episode, float]],
    *,
    now: Optional[int] = None,
    weights: HybridWeights = HybridWeights(w_sim=0.75, w_importance=0.15, w_recency=0.10, w_confidence=0.00),
    half_life_days: float = 14.0,
    limit: int = 3,
) -> List[RetrievedEpisode]:
    now = int(now if now is not None else time.time())

    scored: List[RetrievedEpisode] = []
    for ep, dist in episodes_with_distance:
        sim = distance_to_similarity(dist)
        rec = recency_score(ep.ts, now=now, half_life_days=half_life_days)
        score = hybrid_score(
            similarity=sim,
            importance=ep.importance,
            recency=rec,
            confidence=ep.confidence,
            w=weights,
        )
        scored.append(
            RetrievedEpisode(
                episode=ep,
                score=float(score),
                evidence=RetrievalEvidence(
                    vector_distance=float(dist),
                    vector_similarity=float(sim),
                    recency_score=float(rec),
                    importance_score=float(ep.importance),
                    confidence_score=float(ep.confidence),
                ),
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[: int(limit)]


def rerank_items(
    items_with_distance: Sequence[Tuple[MemoryItem, float]],
    *,
    now: Optional[int] = None,
    weights: HybridWeights = HybridWeights(w_sim=0.75, w_importance=0.20, w_recency=0.05, w_confidence=0.00),
    half_life_days: float = 60.0,
    limit: int = 6,
) -> List[RetrievedMemoryItem]:
    now = int(now if now is not None else time.time())

    scored: List[RetrievedMemoryItem] = []
    for it, dist in items_with_distance:
        sim = distance_to_similarity(dist)
        rec = recency_score(it.ts_updated, now=now, half_life_days=half_life_days)
        score = hybrid_score(
            similarity=sim,
            importance=it.importance,
            recency=rec,
            confidence=it.confidence,
            w=weights,
        )
        scored.append(
            RetrievedMemoryItem(
                item=it,
                score=float(score),
                evidence=RetrievalEvidence(
                    vector_distance=float(dist),
                    vector_similarity=float(sim),
                    recency_score=float(rec),
                    importance_score=float(it.importance),
                    confidence_score=float(it.confidence),
                ),
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[: int(limit)]