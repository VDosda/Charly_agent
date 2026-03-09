from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Normalized embedding result.

    vectors: list of float vectors, each of length = dimensions
    dimensions: embedding dimension
    model: model identifier
    """
    vectors: List[List[float]]
    dimensions: int
    model: str


class EmbeddingProvider(Protocol):
    """
    Contract for all embedding providers.
    """

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """
        Return embeddings for the given list of texts.
        """
        ...

    def supports_batch(self) -> bool:
        """
        Whether provider supports embedding multiple texts in one request efficiently.
        """
        ...