from __future__ import annotations

from typing import List, Sequence

from agent.providers.embeddings.base import EmbeddingProvider, EmbeddingResult


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local deterministic embedding provider for development/testing.

    This is NOT semantic. It produces stable pseudo-vectors derived from hashing.
    Useful for tests and pipeline validation without external dependencies.
    """

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.model = "local-hash"

    def supports_batch(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors: List[List[float]] = [self._hash_to_vec(t) for t in texts]
        return EmbeddingResult(vectors=vectors, dimensions=self.dimensions, model=self.model)

    def _hash_to_vec(self, text: str) -> List[float]:
        # Very simple stable hash embedding: not semantic, only for tests.
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand digest deterministically to dimensions
        out = []
        i = 0
        while len(out) < self.dimensions:
            b = h[i % len(h)]
            # map byte [0..255] to float [-1..1]
            out.append((b / 127.5) - 1.0)
            i += 1
        return out