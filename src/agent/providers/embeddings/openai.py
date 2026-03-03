from __future__ import annotations

from typing import List, Optional, Sequence

from agent.providers.embeddings.base import EmbeddingProvider, EmbeddingResult


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str,
        dimensions: int,
        api_key: Optional[str],
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key
        self.base_url = base_url

    def supports_batch(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAI SDK not installed. Install 'openai'.") from e

        if not texts:
            return EmbeddingResult(vectors=[], dimensions=self.dimensions, model=self.model)

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # OpenAI embeddings endpoint supports batch input
        resp = client.embeddings.create(
            model=self.model,
            input=list(texts),
        )

        vectors: List[List[float]] = []
        for item in resp.data:
            vec = list(item.embedding)
            vectors.append(vec)

        # Best-effort dims validation (some models fixed; rely on config)
        # If you want strict, compare against len(vectors[0]) as truth.
        if vectors and len(vectors[0]) != self.dimensions:
            raise ValueError(
                f"OpenAI embedding dims mismatch: got {len(vectors[0])} expected {self.dimensions}. "
                "Update EMBED_DIMS or choose the correct embedding model."
            )

        return EmbeddingResult(vectors=vectors, dimensions=self.dimensions, model=self.model)