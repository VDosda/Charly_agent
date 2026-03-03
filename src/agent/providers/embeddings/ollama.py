from __future__ import annotations

import json
from typing import List, Sequence
from urllib import request

from agent.providers.embeddings.base import EmbeddingProvider, EmbeddingResult


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str,
        dimensions: int,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")

    def supports_batch(self) -> bool:
        # Ollama embeddings endpoint is typically single prompt; we can loop.
        return False

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], dimensions=self.dimensions, model=self.model)

        vectors: List[List[float]] = []
        for t in texts:
            vectors.append(self._embed_one(t))

        if vectors and len(vectors[0]) != self.dimensions:
            raise ValueError(
                f"Ollama embedding dims mismatch: got {len(vectors[0])} expected {self.dimensions}. "
                "Update EMBED_DIMS to match your Ollama embedding model."
            )

        return EmbeddingResult(vectors=vectors, dimensions=self.dimensions, model=self.model)

    def _embed_one(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed calling Ollama embeddings at {url}: {e}") from e

        obj = json.loads(raw)
        vec = obj.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError("Ollama embeddings response missing 'embedding' list")

        return [float(x) for x in vec]