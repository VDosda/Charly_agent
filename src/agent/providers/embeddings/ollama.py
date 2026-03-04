from __future__ import annotations

import json
from typing import List, Sequence
from urllib import error, request

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
        # Preferred Ollama endpoint (recent versions)
        try:
            return self._embed_via_embed_endpoint(text)
        except error.HTTPError as e:
            # Backward compatibility with older Ollama endpoint
            if e.code != 404:
                raise RuntimeError(f"Failed calling Ollama embeddings at /api/embed: HTTP {e.code} {e.reason}") from e

        return self._embed_via_embeddings_endpoint(text)

    def _embed_via_embed_endpoint(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": text}
        obj = self._post_json(url, payload)

        vectors = obj.get("embeddings")
        if isinstance(vectors, list) and vectors:
            vec = vectors[0]
            if not isinstance(vec, list):
                raise RuntimeError("Ollama /api/embed response 'embeddings[0]' is not a list")
            return [float(x) for x in vec]

        # Some proxies/versions may still return "embedding" shape.
        vec = obj.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError(
                "Ollama /api/embed response missing vector in 'embeddings' or 'embedding'"
            )
        return [float(x) for x in vec]

    def _embed_via_embeddings_endpoint(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        obj = self._post_json(url, payload)

        vec = obj.get("embedding")
        if not isinstance(vec, list):
            raise RuntimeError("Ollama /api/embeddings response missing 'embedding' list")

        return [float(x) for x in vec]

    def _post_json(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed calling Ollama embeddings at {url}: {e}") from e

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from Ollama embeddings endpoint {url}") from e

        if not isinstance(obj, dict):
            raise RuntimeError(f"Unexpected Ollama embeddings response type at {url}: {type(obj).__name__}")

        return obj
