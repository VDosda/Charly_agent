from __future__ import annotations

from agent.config.settings import Settings
from agent.providers.embeddings.base import EmbeddingProvider


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    p = settings.embeddings.provider

    if p == "openai":
        from agent.providers.embeddings.openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(
            model=settings.embeddings.model,
            dimensions=settings.embeddings.dimensions,
            api_key=settings.llm.api_key,      # reuse if you store keys that way
            base_url=settings.llm.base_url,
        )

    if p == "ollama":
        from agent.providers.embeddings.ollama import OllamaEmbeddingProvider
        return OllamaEmbeddingProvider(
            model=settings.embeddings.model,
            dimensions=settings.embeddings.dimensions,
            base_url=settings.llm.base_url or "http://localhost:11434",
        )

    if p == "local":
        from agent.providers.embeddings.local import LocalEmbeddingProvider
        return LocalEmbeddingProvider(
            dimensions=settings.embeddings.dimensions,
        )

    raise ValueError(f"Unknown embedding provider: {p}")