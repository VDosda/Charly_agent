from __future__ import annotations

from agent.config.settings import Settings
from agent.providers.llm.base import LLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """
    Factory that returns the configured LLM provider implementation.

    This keeps your core runtime independent of any provider SDK.
    """
    p = settings.llm.provider

    if p == "openai":
        from agent.providers.llm.openai import OpenAIProvider
        return OpenAIProvider(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    if p == "anthropic":
        from agent.providers.llm.anthropic import AnthropicProvider
        return AnthropicProvider(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    if p == "ollama":
        from agent.providers.llm.ollama import OllamaProvider
        return OllamaProvider(
            model=settings.llm.model,
            base_url=settings.llm.base_url or "http://localhost:11434",
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {p}")