"""Factory helpers returning cached, ready-to-use clients.

Use these instead of constructing providers directly so the rest of the app
stays provider-agnostic and we share a single instance per process.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLMClient
from app.llm.embedding_cache import EmbeddingCache
from app.llm.embeddings import GeminiEmbedder
from app.llm.gemini import GeminiLLM
from app.llm.groq_llm import GroqLLM


@lru_cache
def get_llm() -> LLMClient:
    """Main generation/reasoning model (Gemini 2.0 Flash)."""
    s = get_settings()
    return GeminiLLM(s.google_api_key, s.gemini_model)


@lru_cache
def get_router_llm() -> LLMClient:
    """Fast, cheap model for routing decisions (Groq Llama 3.3 70B)."""
    s = get_settings()
    return GroqLLM(s.groq_api_key, s.groq_model)


@lru_cache
def get_embedder() -> GeminiEmbedder:
    """Embedder (text-embedding-004) with a shared on-disk cache."""
    s = get_settings()
    cache = EmbeddingCache(s.embedding_cache_path)
    return GeminiEmbedder(
        s.google_api_key,
        model=s.embedding_model,
        dim=s.embedding_dim,
        cache=cache,
    )
