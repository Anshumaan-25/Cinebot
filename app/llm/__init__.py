"""LLM & embedding layer.

Provider-agnostic interfaces so generation/routing/embedding providers can be
swapped without touching the rest of the app. Concrete free-tier providers:
Gemini 2.0 Flash (generation), Groq Llama 3.3 70B (routing), text-embedding-004
(embeddings). Use the factory helpers instead of constructing clients directly.
"""

from app.llm.base import ChatMessage, LLMClient
from app.llm.factory import get_embedder, get_llm, get_router_llm

__all__ = [
    "ChatMessage",
    "LLMClient",
    "get_llm",
    "get_router_llm",
    "get_embedder",
]
