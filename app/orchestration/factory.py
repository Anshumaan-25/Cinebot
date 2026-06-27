"""Cached factory for the production chat graph."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.factory import get_llm, get_router_llm
from app.memory.factory import get_memory_manager
from app.orchestration.graph import build_chat_graph
from app.rag.factory import get_hybrid_retriever
from app.tools.factory import get_tool_runner


@lru_cache
def get_chat_graph():
    s = get_settings()
    memory = get_memory_manager()
    router_llm = get_router_llm() if s.has_groq else get_llm()
    hybrid = get_hybrid_retriever()
    # Production generator = Gemini; fall back to Groq when Gemini quota is spent.
    gen_llm = get_llm() if s.has_gemini else get_router_llm()
    fallback_llm = get_router_llm() if (s.has_groq and s.has_gemini) else None
    tools = get_tool_runner()  # live OMDB + Wikipedia search (Part 7)
    return build_chat_graph(
        memory=memory,
        router_llm=router_llm,
        hybrid=hybrid,
        gen_llm=gen_llm,
        fallback_llm=fallback_llm,
        tools=tools,
    )
