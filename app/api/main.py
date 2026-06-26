"""FastAPI application (Part 0 skeleton).

Run with:  uvicorn app.api.main:app --reload
Endpoints:
  GET  /health  — liveness probe
  GET  /info    — which providers/models are configured (never leaks key values)
  POST /chat    — direct Gemini reply for now; full orchestration arrives in Part 6
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app import __version__
from app.api.schemas import ChatRequest, ChatResponse
from app.config import get_settings
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()
settings.ensure_dirs()

app = FastAPI(
    title="Memory-Augmented Chatbot with Knowledge Graph & Hybrid RAG",
    version=__version__,
    description="Part 0 scaffold — RAG, knowledge graph, memory and tools land in later parts.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/info")
def info() -> dict:
    """Report configuration without ever exposing secret values."""
    return {
        "version": __version__,
        "providers_configured": {
            "gemini": settings.has_gemini,
            "groq": settings.has_groq,
            "neo4j_aura": settings.has_neo4j,
            "tmdb": settings.has_tmdb,
        },
        "models": {
            "generation": settings.gemini_model,
            "routing": settings.groq_model,
            "embedding": settings.embedding_model,
        },
        "status": "Part 0 scaffold — RAG / KG / memory / orchestration not yet wired",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not settings.has_gemini:
        return ChatResponse(
            reply="No GOOGLE_API_KEY configured — set it in .env to enable live Gemini replies.",
            provider="none",
            note="Part 0 scaffold",
        )

    # Imported lazily so the app still starts/tests without the SDK side-effects.
    from app.llm.factory import get_llm

    try:
        reply = get_llm().generate(
            req.message,
            system=(
                "You are a helpful assistant specialised in movies. "
                "This is a Part 0 scaffold: you do not yet have RAG, a knowledge "
                "graph, tools or long-term memory."
            ),
        )
    except Exception as exc:  # surface provider errors cleanly to the caller
        logger.exception("Gemini call failed")
        return ChatResponse(reply=f"LLM error: {exc}", provider="gemini", note="error")

    return ChatResponse(
        reply=reply,
        provider="gemini",
        note="Direct LLM — orchestration (RAG/KG/memory/tools) arrives in Part 6.",
    )
