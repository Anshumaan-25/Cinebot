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
from fastapi.responses import JSONResponse, StreamingResponse

from app import __version__
from app.api.schemas import ChatRequest
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
            "omdb": settings.has_omdb,
        },
        "models": {
            "generation": settings.gemini_model,
            "routing": settings.groq_model,
            "embedding": settings.embedding_model,
        },
        "status": "Part 0 scaffold — RAG / KG / memory / orchestration not yet wired",
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Streaming hybrid (GraphRAG) answer (Part 4): link the query to knowledge-
    graph entities, fuse graph facts with focused vector retrieval, and stream a
    grounded, cited Gemini answer followed by sources.

    Orchestration (memory / tools routing) arrives in Part 6.
    """
    if not settings.has_gemini:
        return JSONResponse(
            status_code=503,
            content={"error": "GOOGLE_API_KEY not set — needed for retrieval + generation."},
        )

    # Imported lazily so the app still starts/tests without the SDK side-effects.
    from app.rag.factory import get_hybrid_rag

    rag = get_hybrid_rag()
    if rag.hybrid.retriever.store.count() == 0:
        return JSONResponse(
            status_code=503,
            content={"error": "Vector index is empty. Run: python scripts/build_index.py"},
        )

    def generate():
        try:
            ctx = rag.retrieve(req.message)
        except Exception as exc:  # noqa: BLE001
            logger.exception("hybrid retrieval failed")
            yield f"[retrieval error: {exc}]"
            return
        if not ctx.chunks and not ctx.graph_facts:
            yield "I couldn't find anything relevant in the movie corpus."
            return
        try:
            for token in rag.stream_answer(req.message, ctx):
                yield token
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            yield f"\n[generation error: {exc}]"
        if ctx.graph_facts:
            yield (
                f"\n\n[graph: {len(ctx.graph_facts)} fact(s) — linked "
                f"{len(ctx.linked.people)} people, {len(ctx.linked.films)} films, "
                f"{len(ctx.linked.genres)} genres]"
            )
        if ctx.chunks:
            yield "\n\nSources:\n"
            for i, s in enumerate(ctx.chunks, 1):
                yield f"  [{i}] {s.film_title} — {s.section}\n"

    return StreamingResponse(generate(), media_type="text/plain")
