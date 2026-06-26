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
    """Streaming RAG answer (Part 2): retrieve top-k chunks from the vector
    store, stream a grounded, cited Gemini answer, then list the sources.

    Orchestration (memory / KG / tools routing) arrives in Part 6.
    """
    if not settings.has_gemini:
        return JSONResponse(
            status_code=503,
            content={"error": "GOOGLE_API_KEY not set — needed for retrieval + generation."},
        )

    # Imported lazily so the app still starts/tests without the SDK side-effects.
    from app.rag.factory import get_rag

    rag = get_rag()
    if rag.retriever.store.count() == 0:
        return JSONResponse(
            status_code=503,
            content={"error": "Vector index is empty. Run: python scripts/build_index.py"},
        )

    def generate():
        try:
            sources = rag.retrieve(req.message, k=settings.rag_top_k)
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieval failed")
            yield f"[retrieval error: {exc}]"
            return
        if not sources:
            yield "I couldn't find anything relevant in the movie corpus."
            return
        try:
            for token in rag.stream_answer(req.message, sources):
                yield token
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            yield f"\n[generation error: {exc}]"
        yield "\n\nSources:\n"
        for i, s in enumerate(sources, 1):
            yield f"  [{i}] {s.film_title} — {s.section}\n"

    return StreamingResponse(generate(), media_type="text/plain")
