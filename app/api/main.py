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
    """Streaming answer via the LangGraph orchestration (Part 6).

    The graph recalls per-user memory, routes the query (Groq), retrieves hybrid
    GraphRAG context, generates a personalized cited answer (Gemini, Groq
    fallback), and persists the turn — so memory personalizes every reply.
    """
    if not (settings.has_gemini or settings.has_groq):
        return JSONResponse(
            status_code=503,
            content={"error": "No LLM configured (set GOOGLE_API_KEY and/or GROQ_API_KEY)."},
        )

    # Imported lazily so the app still starts/tests without the SDK side-effects.
    from app.orchestration.factory import get_chat_graph

    graph = get_chat_graph()
    initial = {"user_id": req.user_id, "query": req.message}

    def generate():
        final_state: dict = {}
        try:
            for mode, data in graph.stream(initial, stream_mode=["custom", "values"]):
                if mode == "custom":
                    yield data  # an answer token
                elif mode == "values":
                    final_state = data  # latest full state (last = final)
        except Exception as exc:  # noqa: BLE001
            logger.exception("orchestration failed")
            yield f"\n[error: {exc}]"
            return

        route = final_state.get("route")
        passages = final_state.get("passages", [])
        if route:
            yield f"\n\n[route: {route}]"
        if final_state.get("tool_results"):
            yield "\n[live tools consulted: OMDB / Wikipedia]"
        if passages:
            yield "\n\nSources:\n"
            for i, p in enumerate(passages, 1):
                yield f"  [{i}] {p['film_title']} — {p['section']}\n"

    return StreamingResponse(generate(), media_type="text/plain")
