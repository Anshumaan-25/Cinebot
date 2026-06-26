"""Part 1 — Data pipeline.

TMDB (top films, structured metadata, reviews) + Wikipedia (long-form prose,
resolved via Wikidata) -> clean -> section-aware chunking. Produces
``films.jsonl`` (for the knowledge graph) and ``chunks.jsonl`` (for RAG).

Entry point: ``python scripts/build_corpus.py``.
"""

from app.data.pipeline import CorpusPipeline, Stats

__all__ = ["CorpusPipeline", "Stats"]
