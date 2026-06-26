"""Part 1 — Data pipeline.

Wikipedia 'List of highest-grossing films' (the film list) + OMDB (structured
metadata) + Wikipedia articles (long-form prose), with IMDb ids from Wikidata.
Flow: scrape list -> resolve imdb_id -> OMDB details + article text -> clean ->
section-aware chunking. Produces ``films.jsonl`` (for the knowledge graph) and
``chunks.jsonl`` (for RAG), both keyed by ``imdb_id``.

Entry point: ``python scripts/build_corpus.py``.
"""

from app.data.pipeline import CorpusPipeline, Stats

__all__ = ["CorpusPipeline", "Stats"]
