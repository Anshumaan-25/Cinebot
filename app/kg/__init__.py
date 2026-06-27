"""Part 3 — Knowledge graph.

Builds a Neo4j graph of Film/Person/Genre entities and DIRECTED_BY / ACTED_IN /
BELONGS_TO_GENRE relationships from ``films.jsonl`` (structured OMDB data), with
the cast supplemented by Gemini extraction over the Wikipedia 'Cast' chunks.
``GraphStore`` provides querying; Part 4 fuses this with vector retrieval.

Entry point: ``python scripts/build_graph.py``.
"""

from app.kg.factory import get_cast_extractor, get_graph_store
from app.kg.graph_store import GraphStore
from app.kg.pipeline import GraphPipeline, GraphStats, group_cast_chunks

__all__ = [
    "GraphStore",
    "GraphPipeline",
    "GraphStats",
    "group_cast_chunks",
    "get_graph_store",
    "get_cast_extractor",
]
