"""Data contracts produced by Part 1 and consumed by later parts.

``FilmRecord`` (structured) feeds the knowledge graph (Part 3); ``Chunk``
(retrieval unit) feeds the vector store (Part 2). Both carry ``imdb_id`` — the
universal key that links RAG, the graph, and live tools.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilmRecord(BaseModel):
    imdb_id: str
    title: str
    year: int | None = None
    overview: str | None = None
    genres: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    imdb_rating: float | None = None
    wikipedia_title: str | None = None
    wikipedia_url: str | None = None
    resolution_method: str = "none"  # "wikidata" | "omdb_title" | "none"


class Chunk(BaseModel):
    chunk_id: str
    imdb_id: str
    film_title: str
    section: str
    source: str  # "wikipedia" | "omdb_plot"
    text: str
