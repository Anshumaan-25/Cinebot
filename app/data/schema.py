"""Data contracts produced by Part 1 and consumed by later parts.

``FilmRecord`` (structured) feeds the knowledge graph (Part 3); ``Chunk``
(retrieval unit) feeds the vector store (Part 2). Both carry ``tmdb_id`` — the
universal key that links RAG, the graph, and live tools.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilmRecord(BaseModel):
    tmdb_id: int
    title: str
    year: int | None = None
    overview: str | None = None
    genres: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    imdb_id: str | None = None
    wikidata_id: str | None = None
    wikipedia_title: str | None = None
    wikipedia_url: str | None = None
    resolution_method: str = "none"  # "wikidata" | "search" | "none"


class Chunk(BaseModel):
    chunk_id: str
    tmdb_id: int
    film_title: str
    section: str
    source: str  # "wikipedia" | "tmdb_overview" | "tmdb_review"
    text: str
