"""Embed a query (RETRIEVAL_QUERY) and fetch the most similar chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    imdb_id: str
    film_title: str
    section: str
    source: str
    text: str
    distance: float


class Retriever:
    def __init__(self, store, embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5, where: dict | None = None) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(query)
        rows = self.store.query(vector, k=k, where=where)
        return [RetrievedChunk(**row) for row in rows]
