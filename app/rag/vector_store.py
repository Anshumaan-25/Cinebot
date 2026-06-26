"""Chroma vector store (persistent) for RAG chunks.

We pass our OWN ``text-embedding-004`` vectors (no Chroma default embedder) and
key each chunk by ``chunk_id`` with scalar metadata (imdb_id/film_title/section/
source) so we can do native metadata-filtered retrieval (used in Part 4).
"""

from __future__ import annotations

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.data.schema import Chunk

_COSINE = {"hnsw:space": "cosine"}


class VectorStore:
    def __init__(self, path, collection: str = "film_chunks") -> None:
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._name = collection
        self._collection = self._client.get_or_create_collection(name=collection, metadata=_COSINE)

    def recreate(self) -> None:
        """Drop and recreate the collection for a clean rebuild."""
        try:
            self._client.delete_collection(self._name)
        except Exception:  # noqa: BLE001 — collection may not exist yet
            pass
        self._collection = self._client.get_or_create_collection(name=self._name, metadata=_COSINE)

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk], embeddings: list, batch_size: int = 1000) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vecs = embeddings[start : start + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[np.asarray(v, dtype=np.float32).tolist() for v in vecs],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "imdb_id": c.imdb_id,
                        "film_title": c.film_title,
                        "section": c.section,
                        "source": c.source,
                    }
                    for c in batch
                ],
            )

    def query(self, embedding, k: int = 5, where: dict | None = None) -> list[dict]:
        res = self._collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for i in range(len(ids)):
            m = metas[i] or {}
            out.append(
                {
                    "chunk_id": ids[i],
                    "imdb_id": m.get("imdb_id"),
                    "film_title": m.get("film_title"),
                    "section": m.get("section"),
                    "source": m.get("source"),
                    "text": docs[i],
                    "distance": dists[i],
                }
            )
        return out
