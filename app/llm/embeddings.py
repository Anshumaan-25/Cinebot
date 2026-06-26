"""Embeddings via Google ``text-embedding-004`` — used for BOTH corpus and query.

Project constraint: a single embedding model end-to-end (no mixed models). We do
vary ``task_type`` (RETRIEVAL_DOCUMENT for the corpus, RETRIEVAL_QUERY for user
queries) — that is the *same* model, just told how the text will be used, which
improves retrieval quality. Results are batched and cached on disk.
"""

from __future__ import annotations

import numpy as np
from google import genai
from google.genai import types

from app.llm.embedding_cache import EmbeddingCache
from app.llm.retry import with_backoff

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"


class GeminiEmbedder:
    """Embed text with ``text-embedding-004``, with batching + disk caching."""

    def __init__(
        self,
        api_key: str | None,
        model: str = "text-embedding-004",
        dim: int = 768,
        cache: EmbeddingCache | None = None,
        batch_size: int = 100,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env "
                "(get a free key at https://aistudio.google.com/app/apikey)."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dim = dim
        self._cache = cache
        self._batch_size = max(1, batch_size)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """Embed corpus chunks (RETRIEVAL_DOCUMENT)."""
        return self._embed(list(texts), DOCUMENT_TASK)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query (RETRIEVAL_QUERY)."""
        return self._embed([text], QUERY_TASK)[0]

    # ----- internals -----
    def _embed(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        if not texts:
            return []
        keys = [EmbeddingCache.make_key(self._model, task_type, t) for t in texts]
        results: list[np.ndarray | None] = [None] * len(texts)

        # 1) serve what we can from cache
        missing: list[int] = []
        for i, key in enumerate(keys):
            cached = self._cache.get(key) if self._cache else None
            if cached is not None:
                results[i] = cached
            else:
                missing.append(i)

        # 2) embed cache-misses in batches, then persist
        for start in range(0, len(missing), self._batch_size):
            idx_batch = missing[start : start + self._batch_size]
            vectors = self._embed_batch([texts[i] for i in idx_batch], task_type)
            for i, vec in zip(idx_batch, vectors):
                results[i] = vec
            if self._cache:
                self._cache.put_many([(keys[i], results[i]) for i in idx_batch])

        return [r for r in results if r is not None]

    @with_backoff()
    def _embed_batch(self, batch: list[str], task_type: str) -> list[np.ndarray]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [np.asarray(e.values, dtype=np.float32) for e in response.embeddings]
