"""Thin RAG chain: retrieve -> grounded, cited answer (streaming or full)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from app.llm.base import ChatMessage
from app.rag.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a knowledgeable movie assistant. Answer the user's question using "
    "ONLY the numbered context passages provided. Cite the passages you rely on "
    "inline using their numbers, e.g. [1] or [2]. If the answer is not in the "
    "context, say you don't have that information rather than guessing. Be concise."
)


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, retriever, llm, k: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def retrieve(self, query: str, k: int | None = None, where: dict | None = None):
        return self.retriever.retrieve(query, k=k or self.k, where=where)

    def _messages(self, query: str, sources: list[RetrievedChunk]) -> list[ChatMessage]:
        context = "\n\n".join(
            f"[{i}] ({s.film_title} — {s.section})\n{s.text}"
            for i, s in enumerate(sources, 1)
        )
        user = f"Context passages:\n{context}\n\nQuestion: {query}"
        return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]

    def stream_answer(self, query: str, sources: list[RetrievedChunk]) -> Iterator[str]:
        """Stream answer tokens for an already-retrieved set of sources."""
        if not sources:
            yield "I couldn't find anything relevant in the movie corpus."
            return
        yield from self.llm.stream(self._messages(query, sources))

    def answer(self, query: str, k: int | None = None, where: dict | None = None) -> RAGResponse:
        sources = self.retrieve(query, k=k, where=where)
        if not sources:
            return RAGResponse(answer="I couldn't find anything relevant in the movie corpus.")
        text = self.llm.complete(self._messages(query, sources))
        return RAGResponse(answer=text, sources=sources)
