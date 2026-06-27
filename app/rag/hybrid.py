"""Hybrid retrieval (GraphRAG): fuse knowledge-graph facts with vector RAG.

Flow: link query -> known graph entities -> structured graph lookups
(films_by_director / filmography / cast_of_film / films_by_genre /
directors_for_genres) -> vector retrieval (focused on the linked films'
imdb_ids when present) -> one fused, grounded Gemini answer.

Both paths are keyed by imdb_id: linked films constrain the Chroma metadata
filter, so graph and vector context describe the same entities.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from app.llm.base import ChatMessage
from app.rag.entities import LinkedEntities
from app.rag.retriever import RetrievedChunk

_MAX_LIST = 30  # cap entities listed per fact line

SYSTEM_PROMPT = (
    "You are a knowledgeable movie assistant. Answer the question using the "
    "KNOWLEDGE-GRAPH FACTS and the RETRIEVED PASSAGES below. The graph facts are "
    "authoritative for relationships (who directed or acted in what, and genres); "
    "use them for relational questions and reason over them when the answer needs "
    "combining facts (e.g. an intersection like 'both X and Y'). Use the passages "
    "for descriptive detail and cite them inline as [1], [2]. If the answer is not "
    "supported by the facts or passages, say you don't have that information."
)


@dataclass
class HybridContext:
    graph_facts: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    imdb_ids: list[str] = field(default_factory=list)
    linked: LinkedEntities = field(default_factory=LinkedEntities)


class HybridRetriever:
    def __init__(self, graph_store, entity_linker, retriever, k: int = 5) -> None:
        self.gs = graph_store
        self.linker = entity_linker
        self.retriever = retriever
        self.k = k

    def retrieve(self, query: str, k: int | None = None) -> HybridContext:
        linked = self.linker.link(query)
        facts: list[str] = []
        imdb_ids: list[str] = []

        # --- people: directing + acting ---
        for person in linked.people:
            directed = self.gs.films_by_director(person)
            if directed:
                facts.append(
                    f"{person} directed: "
                    + ", ".join(f"{d['title']} ({d['year']})" for d in directed[:_MAX_LIST])
                )
            acted = self.gs.filmography(person)
            if acted:
                facts.append(
                    f"{person} acted in: "
                    + ", ".join(f"{a['title']} ({a['year']})" for a in acted[:_MAX_LIST])
                )

        # --- films: cast (and focus the vector search on these) ---
        for title, imdb_id in linked.films:
            imdb_ids.append(imdb_id)
            cast = self.gs.cast_of_film(imdb_id)
            if cast:
                facts.append(
                    f"Cast of {title}: " + ", ".join(c["actor"] for c in cast[:_MAX_LIST])
                )

        # --- genres: member films + director-by-genre (for 'both X and Y') ---
        if linked.genres:
            for genre in linked.genres:
                films = self.gs.films_by_genre(genre)
                if films:
                    facts.append(
                        f"{genre} films: " + ", ".join(f["title"] for f in films[:_MAX_LIST])
                    )
            dirs = self.gs.directors_for_genres(linked.genres)
            if dirs:
                facts.append(
                    "Director genres — "
                    + "; ".join(f"{d['director']}: {', '.join(d['genres'])}" for d in dirs[:_MAX_LIST])
                )

        # --- vector retrieval, focused on linked films when we have any ---
        where = {"imdb_id": {"$in": imdb_ids}} if imdb_ids else None
        chunks = self.retriever.retrieve(query, k=k or self.k, where=where)

        return HybridContext(graph_facts=facts, chunks=chunks, imdb_ids=imdb_ids, linked=linked)


class HybridRAG:
    def __init__(self, hybrid_retriever: HybridRetriever, llm) -> None:
        self.hybrid = hybrid_retriever
        self.llm = llm

    def retrieve(self, query: str) -> HybridContext:
        return self.hybrid.retrieve(query)

    def _messages(self, query: str, ctx: HybridContext) -> list[ChatMessage]:
        graph = "\n".join(f"- {f}" for f in ctx.graph_facts) or "(no matching graph facts)"
        passages = (
            "\n\n".join(
                f"[{i}] ({c.film_title} — {c.section})\n{c.text}"
                for i, c in enumerate(ctx.chunks, 1)
            )
            or "(no passages)"
        )
        user = (
            f"KNOWLEDGE-GRAPH FACTS:\n{graph}\n\n"
            f"RETRIEVED PASSAGES:\n{passages}\n\n"
            f"Question: {query}"
        )
        return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]

    def stream_answer(self, query: str, ctx: HybridContext) -> Iterator[str]:
        yield from self.llm.stream(self._messages(query, ctx))

    def answer(self, query: str) -> str:
        ctx = self.retrieve(query)
        return self.llm.complete(self._messages(query, ctx))
