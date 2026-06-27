"""Entity linking: match a query against the KNOWN entities in the graph.

For a bounded domain (50 films, ~11 genres, ~500 people) this is faster and more
reliable than a freeform extraction LLM — every linked entity resolves directly
to a real graph node (and films carry their imdb_id), with no extra API call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LinkedEntities:
    people: list[str] = field(default_factory=list)
    films: list[tuple[str, str]] = field(default_factory=list)  # (title, imdb_id)
    genres: list[str] = field(default_factory=list)

    def any(self) -> bool:
        return bool(self.people or self.films or self.genres)


class EntityLinker:
    def __init__(self, graph_store) -> None:
        self._gs = graph_store
        self._loaded = False
        self._genres: list[tuple[str, str]] = []
        self._people: list[tuple[str, str]] = []
        self._films: list[tuple[str, str, str]] = []  # (lower_title, title, imdb_id)

    def _load(self) -> None:
        if self._loaded:
            return
        self._genres = [(g.lower(), g) for g in self._gs.all_genres()]
        self._people = [(p.lower(), p) for p in self._gs.all_people()]
        self._films = [
            (f["title"].lower(), f["title"], f["imdb_id"])
            for f in self._gs.all_film_titles()
            if f.get("title")
        ]
        # Longer names first so we prefer the most specific match.
        self._people.sort(key=lambda t: len(t[0]), reverse=True)
        self._films.sort(key=lambda t: len(t[0]), reverse=True)
        self._loaded = True

    def link(self, query: str) -> LinkedEntities:
        self._load()
        q = query.lower()
        genres = [orig for low, orig in self._genres if re.search(rf"\b{re.escape(low)}\b", q)]
        people = [orig for low, orig in self._people if low in q]
        films = [(title, imdb) for low, title, imdb in self._films if low in q]
        return LinkedEntities(people=people, films=films, genres=genres)
