"""Corpus build pipeline: TMDB list -> details/ids/reviews + Wikipedia -> clean
-> chunk -> write ``films.jsonl`` (structured, for the KG) and ``chunks.jsonl``
(retrieval units, for RAG).

The TMDB and Wikipedia sources are injected, so the pipeline is fully testable
offline with fakes. A per-film try/except keeps one bad film from killing a run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.data.chunking import make_chunks
from app.data.cleaning import clean_text
from app.data.schema import Chunk, FilmRecord

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


@dataclass
class Stats:
    films_requested: int = 0
    films_processed: int = 0
    films_failed: int = 0
    wiki_resolved_wikidata: int = 0
    wiki_resolved_search: int = 0
    wiki_unresolved: int = 0
    films_with_reviews: int = 0
    total_chunks: int = 0
    chunks_by_source: dict = field(default_factory=dict)
    total_chars: int = 0
    elapsed_s: float = 0.0

    def record_chunk(self, source: str, chars: int) -> None:
        self.chunks_by_source[source] = self.chunks_by_source.get(source, 0) + 1
        self.total_chunks += 1
        self.total_chars += chars


def _year_from(*date_strings: Optional[str]) -> int | None:
    for value in date_strings:
        if value and len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


class CorpusPipeline:
    def __init__(
        self,
        tmdb,
        wiki,
        *,
        cast_limit: int = 10,
        review_limit: int = 10,
        target_chars: int = 1800,
        overlap_chars: int = 200,
    ) -> None:
        self.tmdb = tmdb
        self.wiki = wiki
        self.cast_limit = cast_limit
        self.review_limit = review_limit
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def run(
        self,
        n: int,
        films_path: Path,
        chunks_path: Path,
        *,
        list_name: str = "top_rated",
        progress: ProgressFn | None = None,
    ) -> Stats:
        stats = Stats()
        started = time.monotonic()

        films = self.tmdb.top_films(n, list_name=list_name)
        stats.films_requested = len(films)

        films_path.parent.mkdir(parents=True, exist_ok=True)
        with films_path.open("w", encoding="utf-8") as ff, chunks_path.open("w", encoding="utf-8") as cf:
            for film in films:
                title = film.get("title") or film.get("name") or ""
                try:
                    record, chunks = self._process_film(film, stats)
                except Exception as exc:  # noqa: BLE001 — one film must not abort the batch
                    stats.films_failed += 1
                    logger.warning("Skipping film %r (%s): %s", title, film.get("id"), exc)
                    continue

                ff.write(record.model_dump_json() + "\n")
                for ch in chunks:
                    cf.write(ch.model_dump_json() + "\n")
                    stats.record_chunk(ch.source, len(ch.text))

                stats.films_processed += 1
                if progress:
                    progress(stats.films_processed, stats.films_requested, title)

        stats.elapsed_s = time.monotonic() - started
        return stats

    def _process_film(self, film: dict, stats: Stats) -> tuple[FilmRecord, list[Chunk]]:
        tmdb_id = int(film["id"])
        title = film.get("title") or film.get("name") or ""

        details = self.tmdb.film_details(tmdb_id)
        external = self.tmdb.external_ids(tmdb_id)
        wikidata_id = external.get("wikidata_id") or None
        imdb_id = external.get("imdb_id") or None

        wiki_title, method = self.wiki.resolve_title(wikidata_id, title)
        if method == "wikidata":
            stats.wiki_resolved_wikidata += 1
        elif method == "search":
            stats.wiki_resolved_search += 1
        else:
            stats.wiki_unresolved += 1

        credits = details.get("credits", {})
        record = FilmRecord(
            tmdb_id=tmdb_id,
            title=title,
            year=_year_from(film.get("release_date"), details.get("release_date")),
            overview=details.get("overview") or None,
            genres=[g["name"] for g in details.get("genres", []) if g.get("name")],
            cast=[c["name"] for c in credits.get("cast", [])[: self.cast_limit] if c.get("name")],
            directors=[c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"],
            imdb_id=imdb_id,
            wikidata_id=wikidata_id,
            wikipedia_title=wiki_title,
            wikipedia_url=(
                f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}"
                if wiki_title
                else None
            ),
            resolution_method=method,
        )

        chunks: list[Chunk] = []

        if record.overview:
            chunks += self._chunks(tmdb_id, title, "Overview", "tmdb_overview", record.overview)

        if wiki_title:
            for section, body in self.wiki.fetch_sections(wiki_title).items():
                chunks += self._chunks(tmdb_id, title, section, "wikipedia", body)

        reviews = self.tmdb.film_reviews(tmdb_id, self.review_limit)
        if reviews:
            stats.films_with_reviews += 1
        for i, review in enumerate(reviews):
            chunks += self._chunks(tmdb_id, title, f"Review {i + 1}", "tmdb_review", review)

        return record, chunks

    def _chunks(self, tmdb_id: int, title: str, section: str, source: str, text: str) -> list[Chunk]:
        return make_chunks(
            tmdb_id,
            title,
            section,
            source,
            clean_text(text),
            target_chars=self.target_chars,
            overlap_chars=self.overlap_chars,
        )
