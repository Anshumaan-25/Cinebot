"""Corpus build pipeline (OMDB + Wikipedia).

Flow: scrape Wikipedia 'List of highest-grossing films' -> for each film resolve
its IMDb id (Wikidata P345, OMDB-by-title fallback) -> OMDB structured details +
Wikipedia article text -> clean -> chunk -> write ``films.jsonl`` (structured,
for the KG) and ``chunks.jsonl`` (retrieval units, for RAG), both keyed by
``imdb_id``.

The OMDB and Wikipedia sources are injected, so the pipeline is fully testable
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
    films_listed: int = 0
    films_processed: int = 0
    films_failed: int = 0
    imdb_resolved_wikidata: int = 0
    imdb_resolved_omdb: int = 0
    imdb_unresolved: int = 0
    omdb_found: int = 0
    total_chunks: int = 0
    chunks_by_source: dict = field(default_factory=dict)
    total_chars: int = 0
    elapsed_s: float = 0.0

    def record_chunk(self, source: str, chars: int) -> None:
        self.chunks_by_source[source] = self.chunks_by_source.get(source, 0) + 1
        self.total_chunks += 1
        self.total_chars += chars


def _year_from(*values: Optional[str]) -> int | None:
    for value in values:
        if value and len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


def _omdb_list(value: Optional[str], limit: int | None = None) -> list[str]:
    if not value or value == "N/A":
        return []
    items = [x.strip() for x in value.split(",") if x.strip() and x.strip() != "N/A"]
    return items[:limit] if limit else items


def _float_or_none(value: Optional[str]) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class CorpusPipeline:
    def __init__(
        self,
        omdb,
        wiki,
        *,
        list_page: str = "List of highest-grossing films",
        cast_limit: int = 10,
        target_chars: int = 1800,
        overlap_chars: int = 200,
    ) -> None:
        self.omdb = omdb
        self.wiki = wiki
        self.list_page = list_page
        self.cast_limit = cast_limit
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def run(
        self,
        n: int,
        films_path: Path,
        chunks_path: Path,
        *,
        progress: ProgressFn | None = None,
    ) -> Stats:
        stats = Stats()
        started = time.monotonic()

        listings = self.wiki.scrape_film_list(self.list_page, n)
        stats.films_listed = len(listings)

        films_path.parent.mkdir(parents=True, exist_ok=True)
        with films_path.open("w", encoding="utf-8") as ff, chunks_path.open("w", encoding="utf-8") as cf:
            for listing in listings:
                title = listing.get("title") or listing.get("wiki_title") or ""
                try:
                    result = self._process_film(listing, stats)
                except Exception as exc:  # noqa: BLE001 — one film must not abort the batch
                    stats.films_failed += 1
                    logger.warning("Skipping film %r: %s", title, exc)
                    continue
                if result is None:
                    continue  # unresolved IMDb id — counted in stats, nothing to write

                record, chunks = result
                ff.write(record.model_dump_json() + "\n")
                for ch in chunks:
                    cf.write(ch.model_dump_json() + "\n")
                    stats.record_chunk(ch.source, len(ch.text))

                stats.films_processed += 1
                if progress:
                    progress(stats.films_processed, stats.films_listed, record.title)

        stats.elapsed_s = time.monotonic() - started
        return stats

    def _process_film(self, listing: dict, stats: Stats):
        wiki_title = listing["wiki_title"]
        display_title = listing.get("title") or wiki_title

        # 1) IMDb id: Wikidata first, OMDB-by-title fallback.
        imdb_id = self.wiki.imdb_id_from_title(wiki_title)
        if imdb_id:
            method = "wikidata"
            stats.imdb_resolved_wikidata += 1
            details = self.omdb.by_imdb_id(imdb_id)
        else:
            details = self.omdb.by_title(display_title)
            if self.omdb.found(details) and details.get("imdbID"):
                imdb_id = details["imdbID"]
                method = "omdb_title"
                stats.imdb_resolved_omdb += 1
            else:
                stats.imdb_unresolved += 1
                return None

        ok = self.omdb.found(details)
        if ok:
            stats.omdb_found += 1
        record = self._to_record(imdb_id, details if ok else {}, wiki_title, method, display_title)

        # 2) chunks: OMDB plot + whitelisted Wikipedia sections.
        chunks: list[Chunk] = []
        if record.overview:
            chunks += self._chunks(imdb_id, record.title, "Plot", "omdb_plot", record.overview)
        for section, body in self.wiki.fetch_sections(wiki_title).items():
            chunks += self._chunks(imdb_id, record.title, section, "wikipedia", body)

        return record, chunks

    def _to_record(self, imdb_id, details, wiki_title, method, fallback_title) -> FilmRecord:
        title = details.get("Title")
        if not title or title == "N/A":
            title = fallback_title
        plot = details.get("Plot")
        return FilmRecord(
            imdb_id=imdb_id,
            title=title,
            year=_year_from(details.get("Year")),
            overview=plot if plot and plot != "N/A" else None,
            genres=_omdb_list(details.get("Genre")),
            cast=_omdb_list(details.get("Actors"), self.cast_limit),
            directors=_omdb_list(details.get("Director")),
            imdb_rating=_float_or_none(details.get("imdbRating")),
            wikipedia_title=wiki_title,
            wikipedia_url=(
                f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}"
                if wiki_title
                else None
            ),
            resolution_method=method,
        )

    def _chunks(self, imdb_id, title, section, source, text) -> list[Chunk]:
        return make_chunks(
            imdb_id,
            title,
            section,
            source,
            clean_text(text),
            target_chars=self.target_chars,
            overlap_chars=self.overlap_chars,
        )
