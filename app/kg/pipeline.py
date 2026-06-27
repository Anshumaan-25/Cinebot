"""Knowledge-graph build pipeline (Part 3).

Part A (no LLM): from ``films.jsonl``, MERGE Film/Person/Genre nodes and
DIRECTED_BY / ACTED_IN / BELONGS_TO_GENRE relationships using the structured
OMDB fields (directors, top-3 cast, genres).

Part B (Gemini): from the Wikipedia 'Cast' chunks in ``chunks.jsonl``, extract
the full cast per film and MERGE additional ACTED_IN edges (with character),
supplementing OMDB's 3-actor limit.

The store and extractor are injected, so the pipeline is testable offline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.data.schema import Chunk, FilmRecord

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


@dataclass
class GraphStats:
    films_added: int = 0
    cast_films_extracted: int = 0
    cast_members_extracted: int = 0
    cast_films_failed: int = 0
    elapsed_s: float = 0.0
    graph: dict = field(default_factory=dict)


def group_cast_chunks(chunks: list[Chunk]) -> dict[str, tuple[str, str]]:
    """Group Wikipedia 'Cast' chunks by imdb_id -> (film_title, combined_text)."""
    grouped: dict[str, dict] = {}
    for c in chunks:
        if c.source != "wikipedia" or "cast" not in c.section.lower():
            continue
        entry = grouped.setdefault(c.imdb_id, {"title": c.film_title, "parts": []})
        entry["parts"].append(c.text)
    return {
        imdb_id: (entry["title"], "\n\n".join(entry["parts"]))
        for imdb_id, entry in grouped.items()
    }


class GraphPipeline:
    def __init__(self, store, extractor=None) -> None:
        self.store = store
        self.extractor = extractor  # None -> skip Part B (cast supplement)

    def run(
        self,
        films_path: Path,
        chunks_path: Path,
        *,
        extract_cast: bool = True,
        progress: ProgressFn | None = None,
    ) -> GraphStats:
        stats = GraphStats()
        started = time.monotonic()

        self.store.setup_constraints()

        # --- Part A: structured films ---
        films = [
            FilmRecord.model_validate_json(line)
            for line in films_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        for film in films:
            self.store.add_film(
                imdb_id=film.imdb_id,
                title=film.title,
                year=film.year,
                imdb_rating=film.imdb_rating,
                genres=film.genres,
                directors=film.directors,
                cast=film.cast,
            )
            stats.films_added += 1
            if progress:
                progress(stats.films_added, len(films), film.title)

        # --- Part B: cast supplement from Wikipedia ---
        if extract_cast and self.extractor is not None:
            chunks = [
                Chunk.model_validate_json(line)
                for line in chunks_path.read_text("utf-8").splitlines()
                if line.strip()
            ]
            cast_by_film = group_cast_chunks(chunks)
            total = len(cast_by_film)
            logger.info("Extracting cast for %d films from Wikipedia 'Cast' sections", total)
            for i, (imdb_id, (title, text)) in enumerate(cast_by_film.items(), 1):
                try:
                    members = self.extractor.extract(imdb_id, title, text)
                except Exception as exc:  # noqa: BLE001 — one film must not abort the batch
                    msg = str(exc)
                    if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "quota" in msg.lower():
                        logger.warning(
                            "Stopping cast extraction — Gemini quota hit after %d/%d films. "
                            "Cached results persist; re-run to resume.",
                            i - 1,
                            total,
                        )
                        break
                    stats.cast_films_failed += 1
                    logger.warning("Cast extraction failed for %r: %s", title, exc)
                    continue
                self.store.add_cast(imdb_id, [(m.actor, m.character) for m in members])
                stats.cast_films_extracted += 1
                stats.cast_members_extracted += len(members)
                if progress:
                    progress(i, total, f"cast: {title}")

        stats.graph = self.store.stats()
        stats.elapsed_s = time.monotonic() - started
        return stats
