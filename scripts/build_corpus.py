"""Build the movie corpus (Part 1).

    python scripts/build_corpus.py            # uses N_FILMS from .env (default 50)
    python scripts/build_corpus.py --n 200    # larger corpus

Requires OMDB_API_KEY in .env. All fetches are cached, so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `python scripts/build_corpus.py` runnable directly by putting the project
# root (the parent of scripts/) on sys.path before importing the app package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.data.omdb_client import OMDBClient  # noqa: E402
from app.data.pipeline import CorpusPipeline, Stats  # noqa: E402
from app.data.wikipedia import WikipediaSource  # noqa: E402
from app.logging_config import setup_logging  # noqa: E402


def _print_summary(stats: Stats, settings, cache_info: dict) -> None:
    pct = (stats.films_processed / stats.films_listed * 100) if stats.films_listed else 0.0
    resolved = stats.imdb_resolved_wikidata + stats.imdb_resolved_omdb
    avg_chunks = (stats.total_chunks / stats.films_processed) if stats.films_processed else 0.0
    by_source = ", ".join(f"{k}={v}" for k, v in sorted(stats.chunks_by_source.items())) or "none"
    oh, om = cache_info["omdb"]
    wh, wm = cache_info["wiki"]

    print("\n" + "=" * 60)
    print("  CORPUS BUILD SUMMARY")
    print("=" * 60)
    print(f"  Film list source    : {settings.highest_grossing_page}")
    print(f"  Films listed        : {stats.films_listed}")
    print(f"  Films processed     : {stats.films_processed} ({pct:.0f}%)")
    print(f"  Films failed        : {stats.films_failed}")
    print("  ── IMDb id resolution ──")
    print(f"    via Wikidata      : {stats.imdb_resolved_wikidata}")
    print(f"    via OMDB title    : {stats.imdb_resolved_omdb}")
    print(f"    unresolved        : {stats.imdb_unresolved}")
    print(f"    resolved total    : {resolved}/{stats.films_listed}")
    print(f"  OMDB details found  : {stats.omdb_found}")
    print("  ── Chunks ──")
    print(f"    total             : {stats.total_chunks}")
    print(f"    by source         : {by_source}")
    print(f"    avg per film      : {avg_chunks:.1f}")
    print(f"    total text        : {stats.total_chars / 1_000_000:.2f} M chars")
    print("  ── Cache (hits/misses) ──")
    print(f"    omdb              : {oh}/{om}")
    print(f"    wikipedia         : {wh}/{wm}")
    print(f"  Elapsed             : {stats.elapsed_s:.1f}s")
    print("  ── Output ──")
    print(f"    films             : {settings.films_path}")
    print(f"    chunks            : {settings.chunks_path}")
    print("=" * 60)


def main() -> int:
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()

    parser = argparse.ArgumentParser(description="Build the movie corpus (Part 1).")
    parser.add_argument("--n", type=int, default=settings.n_films, help="number of films")
    args = parser.parse_args()

    if not settings.has_omdb:
        print("ERROR: OMDB_API_KEY is not set. Add it to your .env and retry.", file=sys.stderr)
        return 1

    omdb = OMDBClient(
        settings.omdb_api_key,
        settings.omdb_base_url,
        settings.omdb_cache_dir,
        min_interval=settings.omdb_min_interval,
    )
    wiki = WikipediaSource(
        settings.wikipedia_api_url,
        settings.wikidata_api_url,
        settings.wikipedia_cache_dir,
        settings.wikipedia_user_agent,
        min_interval=settings.wikipedia_min_interval,
    )
    pipeline = CorpusPipeline(
        omdb,
        wiki,
        list_page=settings.highest_grossing_page,
        cast_limit=settings.omdb_cast_limit,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )

    def progress(done: int, total: int, title: str) -> None:
        print(f"  [{done}/{total}] {title}")

    stats = None
    cache_info = {"omdb": (0, 0), "wiki": (0, 0)}
    try:
        stats = pipeline.run(args.n, settings.films_path, settings.chunks_path, progress=progress)
    finally:
        cache_info["omdb"] = (omdb.cache_hits, omdb.cache_misses)
        cache_info["wiki"] = (wiki.cache_hits, wiki.cache_misses)
        omdb.close()
        wiki.close()

    _print_summary(stats, settings, cache_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
