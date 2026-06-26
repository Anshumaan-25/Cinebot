"""Build the movie corpus (Part 1).

    python scripts/build_corpus.py            # uses N_FILMS from .env (default 50)
    python scripts/build_corpus.py --n 500    # full corpus
    python scripts/build_corpus.py --n 20 --list popular

Requires TMDB_API_KEY in .env. All fetches are cached, so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `python scripts/build_corpus.py` runnable directly by putting the project
# root (the parent of scripts/) on sys.path before importing the app package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.data.pipeline import CorpusPipeline, Stats
from app.data.tmdb_client import TMDBClient
from app.data.wikipedia import WikipediaSource
from app.logging_config import setup_logging


def _print_summary(stats: Stats, settings, list_name: str, cache_info: dict) -> None:
    pct = (stats.films_processed / stats.films_requested * 100) if stats.films_requested else 0.0
    resolved = stats.wiki_resolved_wikidata + stats.wiki_resolved_search
    avg_chunks = (stats.total_chunks / stats.films_processed) if stats.films_processed else 0.0
    by_source = ", ".join(f"{k}={v}" for k, v in sorted(stats.chunks_by_source.items())) or "none"

    print("\n" + "=" * 60)
    print("  CORPUS BUILD SUMMARY")
    print("=" * 60)
    print(f"  TMDB list           : {list_name}")
    print(f"  Films requested     : {stats.films_requested}")
    print(f"  Films processed     : {stats.films_processed} ({pct:.0f}%)")
    print(f"  Films failed        : {stats.films_failed}")
    print("  ── Wikipedia resolution ──")
    print(f"    via Wikidata      : {stats.wiki_resolved_wikidata}")
    print(f"    via search        : {stats.wiki_resolved_search}")
    print(f"    unresolved        : {stats.wiki_unresolved}")
    print(f"    resolved total    : {resolved}/{stats.films_requested}")
    print(f"  Films with reviews  : {stats.films_with_reviews}")
    print("  ── Chunks ──")
    print(f"    total             : {stats.total_chunks}")
    print(f"    by source         : {by_source}")
    print(f"    avg per film      : {avg_chunks:.1f}")
    print(f"    total text        : {stats.total_chars / 1_000_000:.2f} M chars")
    th, tm = cache_info["tmdb"]
    wh, wm = cache_info["wiki"]
    print("  ── Cache (hits/misses) ──")
    print(f"    tmdb              : {th}/{tm}")
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
    parser.add_argument("--list", default=settings.tmdb_list_endpoint, help="TMDB list endpoint")
    args = parser.parse_args()

    if not settings.has_tmdb:
        print("ERROR: TMDB_API_KEY is not set. Add it to your .env and retry.", file=sys.stderr)
        return 1

    tmdb = TMDBClient(
        settings.tmdb_api_key,
        settings.tmdb_base_url,
        settings.tmdb_cache_dir,
        min_interval=settings.tmdb_min_interval,
        cast_limit=settings.tmdb_cast_limit,
    )
    wiki = WikipediaSource(
        settings.wikipedia_api_url,
        settings.wikidata_api_url,
        settings.wikipedia_cache_dir,
        settings.wikipedia_user_agent,
        min_interval=settings.wikipedia_min_interval,
    )
    pipeline = CorpusPipeline(
        tmdb,
        wiki,
        cast_limit=settings.tmdb_cast_limit,
        review_limit=settings.tmdb_review_limit,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )

    def progress(done: int, total: int, title: str) -> None:
        print(f"  [{done}/{total}] {title}")

    stats = None
    cache_info = {"tmdb": (0, 0), "wiki": (0, 0)}
    try:
        stats = pipeline.run(
            args.n,
            settings.films_path,
            settings.chunks_path,
            list_name=args.list,
            progress=progress,
        )
    finally:
        cache_info["tmdb"] = (tmdb.cache_hits, tmdb.cache_misses)
        cache_info["wiki"] = (wiki.cache_hits, wiki.cache_misses)
        tmdb.close()
        wiki.close()

    _print_summary(stats, settings, args.list, cache_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
