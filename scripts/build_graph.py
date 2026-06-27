"""Build the Neo4j knowledge graph (Part 3).

    python scripts/build_graph.py            # films + cast supplement (needs GOOGLE_API_KEY)
    python scripts/build_graph.py --no-cast  # structured films only (no Gemini calls)
    python scripts/build_graph.py --reset    # wipe the graph first

Requires NEO4J_* in .env and a built corpus (data/processed/{films,chunks}.jsonl).
Cast extraction is cached, so re-runs only call Gemini for new films.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.kg.factory import get_cast_extractor, get_graph_store  # noqa: E402
from app.kg.pipeline import GraphPipeline, GraphStats  # noqa: E402
from app.logging_config import setup_logging  # noqa: E402


def _print_summary(stats: GraphStats, settings, did_cast: bool, cast_calls: int) -> None:
    nodes = ", ".join(f"{n['label']}={n['count']}" for n in stats.graph.get("nodes", [])) or "none"
    rels = ", ".join(f"{r['type']}={r['count']}" for r in stats.graph.get("relationships", [])) or "none"
    print("\n" + "=" * 60)
    print("  KNOWLEDGE GRAPH BUILD SUMMARY")
    print("=" * 60)
    print(f"  Films added         : {stats.films_added}")
    if did_cast:
        print(f"  Cast films extracted: {stats.cast_films_extracted}")
        print(f"  Cast members found  : {stats.cast_members_extracted}")
        print(f"  Cast films failed   : {stats.cast_films_failed}")
        print(f"  Gemini calls        : {cast_calls}")
    else:
        print("  Cast supplement     : skipped (--no-cast or no GOOGLE_API_KEY)")
    print("  ── Graph ──")
    print(f"    nodes             : {nodes}")
    print(f"    relationships     : {rels}")
    print(f"  Elapsed             : {stats.elapsed_s:.1f}s")
    print(f"  Neo4j               : {settings.neo4j_uri}")
    print("=" * 60)


def main() -> int:
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()

    parser = argparse.ArgumentParser(description="Build the Neo4j knowledge graph (Part 3).")
    parser.add_argument("--reset", action="store_true", help="DETACH DELETE all nodes first")
    parser.add_argument("--no-cast", action="store_true", help="skip Gemini cast extraction")
    args = parser.parse_args()

    if not settings.has_neo4j:
        print("ERROR: Neo4j is not configured (set NEO4J_URI / NEO4J_PASSWORD).", file=sys.stderr)
        return 1
    if not settings.films_path.exists():
        print(f"ERROR: {settings.films_path} not found. Run scripts/build_corpus.py first.", file=sys.stderr)
        return 1

    do_cast = not args.no_cast
    if do_cast and not settings.has_gemini:
        print("WARNING: GOOGLE_API_KEY not set — skipping cast extraction (Part A only).", file=sys.stderr)
        do_cast = False

    store = get_graph_store()
    try:
        store.verify()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot connect to Neo4j: {exc}", file=sys.stderr)
        store.close()
        return 1

    extractor = get_cast_extractor() if do_cast else None
    if args.reset:
        print("Resetting graph (DETACH DELETE all)...")
        store.reset()

    pipeline = GraphPipeline(store, extractor)

    def progress(done: int, total: int, label: str) -> None:
        print(f"  [{done}/{total}] {label}")

    try:
        stats = pipeline.run(
            settings.films_path,
            settings.chunks_path,
            extract_cast=do_cast,
            progress=progress,
        )
    finally:
        cast_calls = extractor.api_calls if extractor else 0
        store.close()

    _print_summary(stats, settings, do_cast, cast_calls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
