"""Run the lite retrieval eval (Part 2): Hit@1 / Hit@k / MRR over the gold set.

    python scripts/run_eval.py

Requires GOOGLE_API_KEY and a built index (scripts/build_index.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.evaluation.retrieval_eval import evaluate, load_gold  # noqa: E402
from app.logging_config import setup_logging  # noqa: E402
from app.rag.factory import get_retriever, get_vector_store  # noqa: E402


def main() -> int:
    setup_logging()
    settings = get_settings()

    if not settings.has_gemini:
        print("ERROR: GOOGLE_API_KEY is not set.", file=sys.stderr)
        return 1
    if get_vector_store().count() == 0:
        print("ERROR: vector index is empty. Run scripts/build_index.py first.", file=sys.stderr)
        return 1

    gold = load_gold()
    result = evaluate(get_retriever(), gold, k=settings.rag_top_k)

    print("\n" + "=" * 60)
    print("  RETRIEVAL EVAL (lite)")
    print("=" * 60)
    print(f"  Gold queries : {result.n}")
    print(f"  Hit@1        : {result.hit_at_1:.0%}")
    print(f"  Hit@{result.k}        : {result.hit_at_k:.0%}")
    print(f"  MRR          : {result.mrr:.3f}")
    print("  ── per query ──")
    for r in result.rows:
        mark = "OK  " if r["rank"] else "MISS"
        rank = r["rank"] if r["rank"] else "-"
        print(f"   [{mark}] rank={rank}  {r['query']}")
        print(f"          top: {r['top']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
