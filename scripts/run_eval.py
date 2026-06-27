"""Run the retrieval eval: vector quality (Part 2) + graph contribution (Part 4).

    python scripts/run_eval.py

Requires GOOGLE_API_KEY + a built index. The relational (graph) section also
needs the Neo4j graph (scripts/build_graph.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.evaluation.retrieval_eval import evaluate, evaluate_graph, load_gold  # noqa: E402
from app.logging_config import setup_logging  # noqa: E402
from app.rag.factory import get_hybrid_retriever, get_retriever, get_vector_store  # noqa: E402


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
    lookup = [g for g in gold if g.get("type", "lookup") == "lookup"]
    relational = [g for g in gold if g.get("type") == "relational"]

    vres = evaluate(get_retriever(), lookup, k=settings.rag_top_k)
    gres = None
    if relational and settings.has_neo4j:
        gres = evaluate_graph(get_hybrid_retriever(), relational)

    print("\n" + "=" * 64)
    print("  RETRIEVAL EVAL")
    print("=" * 64)
    print(f"  VECTOR — descriptive 'lookup' queries (n={vres.n})")
    print(f"    Hit@1 : {vres.hit_at_1:.0%}    Hit@{vres.k} : {vres.hit_at_k:.0%}    MRR : {vres.mrr:.3f}")
    if gres is not None:
        print(f"\n  GRAPH — relational/multi-hop queries (n={gres.n})")
        print(f"    entity recall : {gres.entity_recall:.0%}  (answers vector RAG alone can't surface)")
        for r in gres.rows:
            mark = "OK  " if r["recall"] == 1.0 else ("PART" if r["found"] else "MISS")
            print(f"     [{mark}] {r['query']}")
            print(f"            expected {r['expected']} -> found {r['found']}  ({r['n_facts']} graph facts)")
    elif relational:
        print("\n  GRAPH — skipped (Neo4j not configured)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
