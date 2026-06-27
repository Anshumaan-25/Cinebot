"""Lite retrieval eval: Hit@1, Hit@k, MRR over a small gold set.

Measures whether retrieval surfaces chunks from the EXPECTED film (by imdb_id) —
a context-relevance proxy. Full RAGAS-style faithfulness/answer-correctness is
Part 9; this exists so we can measure retrieval quality from Part 2 onward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

GOLD_PATH = Path(__file__).with_name("gold_set.json")


def load_gold(path: Path | str = GOLD_PATH) -> list[dict]:
    return json.loads(Path(path).read_text("utf-8"))


@dataclass
class EvalResult:
    n: int
    k: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    rows: list = field(default_factory=list)


def evaluate(retriever, gold: list[dict], k: int = 5) -> EvalResult:
    hits1 = hitsk = 0
    reciprocal = 0.0
    rows = []
    for item in gold:
        results = retriever.retrieve(item["query"], k=k)
        rank = next(
            (i for i, r in enumerate(results, 1) if r.imdb_id == item["imdb_id"]),
            None,
        )
        if rank == 1:
            hits1 += 1
        if rank is not None:
            hitsk += 1
            reciprocal += 1.0 / rank
        rows.append(
            {
                "query": item["query"],
                "expected": item["imdb_id"],
                "rank": rank,
                "top": [(r.film_title, r.section) for r in results[:3]],
            }
        )
    n = len(gold)
    denom = n or 1
    return EvalResult(
        n=n,
        k=k,
        hit_at_1=hits1 / denom,
        hit_at_k=hitsk / denom,
        mrr=reciprocal / denom,
        rows=rows,
    )


@dataclass
class GraphEvalResult:
    n: int
    entity_recall: float  # avg fraction of expected entities surfaced by the graph path
    rows: list = field(default_factory=list)


def evaluate_graph(hybrid_retriever, gold: list[dict]) -> GraphEvalResult:
    """Measure the GRAPH's contribution on relational queries: does the hybrid
    retriever's graph path surface the expected entities? (Vector RAG alone can't
    answer these — they need relationship traversal.)"""
    total = 0.0
    rows = []
    for item in gold:
        ctx = hybrid_retriever.retrieve(item["query"])
        facts = " ".join(ctx.graph_facts).lower()
        expected = item.get("expected", [])
        found = [e for e in expected if e.lower() in facts]
        recall = (len(found) / len(expected)) if expected else 0.0
        total += recall
        rows.append(
            {
                "query": item["query"],
                "expected": expected,
                "found": found,
                "recall": recall,
                "n_facts": len(ctx.graph_facts),
            }
        )
    n = len(gold)
    return GraphEvalResult(n=n, entity_recall=total / (n or 1), rows=rows)
