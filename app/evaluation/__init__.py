"""Part 9 — Evaluation framework (lite retrieval eval landed early, in Part 2).

Hit@1 / Hit@k / MRR over a small gold set (context-relevance proxy). RAGAS-style
faithfulness and answer-correctness come in the full Part 9.
"""

from app.evaluation.retrieval_eval import (
    EvalResult,
    GraphEvalResult,
    evaluate,
    evaluate_graph,
    load_gold,
)

__all__ = ["EvalResult", "GraphEvalResult", "evaluate", "evaluate_graph", "load_gold"]
