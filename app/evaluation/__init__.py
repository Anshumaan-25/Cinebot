"""Part 9 — Evaluation framework.

Two complementary layers:

- **Retrieval eval** (landed early, Part 2): Hit@1 / Hit@k / MRR + graph entity
  recall — cheap, deterministic, no LLM.
- **RAGAS-style eval** (Part 9): an LLM judge (Groq) scores generated answers for
  faithfulness, answer relevance, context relevance and answer correctness.
"""

from app.evaluation.llm_judge import LLMJudge
from app.evaluation.ragas_eval import (
    RagasEvaluator,
    RagasReport,
    RagasRow,
    hybrid_answerer,
    render_context,
)
from app.evaluation.retrieval_eval import (
    EvalResult,
    GraphEvalResult,
    evaluate,
    evaluate_graph,
    load_gold,
)

__all__ = [
    "EvalResult",
    "GraphEvalResult",
    "evaluate",
    "evaluate_graph",
    "load_gold",
    "LLMJudge",
    "RagasEvaluator",
    "RagasReport",
    "RagasRow",
    "hybrid_answerer",
    "render_context",
]
