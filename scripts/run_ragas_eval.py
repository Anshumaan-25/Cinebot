"""RAGAS-style evaluation: LLM-judged faithfulness / relevance / correctness.

    python scripts/run_ragas_eval.py

For each gold question we generate an answer over real hybrid-GraphRAG context
and score it with an LLM judge (Groq — Gemini's daily quota is spent, and Groq
is a separate free quota). Requires GROQ_API_KEY + a built index; the graph
context also benefits from Neo4j (scripts/build_graph.py).

The answer generator defaults to Gemini when available, else Groq. The judge is
always Groq.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.evaluation.llm_judge import LLMJudge  # noqa: E402
from app.evaluation.ragas_eval import RagasEvaluator, hybrid_answerer  # noqa: E402
from app.evaluation.retrieval_eval import load_gold  # noqa: E402
from app.llm.factory import get_llm, get_router_llm  # noqa: E402
from app.logging_config import setup_logging  # noqa: E402
from app.rag.factory import get_hybrid_retriever, get_vector_store  # noqa: E402


def _fmt(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "  n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS-style LLM-judged evaluation.")
    parser.add_argument(
        "--generator",
        choices=("gemini", "groq"),
        default="gemini",
        help="Model that generates the answers under test (judge is always Groq). "
        "Use 'groq' when Gemini's daily quota is spent.",
    )
    args = parser.parse_args()

    setup_logging()
    s = get_settings()

    if not s.has_groq:
        print("ERROR: GROQ_API_KEY is not set (needed for the LLM judge).", file=sys.stderr)
        return 1
    if not s.has_gemini:
        print("ERROR: GOOGLE_API_KEY is not set (needed to embed queries).", file=sys.stderr)
        return 1
    if get_vector_store().count() == 0:
        print("ERROR: vector index is empty. Run scripts/build_index.py first.", file=sys.stderr)
        return 1

    judge_llm = get_router_llm()                      # Groq judge
    gen_llm = get_router_llm() if args.generator == "groq" else get_llm()
    answerer = hybrid_answerer(get_hybrid_retriever(), gen_llm)
    evaluator = RagasEvaluator(answerer, LLMJudge(judge_llm))

    # The lookup queries are the answerable ones for an end-to-end RAGAS pass.
    gold = [g for g in load_gold() if g.get("type", "lookup") == "lookup"]
    gen_name = s.groq_model if args.generator == "groq" else s.gemini_model
    print(
        f"Evaluating {len(gold)} questions "
        f"(generator = {gen_name}, judge = Groq {s.groq_model}) …\n"
    )
    report = evaluator.evaluate(gold)

    print("=" * 70)
    print("  RAGAS-STYLE EVAL  (LLM judge: Groq)")
    print("=" * 70)
    for r in report.rows:
        print(f"  • {r.query}")
        print(
            f"      faithfulness {_fmt(r.faithfulness)}   "
            f"answer-rel {_fmt(r.answer_relevance)}   "
            f"context-rel {_fmt(r.context_relevance)}   "
            f"correctness {_fmt(r.answer_correctness)}"
        )
    print("-" * 70)
    print(f"  AVERAGES (n={report.n})")
    print(f"    faithfulness      : {_fmt(report.faithfulness)}")
    print(f"    answer relevance  : {_fmt(report.answer_relevance)}")
    print(f"    context relevance : {_fmt(report.context_relevance)}")
    print(f"    answer correctness: {_fmt(report.answer_correctness)}  (rows with a gold reference)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
