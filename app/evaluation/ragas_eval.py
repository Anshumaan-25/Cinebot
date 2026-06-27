"""RAGAS-style end-to-end evaluation (Part 9).

For each gold question we generate an answer over real retrieved context, then
score it with an :class:`~app.evaluation.llm_judge.LLMJudge`. The evaluator is
decoupled from how answers are produced: pass any ``answerer(query) ->
(answer, context)``. :func:`hybrid_answerer` adapts the production hybrid
GraphRAG retriever for this (with any LLM as the generator), so the same context
the chatbot would use is what gets judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.evaluation.llm_judge import LLMJudge
from app.llm.base import ChatMessage

_ANSWER_SYSTEM = (
    "You are a movie assistant. Answer the question using ONLY the provided context "
    "(knowledge-graph facts and passages). Cite passages as [1], [2]. If the context "
    "does not contain the answer, say you don't have that information."
)


def render_context(graph_facts: list[str], passages: list) -> str:
    """One text block of graph facts + numbered passages (what the judge sees)."""
    facts = "\n".join(f"- {f}" for f in graph_facts) or "(no graph facts)"
    chunks = (
        "\n\n".join(
            f"[{i}] ({p.film_title} — {p.section})\n{p.text}"
            for i, p in enumerate(passages, 1)
        )
        or "(no passages)"
    )
    return f"KNOWLEDGE-GRAPH FACTS:\n{facts}\n\nRETRIEVED PASSAGES:\n{chunks}"


def hybrid_answerer(hybrid_retriever, llm):
    """Build an ``answerer(query) -> (answer, context)`` over hybrid GraphRAG."""

    def answer(query: str) -> tuple[str, str]:
        ctx = hybrid_retriever.retrieve(query)
        context = render_context(ctx.graph_facts, ctx.chunks)
        messages = [
            ChatMessage("system", _ANSWER_SYSTEM),
            ChatMessage("user", f"{context}\n\nQuestion: {query}"),
        ]
        return llm.complete(messages), context

    return answer


@dataclass
class RagasRow:
    query: str
    faithfulness: float | None
    answer_relevance: float | None
    context_relevance: float | None
    answer_correctness: float | None
    answer: str


@dataclass
class RagasReport:
    n: int
    faithfulness: float | None
    answer_relevance: float | None
    context_relevance: float | None
    answer_correctness: float | None  # avg over rows that had a reference
    rows: list[RagasRow] = field(default_factory=list)


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


class RagasEvaluator:
    def __init__(self, answerer, judge: LLMJudge) -> None:
        self._answer = answerer
        self._judge = judge

    def evaluate(self, gold: list[dict]) -> RagasReport:
        rows: list[RagasRow] = []
        for item in gold:
            query = item["query"]
            answer, context = self._answer(query)
            reference = item.get("reference")
            rows.append(
                RagasRow(
                    query=query,
                    faithfulness=self._judge.faithfulness(answer, context),
                    answer_relevance=self._judge.answer_relevance(query, answer),
                    context_relevance=self._judge.context_relevance(query, context),
                    answer_correctness=(
                        self._judge.answer_correctness(query, answer, reference)
                        if reference
                        else None
                    ),
                    answer=answer,
                )
            )
        return RagasReport(
            n=len(rows),
            faithfulness=_avg([r.faithfulness for r in rows]),
            answer_relevance=_avg([r.answer_relevance for r in rows]),
            context_relevance=_avg([r.context_relevance for r in rows]),
            answer_correctness=_avg([r.answer_correctness for r in rows]),
            rows=rows,
        )
