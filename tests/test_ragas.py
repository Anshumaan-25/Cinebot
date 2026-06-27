"""Offline tests for Part 9 (RAGAS-style eval) — no LLM/network required.

The judge runs against a scripted LLM (canned JSON replies), and the evaluator
runs against a fake answerer, covering metric math, the claim-grounded
faithfulness score, lenient JSON parsing, and aggregation that ignores
unparseable / missing scores.
"""

from __future__ import annotations

from app.evaluation.llm_judge import LLMJudge, extract_json
from app.evaluation.ragas_eval import RagasEvaluator, render_context
from app.rag.retriever import RetrievedChunk


class _ScriptedLLM:
    """Returns a reply chosen by matching keywords in the prompt."""

    def __init__(self, rules: dict[str, str], default: str = "{}"):
        self._rules = rules
        self._default = default

    def generate(self, prompt, temperature=0.0):
        for needle, reply in self._rules.items():
            if needle in prompt:
                return reply
        return self._default


# ---------------- JSON helper ----------------
def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"score": 0.5}\n```') == {"score": 0.5}
    assert extract_json("here you go: {\"score\": 1} thanks") == {"score": 1}
    assert extract_json("no json") is None


# ---------------- judge metrics ----------------
def test_faithfulness_is_fraction_of_supported_claims():
    llm = _ScriptedLLM(
        {
            "atomic factual claims": '{"claims": ['
            '{"claim": "a", "supported": true},'
            '{"claim": "b", "supported": false},'
            '{"claim": "c", "supported": true}]}'
        }
    )
    assert LLMJudge(llm).faithfulness("answer", "context") == 2 / 3


def test_faithfulness_no_claims_is_one():
    llm = _ScriptedLLM({"atomic factual claims": '{"claims": []}'})
    assert LLMJudge(llm).faithfulness("I don't have that information.", "ctx") == 1.0


def test_score_metrics_clamp_and_parse():
    judge = LLMJudge(
        _ScriptedLLM(
            {
                "how directly the ANSWER": '{"score": 0.9, "reason": "ok"}',
                "fraction of the CONTEXT": '{"score": 1.4}',   # over 1 -> clamped
                "REFERENCE": '{"score": -0.2}',                 # under 0 -> clamped
            }
        )
    )
    assert judge.answer_relevance("q", "a") == 0.9
    assert judge.context_relevance("q", "ctx") == 1.0
    assert judge.answer_correctness("q", "a", "ref") == 0.0


def test_metric_returns_none_on_unparseable():
    assert LLMJudge(_ScriptedLLM({}, default="garbage")).answer_relevance("q", "a") is None


# ---------------- context rendering ----------------
def test_render_context_includes_facts_and_passages():
    chunks = [RetrievedChunk("c", "tt1", "Avatar", "Plot", "wikipedia", "Pandora text", 0.1)]
    out = render_context(["James Cameron directed Avatar"], chunks)
    assert "James Cameron directed Avatar" in out
    assert "[1] (Avatar — Plot)" in out and "Pandora text" in out


# ---------------- evaluator aggregation ----------------
class _FakeJudge:
    def faithfulness(self, answer, context):
        return 1.0

    def answer_relevance(self, question, answer):
        return 0.8

    def context_relevance(self, question, context):
        return 0.6

    def answer_correctness(self, question, answer, reference):
        return 0.5


def test_evaluator_aggregates_and_correctness_only_when_reference():
    def answerer(query):
        return f"answer to {query}", "some context"

    gold = [
        {"query": "Q1", "reference": "R1"},  # correctness scored
        {"query": "Q2"},                      # no reference -> correctness None
    ]
    report = RagasEvaluator(answerer, _FakeJudge()).evaluate(gold)

    assert report.n == 2
    assert report.faithfulness == 1.0
    assert report.answer_relevance == 0.8
    assert report.context_relevance == 0.6
    assert report.answer_correctness == 0.5  # averaged over the single referenced row
    assert report.rows[1].answer_correctness is None
