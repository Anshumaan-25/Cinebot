"""RAGAS-style LLM judge (Part 9).

Reference-free RAG quality, scored by an LLM (we use Groq — separate free quota,
and Gemini's daily quota is spent). Four metrics, each one constrained-JSON call:

- **faithfulness**      — fraction of the answer's atomic claims that the
                          retrieved context supports (catches hallucination).
- **answer relevance**  — how directly the answer addresses the question.
- **context relevance** — how much of the retrieved context is useful for the
                          question (a context-precision proxy).
- **answer correctness**— (optional) agreement with a gold reference answer.

Every metric returns a float in ``[0, 1]`` or ``None`` if the judge's reply
can't be parsed, so one flaky call never crashes a run — it's just excluded
from the average.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (tolerates code fences)."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _clamp(x) -> float | None:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return None


_FAITHFULNESS_PROMPT = (
    "You are a strict evaluator. Break the ANSWER into atomic factual claims, then "
    "decide for each whether it is supported by the CONTEXT. Judge ONLY support by "
    "the context, not whether it sounds true.\n"
    'Return JSON: {{"claims": [{{"claim": "...", "supported": true|false}}]}}.\n\n'
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}"
)

_RELEVANCE_PROMPT = (
    "Rate from 0.0 to 1.0 how directly the ANSWER addresses the QUESTION "
    "(ignore whether it is factually correct). 1.0 = fully on-topic and complete; "
    "0.0 = unrelated or evasive.\n"
    'Return JSON: {{"score": <0..1>, "reason": "..."}}.\n\n'
    "QUESTION:\n{question}\n\nANSWER:\n{answer}"
)

_CONTEXT_PROMPT = (
    "Rate from 0.0 to 1.0 the fraction of the CONTEXT that is relevant and useful "
    "for answering the QUESTION. 1.0 = every part is on-point; 0.0 = nothing useful.\n"
    'Return JSON: {{"score": <0..1>, "reason": "..."}}.\n\n'
    "QUESTION:\n{question}\n\nCONTEXT:\n{context}"
)

_CORRECTNESS_PROMPT = (
    "Compare the ANSWER to the REFERENCE answer. Rate from 0.0 to 1.0 how factually "
    "consistent and complete the answer is relative to the reference "
    "(wording may differ).\n"
    'Return JSON: {{"score": <0..1>, "reason": "..."}}.\n\n'
    "QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}"
)


class LLMJudge:
    def __init__(self, llm) -> None:
        self._llm = llm

    def _ask(self, prompt: str) -> dict | None:
        try:
            return extract_json(self._llm.generate(prompt, temperature=0.0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge call failed: %s", exc)
            return None

    # ----- metrics -----
    def faithfulness(self, answer: str, context: str) -> float | None:
        data = self._ask(_FAITHFULNESS_PROMPT.format(context=context, answer=answer))
        if not data:
            return None
        claims = data.get("claims")
        if not isinstance(claims, list) or not claims:
            return 1.0  # nothing asserted -> nothing to hallucinate
        supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported"))
        return supported / len(claims)

    def answer_relevance(self, question: str, answer: str) -> float | None:
        data = self._ask(_RELEVANCE_PROMPT.format(question=question, answer=answer))
        return _clamp(data.get("score")) if data else None

    def context_relevance(self, question: str, context: str) -> float | None:
        data = self._ask(_CONTEXT_PROMPT.format(question=question, context=context))
        return _clamp(data.get("score")) if data else None

    def answer_correctness(self, question: str, answer: str, reference: str) -> float | None:
        data = self._ask(
            _CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer)
        )
        return _clamp(data.get("score")) if data else None
