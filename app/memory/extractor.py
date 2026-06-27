"""Write policy: extract durable user preferences from a message via the LLM.

Uses the provider-agnostic LLM interface (the Groq router model in practice, so
this costs no Gemini quota). Output is parsed leniently from a JSON array so it
works without provider-specific structured-output modes.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_PROMPT = (
    "From the user's message below, extract any DURABLE preferences about movies "
    "(liked or disliked genres, directors, actors, films, or viewing tastes). "
    "Return ONLY a JSON array of short preference strings, e.g. "
    '["likes science fiction", "favorite director: Christopher Nolan", "dislikes horror"]. '
    "Capture only stable preferences, not one-off questions or requests. "
    "If there are none, return [].\n\n"
    "Message: {message}"
)
_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class PreferenceExtractor:
    def __init__(self, llm) -> None:
        self._llm = llm

    def extract(self, message: str) -> list[str]:
        if not message or not message.strip():
            return []
        try:
            out = self._llm.generate(_PROMPT.format(message=message[:2000]), temperature=0.0)
        except Exception as exc:  # noqa: BLE001 — extraction must never break the chat
            logger.warning("preference extraction failed: %s", exc)
            return []
        return self._parse(out)

    @staticmethod
    def _parse(text: str) -> list[str]:
        match = _ARRAY.search(text or "")
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [s.strip() for s in data if isinstance(s, str) and s.strip()][:10]
