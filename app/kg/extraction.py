"""LLM cast extraction (Part 3).

Wikipedia 'Cast' sections list far more actors than OMDB's 3, but as free text.
We extract (actor, character) pairs with Gemini structured output — one call per
film (the film's Cast chunks are concatenated first), throttled and cached so a
re-run never re-spends quota.
"""

from __future__ import annotations

import hashlib
import logging

from google import genai
from google.genai import types

from app.data.cache import DiskCache
from app.data.throttle import RateLimiter
from app.kg.schema import CastMember
from app.llm.retry import with_backoff

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are extracting the cast of a film from the text of its Wikipedia 'Cast' "
    "section. Return every actor and the character they play. Include ONLY real "
    "cast members (actors/voice actors); ignore directors, crew, and any prose that "
    "is not a cast listing. If a character is not stated, leave it null.\n\n"
    "Film: {title}\n\nCast section text:\n{text}"
)


class CastExtractor:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        cache: DiskCache | None = None,
        min_interval: float = 4.0,
        max_chars: int = 6000,
    ) -> None:
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (needed for cast extraction).")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._cache = cache
        self._throttle = RateLimiter(min_interval)
        self._max_chars = max_chars
        self.api_calls = 0
        self.cache_hits = 0

    def extract(self, imdb_id: str, title: str, cast_text: str) -> list[CastMember]:
        text = (cast_text or "").strip()
        if not text:
            return []
        key = self._key(imdb_id, text)
        if self._cache:
            cached = self._cache.get_json(key)
            if cached is not None:
                self.cache_hits += 1
                return [CastMember(**c) for c in cached]
        members = self._extract(title, text)
        if self._cache:
            self._cache.put_json(key, [m.model_dump() for m in members])
        return members

    def _key(self, imdb_id: str, text: str) -> str:
        raw = f"{self._model}\x00{imdb_id}\x00{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @with_backoff(max_attempts=6)
    def _extract(self, title: str, text: str) -> list[CastMember]:
        self._throttle.wait()
        self.api_calls += 1
        response = self._client.models.generate_content(
            model=self._model,
            contents=_PROMPT.format(title=title, text=text[: self._max_chars]),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[CastMember],
                temperature=0.0,
            ),
        )
        parsed = response.parsed or []
        # De-dupe by actor name, keep first character seen.
        seen: dict[str, CastMember] = {}
        for m in parsed:
            if m.actor and m.actor.strip() and m.actor not in seen:
                seen[m.actor] = m
        return list(seen.values())
