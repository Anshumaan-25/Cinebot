"""Gemini 2.0 Flash chat client (Google AI Studio free tier)."""

from __future__ import annotations

from collections.abc import Iterator

from google import genai
from google.genai import types

from app.llm.base import ChatMessage, LLMClient
from app.llm.retry import with_backoff


class GeminiLLM(LLMClient):
    """:class:`LLMClient` backed by Gemini via the ``google-genai`` SDK."""

    def __init__(self, api_key: str | None, model: str = "gemini-2.0-flash") -> None:
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env "
                "(get a free key at https://aistudio.google.com/app/apikey)."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _prepare(self, messages: list[ChatMessage], temperature: float, max_tokens: int | None):
        # Gemini takes system text separately and uses "model" (not "assistant").
        system_text = "\n".join(m.content for m in messages if m.role == "system")
        contents: list[types.Content] = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m.content)]))
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_text or None,
        )
        return contents, config

    @with_backoff()
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        contents, config = self._prepare(messages, temperature, max_tokens)
        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        try:
            return response.text or ""
        except (ValueError, AttributeError):
            # No candidates / blocked response — return empty rather than raising.
            return ""

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        contents, config = self._prepare(messages, temperature, max_tokens)
        for chunk in self._client.models.generate_content_stream(
            model=self._model, contents=contents, config=config
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text
