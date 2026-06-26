"""Groq chat client — Llama 3.3 70B for fast routing decisions (free tier)."""

from __future__ import annotations

from groq import Groq

from app.llm.base import ChatMessage, LLMClient
from app.llm.retry import with_backoff


class GroqLLM(LLMClient):
    """:class:`LLMClient` backed by Groq's OpenAI-style chat API."""

    def __init__(self, api_key: str | None, model: str = "llama-3.3-70b-versatile") -> None:
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env "
                "(get a free key at https://console.groq.com/keys)."
            )
        self._client = Groq(api_key=api_key)
        self._model = model

    @with_backoff()
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
