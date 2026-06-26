"""Provider-agnostic chat-LLM interface.

Concrete providers (Gemini, Groq, …) implement :meth:`LLMClient.complete`.
Everything else in the app depends only on this abstraction, so swapping a
provider never ripples outward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

Role = str  # "system" | "user" | "assistant"


@dataclass(frozen=True)
class ChatMessage:
    """A single turn in a conversation."""

    role: Role
    content: str


class LLMClient(ABC):
    """Minimal chat-completion contract every provider must satisfy."""

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's text reply for a list of messages."""
        raise NotImplementedError

    # ----- convenience helpers (provider-independent) -----
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Single-prompt convenience wrapper around :meth:`complete`."""
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage("system", system))
        messages.append(ChatMessage("user", prompt))
        return self.complete(messages, temperature=temperature, max_tokens=max_tokens)

    # ----- streaming -----
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield the reply incrementally. Providers without native streaming
        fall back to yielding the full reply once."""
        yield self.complete(messages, temperature=temperature, max_tokens=max_tokens)

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage("system", system))
        messages.append(ChatMessage("user", prompt))
        yield from self.stream(messages, temperature=temperature, max_tokens=max_tokens)
