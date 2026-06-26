"""Shared retry/backoff for free-tier rate limits and transient API errors.

We can't fall back to a paid tier, so we retry rate-limit (429 /
RESOURCE_EXHAUSTED) and transient 5xx errors with exponential backoff. The
predicate matches on the error text rather than provider-specific exception
classes, so the same policy works for Gemini, Groq and the embedding API.
"""

from __future__ import annotations

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_RETRYABLE_MARKERS = (
    "429",
    "rate",
    "resource_exhausted",
    "quota",
    "503",
    "unavailable",
    "500",
    "internal",
    "timeout",
    "deadline",
    "overloaded",
)


def is_retryable(exc: BaseException) -> bool:
    """True for rate-limit / transient errors worth retrying."""
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def with_backoff(max_attempts: int = 5):
    """Decorator: retry the wrapped call on retryable errors with backoff."""
    return retry(
        retry=retry_if_exception(is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
