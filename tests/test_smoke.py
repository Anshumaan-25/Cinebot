"""Offline unit tests for Part 0 — no API keys or network required."""

from __future__ import annotations

import numpy as np
import pytest

from app.llm.base import ChatMessage, LLMClient
from app.llm.embedding_cache import EmbeddingCache


class _FakeLLM(LLMClient):
    """Records the messages it receives so we can assert on them."""

    def complete(self, messages, *, temperature=0.2, max_tokens=None):
        return "|".join(f"{m.role}:{m.content}" for m in messages)


def test_generate_builds_system_and_user_messages():
    out = _FakeLLM().generate("hello", system="be brief")
    assert out == "system:be brief|user:hello"


def test_generate_without_system():
    out = _FakeLLM().generate("hello")
    assert out == "user:hello"


def test_chatmessage_is_immutable():
    msg = ChatMessage("user", "hi")
    with pytest.raises(Exception):
        msg.content = "changed"  # frozen dataclass


def test_settings_defaults():
    from app.config import Settings

    s = Settings(_env_file=None)  # ignore any real .env
    assert s.gemini_model == "gemini-2.0-flash"
    assert s.groq_model == "llama-3.3-70b-versatile"
    assert s.embedding_model == "text-embedding-004"
    assert s.embedding_dim == 768
    assert s.has_gemini is False  # no key provided


def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.db")
    key = EmbeddingCache.make_key("text-embedding-004", "RETRIEVAL_QUERY", "hello")

    assert cache.get(key) is None  # miss before write

    vec = np.arange(768, dtype=np.float32)
    cache.put(key, vec)
    got = cache.get(key)

    assert got is not None
    assert got.shape == (768,)
    assert np.allclose(got, vec)


def test_embedding_cache_key_is_deterministic_and_specific():
    k1 = EmbeddingCache.make_key("m", "RETRIEVAL_QUERY", "a")
    k2 = EmbeddingCache.make_key("m", "RETRIEVAL_QUERY", "a")
    k3 = EmbeddingCache.make_key("m", "RETRIEVAL_DOCUMENT", "a")  # different task
    assert k1 == k2
    assert k1 != k3
