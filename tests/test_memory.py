"""Offline unit tests for Part 5 (long-term memory) — no LLM/network needed."""

from __future__ import annotations

from app.memory.extractor import PreferenceExtractor
from app.memory.manager import MemoryManager
from app.memory.store import MemoryStore


# ---------------- store ----------------
def test_store_roundtrip_and_dedup(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    s.add_interaction("u1", "user", "hi")
    s.add_interaction("u1", "assistant", "hello")
    s.add_preference("u1", "likes sci-fi")
    s.add_preference("u1", "likes sci-fi")  # duplicate -> ignored
    s.add_preference("u1", "dislikes horror")

    assert s.preferences("u1") == ["likes sci-fi", "dislikes horror"]
    assert s.recent_interactions("u1") == [("user", "hi"), ("assistant", "hello")]

    recall = s.recall("u1")
    assert "likes sci-fi" in recall and "Recent conversation" in recall

    # per-user isolation
    assert s.preferences("u2") == []

    s.clear_user("u1")
    assert s.preferences("u1") == [] and s.recent_interactions("u1") == []


def test_recent_interactions_limit_and_chronological(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    for i in range(10):
        s.add_interaction("u", "user", f"m{i}")
    hist = s.recent_interactions("u", limit=3)
    assert [c for _, c in hist] == ["m7", "m8", "m9"]  # last 3, oldest-first


def test_recall_empty_user_is_blank(tmp_path):
    assert MemoryStore(tmp_path / "m.db").recall("nobody") == ""


# ---------------- manager (write policy) ----------------
class _FakeExtractor:
    def extract(self, message):
        return ["likes action"] if "action" in message.lower() else []


def test_manager_observe_writes_history_and_preferences(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    MemoryManager(s, _FakeExtractor()).observe("u", "I love action movies", "Here are some.")

    assert "likes action" in s.preferences("u")
    assert s.recent_interactions("u") == [
        ("user", "I love action movies"),
        ("assistant", "Here are some."),
    ]


def test_manager_without_extractor_still_records(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    MemoryManager(s, extractor=None).observe("u", "hello", "hi")
    assert s.preferences("u") == []
    assert len(s.recent_interactions("u")) == 2


# ---------------- extractor parsing ----------------
def test_preference_parse_variants():
    p = PreferenceExtractor._parse
    assert p('["likes sci-fi", "dislikes horror"]') == ["likes sci-fi", "dislikes horror"]
    assert p('Sure: ["likes drama"] done') == ["likes drama"]
    assert p("no json here") == []
    assert p("[]") == []
    assert p('[1, "ok", "  "]') == ["ok"]  # non-strings/blank dropped


class _FakeLLM:
    def generate(self, prompt, temperature=0.0):
        return '["likes thrillers"]'


def test_extractor_end_to_end_with_fake_llm():
    assert PreferenceExtractor(_FakeLLM()).extract("I really enjoy thrillers") == ["likes thrillers"]
