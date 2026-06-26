"""Offline unit tests for Part 2 (RAG) — no API keys/network.

The vector-store test uses real Chroma with hand-made vectors; everything else
uses fakes. Live Gemini embedding/answers are exercised via build_index / /chat.
"""

from __future__ import annotations

import numpy as np

from app.data.schema import Chunk
from app.evaluation.retrieval_eval import evaluate
from app.llm.base import ChatMessage, LLMClient
from app.rag.rag import RAGPipeline
from app.rag.retriever import RetrievedChunk, Retriever


# ---------------- vector store (real Chroma, deterministic vectors) ----------------
def test_vector_store_similarity_and_metadata_filters(tmp_path):
    from app.rag.vector_store import VectorStore

    store = VectorStore(tmp_path / "chroma", collection="test_films")
    store.recreate()
    chunks = [
        Chunk(chunk_id="a", imdb_id="tt1", film_title="A", section="Plot", source="wikipedia", text="aliens on pandora"),
        Chunk(chunk_id="b", imdb_id="tt2", film_title="B", section="Reception", source="wikipedia", text="critics loved it"),
        Chunk(chunk_id="c", imdb_id="tt1", film_title="A", section="Cast", source="omdb_plot", text="starring actors"),
    ]
    embs = [
        np.array([1.0, 0.0, 0.0], np.float32),
        np.array([0.0, 1.0, 0.0], np.float32),
        np.array([0.9, 0.1, 0.0], np.float32),
    ]
    store.add(chunks, embs)
    assert store.count() == 3

    top = store.query(np.array([1.0, 0.0, 0.0], np.float32), k=3)
    assert top[0]["chunk_id"] == "a"  # nearest by cosine

    by_film = store.query(np.array([1.0, 0.0, 0.0], np.float32), k=3, where={"imdb_id": "tt2"})
    assert [r["chunk_id"] for r in by_film] == ["b"]

    by_section = store.query(np.array([1.0, 0.0, 0.0], np.float32), k=3, where={"section": "Cast"})
    assert [r["chunk_id"] for r in by_section] == ["c"]

    by_source = store.query(np.array([1.0, 0.0, 0.0], np.float32), k=3, where={"source": "omdb_plot"})
    assert [r["chunk_id"] for r in by_source] == ["c"]


# ---------------- retriever maps rows -> dataclass ----------------
class _FakeStore:
    def query(self, embedding, k=5, where=None):
        return [{
            "chunk_id": "x", "imdb_id": "tt9", "film_title": "X", "section": "Plot",
            "source": "wikipedia", "text": "hi", "distance": 0.1,
        }]


class _FakeEmbedder:
    def embed_query(self, text):
        return np.zeros(3, np.float32)


def test_retriever_maps_to_dataclass():
    out = Retriever(_FakeStore(), _FakeEmbedder()).retrieve("q", k=1)
    assert len(out) == 1 and isinstance(out[0], RetrievedChunk)
    assert out[0].imdb_id == "tt9" and out[0].section == "Plot"


# ---------------- RAG prompt + streaming + citations ----------------
class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, k=5, where=None):
        return self._chunks[:k]


class _EchoLLM(LLMClient):
    def complete(self, messages, *, temperature=0.2, max_tokens=None):
        return messages[-1].content  # echo the user message (context + question)


def _rc(imdb, title, section):
    return RetrievedChunk(f"{imdb}-x", imdb, title, section, "wikipedia", f"{title} {section} text", 0.1)


def test_rag_answer_builds_numbered_cited_context():
    chunks = [_rc("tt1", "Avatar", "Plot"), _rc("tt1", "Avatar", "Reception")]
    resp = RAGPipeline(_FakeRetriever(chunks), _EchoLLM(), k=5).answer("about Avatar?")
    assert len(resp.sources) == 2
    assert "[1] (Avatar — Plot)" in resp.answer
    assert "[2] (Avatar — Reception)" in resp.answer
    assert "Question: about Avatar?" in resp.answer


def test_rag_stream_default_yields_full_answer():
    chunks = [_rc("tt1", "Avatar", "Plot")]
    out = "".join(RAGPipeline(_FakeRetriever(chunks), _EchoLLM()).stream_answer("q", chunks))
    assert "[1] (Avatar — Plot)" in out


def test_rag_no_sources_message():
    resp = RAGPipeline(_FakeRetriever([]), _EchoLLM()).answer("q")
    assert resp.sources == [] and "couldn't find" in resp.answer.lower()


def test_llm_stream_default_yields_complete_once():
    class _LLM(LLMClient):
        def complete(self, messages, *, temperature=0.2, max_tokens=None):
            return "hello"

    assert list(_LLM().generate_stream("hi")) == ["hello"]


# ---------------- eval metrics ----------------
class _RankRetriever:
    def __init__(self, mapping):
        self._m = mapping

    def retrieve(self, query, k=5):
        return [
            RetrievedChunk(f"{imdb}-{i}", imdb, imdb, "Plot", "wikipedia", "t", 0.1)
            for i, imdb in enumerate(self._m[query][:k])
        ]


def test_eval_hit_and_mrr():
    gold = [
        {"query": "q1", "imdb_id": "tt1"},  # rank 1
        {"query": "q2", "imdb_id": "tt2"},  # rank 2
        {"query": "q3", "imdb_id": "tt9"},  # miss
    ]
    retr = _RankRetriever({
        "q1": ["tt1", "tt5"],
        "q2": ["tt5", "tt2"],
        "q3": ["tt5", "tt6"],
    })
    res = evaluate(retr, gold, k=5)
    assert res.n == 3
    assert abs(res.hit_at_1 - 1 / 3) < 1e-9
    assert abs(res.hit_at_k - 2 / 3) < 1e-9
    assert abs(res.mrr - (1.0 + 0.5 + 0.0) / 3) < 1e-9
