"""Offline tests for Part 6 (LangGraph orchestration) — no LLM/Neo4j/Chroma.

The graph is built with fakes and run via invoke(), exercising routing, the
node wiring, memory recall/persist, and the generate quota-fallback.
"""

from __future__ import annotations

from app.orchestration.graph import build_chat_graph
from app.orchestration.nodes import _build_messages, classify
from app.rag.hybrid import HybridContext
from app.rag.retriever import RetrievedChunk


# ---------------- fakes ----------------
class _FakeMemory:
    def __init__(self):
        self.observed = []

    def recall(self, user_id):
        return "Known preferences: likes sci-fi"

    def observe(self, user_id, query, answer):
        self.observed.append((user_id, query, answer))


class _FakeRouterLLM:
    def __init__(self, label="KNOWLEDGE"):
        self.label = label

    def generate(self, prompt, temperature=0.0):
        return self.label


class _FakeHybrid:
    def retrieve(self, query):
        return HybridContext(
            graph_facts=["James Cameron directed Avatar"],
            chunks=[RetrievedChunk("c", "tt1", "Avatar", "Plot", "wikipedia", "text", 0.1)],
        )


class _FakeGenLLM:
    def stream(self, messages):
        yield "ANSWER"


# ---------------- routing ----------------
def test_classify_maps_labels_with_default():
    assert classify(_FakeRouterLLM("KNOWLEDGE"), "q") == "retrieve"
    assert classify(_FakeRouterLLM("REALTIME"), "q") == "tools"
    assert classify(_FakeRouterLLM("CHITCHAT"), "q") == "generate"
    assert classify(_FakeRouterLLM("???"), "q") == "retrieve"  # default


def test_build_messages_includes_all_context():
    state = {
        "query": "q",
        "memory_context": "PREFS",
        "graph_facts": ["F1"],
        "passages": [{"film_title": "A", "section": "Plot", "text": "T"}],
        "tool_results": "",
    }
    user = _build_messages(state)[-1].content
    assert "PREFS" in user and "F1" in user and "[1] (A — Plot)" in user and "Question: q" in user


# ---------------- full graph (invoke) ----------------
def test_graph_knowledge_route_end_to_end():
    mem = _FakeMemory()
    g = build_chat_graph(
        memory=mem, router_llm=_FakeRouterLLM("KNOWLEDGE"), hybrid=_FakeHybrid(), gen_llm=_FakeGenLLM()
    )
    out = g.invoke({"user_id": "u", "query": "who directed Avatar"})

    assert out["route"] == "retrieve"
    assert out["answer"] == "ANSWER"
    assert out["graph_facts"] == ["James Cameron directed Avatar"]
    assert out["passages"][0]["film_title"] == "Avatar"
    assert mem.observed == [("u", "who directed Avatar", "ANSWER")]  # memory persisted


def test_graph_chitchat_skips_retrieval():
    mem = _FakeMemory()
    g = build_chat_graph(
        memory=mem, router_llm=_FakeRouterLLM("CHITCHAT"), hybrid=_FakeHybrid(), gen_llm=_FakeGenLLM()
    )
    out = g.invoke({"user_id": "u", "query": "hi there"})

    assert out["route"] == "generate"
    assert not out.get("graph_facts")  # retrieve node was skipped
    assert out["answer"] == "ANSWER"


# ---------------- generate quota fallback ----------------
class _QuotaLLM:
    def stream(self, messages):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


class _FallbackLLM:
    def stream(self, messages):
        yield "FALLBACK-ANSWER"


def test_generate_falls_back_on_quota():
    g = build_chat_graph(
        memory=_FakeMemory(),
        router_llm=_FakeRouterLLM("CHITCHAT"),
        hybrid=_FakeHybrid(),
        gen_llm=_QuotaLLM(),
        fallback_llm=_FallbackLLM(),
    )
    out = g.invoke({"user_id": "u", "query": "hi"})
    assert "FALLBACK-ANSWER" in out["answer"]


# ---------------- structure is inspectable ----------------
def test_graph_structure_has_all_nodes():
    g = build_chat_graph(memory=None, router_llm=None, hybrid=None, gen_llm=None)
    nodes = set(g.get_graph().nodes)
    for n in ("recall", "router", "retrieve", "tools", "generate", "persist"):
        assert n in nodes
