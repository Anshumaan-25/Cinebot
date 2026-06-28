"""LangGraph node factories + routing for the chat orchestration (Part 6).

Each ``make_*_node`` closes over its injected dependency and returns a node
callable (state -> partial state), so the graph is testable with fakes. The
generate node streams tokens via LangGraph's stream writer and falls back to a
secondary model if the primary hits its quota (production primary = Gemini).
"""

from __future__ import annotations

import logging

from app.llm.base import ChatMessage

logger = logging.getLogger(__name__)

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover - very old langgraph
    get_stream_writer = None


# ---------------- router ----------------
_ROUTER_PROMPT = (
    "Classify the user's message into exactly one label:\n"
    "KNOWLEDGE - questions about well-known movies/films/actors/directors/genres/plots/reviews "
    "(answerable from a curated movie knowledge base).\n"
    "REALTIME - needs live or out-of-corpus data: current IMDb ratings, what is playing now, "
    "latest box office, or a specific recent/upcoming/obscure film likely missing from a fixed "
    "knowledge base.\n"
    "CHITCHAT - greetings, statements of preference or opinion, or anything not needing retrieval.\n"
    "Respond with ONLY the label.\n\nMessage: {query}"
)
_ROUTE_MAP = {"KNOWLEDGE": "retrieve", "REALTIME": "tools", "CHITCHAT": "generate"}


def classify(router_llm, query: str) -> str:
    try:
        out = (router_llm.generate(_ROUTER_PROMPT.format(query=query), temperature=0.0) or "").upper()
    except Exception as exc:  # noqa: BLE001
        logger.warning("router failed, defaulting to retrieve: %s", exc)
        return "retrieve"
    for label, route in _ROUTE_MAP.items():
        if label in out:
            return route
    return "retrieve"


def route_decision(state) -> str:
    return state.get("route", "retrieve")


# ---------------- generation prompt ----------------
SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable movie assistant. Personalize tone and "
    "suggestions using the USER PREFERENCES & HISTORY.\n"
    "Treat the supplied context as your evidence: the KNOWLEDGE-GRAPH FACTS are "
    "authoritative for relationships (who directed or acted in what, and genres) — "
    "prefer and reason over them for relational or 'both X and Y' questions; the "
    "RETRIEVED PASSAGES give descriptive detail (cite inline as [1], [2] when you "
    "use them); LIVE TOOL RESULTS carry fresh, out-of-corpus data.\n"
    "Guidelines:\n"
    "- Always be genuinely helpful. For recommendations, opinions, or 'similar to' "
    "requests, combine the context with your own general film knowledge to give a "
    "concrete answer with specific titles and a brief reason for each.\n"
    "- Ground strict factual claims (dates, box-office figures, exact cast) in the "
    "context; if such a specific fact isn't supported there, say you're not certain "
    "instead of inventing it.\n"
    "- Never mention your tools, retrieval, a 'knowledge graph', or that you're "
    "waiting for data, and never say things like 'once I have that data'. Just "
    "answer naturally, the way a film expert would."
)


def _build_messages(state) -> list[ChatMessage]:
    memory = state.get("memory_context") or "(none)"
    facts = "\n".join(f"- {f}" for f in state.get("graph_facts", [])) or "(none)"
    passages = state.get("passages", [])
    passages_txt = (
        "\n\n".join(
            f"[{i}] ({p['film_title']} — {p['section']})\n{p['text']}"
            for i, p in enumerate(passages, 1)
        )
        or "(none)"
    )
    tools_txt = state.get("tool_results") or "(none)"
    user = (
        f"USER PREFERENCES & HISTORY:\n{memory}\n\n"
        f"KNOWLEDGE-GRAPH FACTS:\n{facts}\n\n"
        f"RETRIEVED PASSAGES:\n{passages_txt}\n\n"
        f"LIVE TOOL RESULTS:\n{tools_txt}\n\n"
        f"Question: {state['query']}"
    )
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]


def _writer():
    if get_stream_writer is None:
        return None
    try:
        return get_stream_writer()
    except Exception:  # outside a streaming run
        return None


def _is_quota(exc) -> bool:
    s = str(exc).lower()
    return "resource_exhausted" in s or "429" in s or "quota" in s


# ---------------- node factories ----------------
def make_recall_node(memory):
    def recall(state):
        return {"memory_context": memory.recall(state["user_id"])}

    return recall


def make_router_node(router_llm):
    def router(state):
        return {"route": classify(router_llm, state["query"])}

    return router


def make_retrieve_node(hybrid):
    def retrieve(state):
        try:
            ctx = hybrid.retrieve(state["query"])
        except Exception as exc:  # noqa: BLE001 — degrade gracefully (e.g. embed quota)
            logger.warning("retrieve failed: %s", exc)
            return {"graph_facts": [], "passages": []}
        passages = [
            {"film_title": c.film_title, "section": c.section, "text": c.text}
            for c in ctx.chunks
        ]
        return {"graph_facts": ctx.graph_facts, "passages": passages}

    return retrieve


def make_tools_node(tools):
    def tools_node(state):
        if tools is None:
            return {"tool_results": ""}
        try:
            return {"tool_results": tools(state["query"]) or ""}
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool error: %s", exc)
            return {"tool_results": ""}

    return tools_node


def make_generate_node(gen_llm, fallback_llm=None):
    def generate(state):
        messages = _build_messages(state)
        writer = _writer()
        try:
            parts = []
            for tok in gen_llm.stream(messages):
                if writer:
                    writer(tok)
                parts.append(tok)
            return {"answer": "".join(parts)}
        except Exception as exc:  # noqa: BLE001
            if fallback_llm is not None and _is_quota(exc):
                logger.warning("primary generator quota hit; using fallback model")
                note = "[primary model quota reached — answering with fallback model]\n"
                if writer:
                    writer(note)
                parts = [note]
                for tok in fallback_llm.stream(messages):
                    if writer:
                        writer(tok)
                    parts.append(tok)
                return {"answer": "".join(parts)}
            raise

    return generate


def make_persist_node(memory):
    def persist(state):
        memory.observe(state["user_id"], state.get("query", ""), state.get("answer"))
        return {}

    return persist
