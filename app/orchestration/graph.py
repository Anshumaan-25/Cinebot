"""The LangGraph chat orchestration graph (Part 6).

    START -> recall -> router --(conditional)--> {retrieve | tools | generate}
             retrieve -> generate
             tools    -> generate
             generate -> persist -> END

Dependencies are injected so the graph is testable with fakes and renderable
(structure-only) without live services. Build via the factory for production.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.orchestration.nodes import (
    make_generate_node,
    make_persist_node,
    make_recall_node,
    make_retrieve_node,
    make_router_node,
    make_tools_node,
    route_decision,
)
from app.orchestration.state import ChatState


def build_chat_graph(*, memory, router_llm, hybrid, gen_llm, fallback_llm=None, tools=None):
    g = StateGraph(ChatState)

    g.add_node("recall", make_recall_node(memory))
    g.add_node("router", make_router_node(router_llm))
    g.add_node("retrieve", make_retrieve_node(hybrid))
    g.add_node("tools", make_tools_node(tools))
    g.add_node("generate", make_generate_node(gen_llm, fallback_llm))
    g.add_node("persist", make_persist_node(memory))

    g.add_edge(START, "recall")
    g.add_edge("recall", "router")
    g.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "tools": "tools", "generate": "generate"},
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("tools", "generate")
    g.add_edge("generate", "persist")
    g.add_edge("persist", END)

    return g.compile()
