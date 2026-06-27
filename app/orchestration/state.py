"""Shared state for the LangGraph chat orchestration (Part 6)."""

from __future__ import annotations

from typing import TypedDict


class ChatState(TypedDict, total=False):
    user_id: str
    query: str
    route: str               # router decision: "retrieve" | "tools" | "generate"
    memory_context: str      # recalled preferences + history
    graph_facts: list        # knowledge-graph facts (strings)
    passages: list           # retrieved chunks as {film_title, section, text}
    tool_results: str        # live tool output (Part 7)
    answer: str
