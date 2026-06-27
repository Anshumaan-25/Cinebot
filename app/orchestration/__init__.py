"""Part 6 — LangGraph orchestration.

A compiled graph with conditional routing over nodes: recall (memory) -> router
(Groq classification) -> {retrieve (hybrid GraphRAG) | tools | generate} ->
generate (Gemini, Groq fallback) -> persist (memory write). ``/chat`` runs this
graph so memory personalizes every answer. Visualize: ``python scripts/show_graph.py``.
"""

from app.orchestration.factory import get_chat_graph
from app.orchestration.graph import build_chat_graph
from app.orchestration.state import ChatState

__all__ = ["build_chat_graph", "get_chat_graph", "ChatState"]
