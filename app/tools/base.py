"""Tool abstraction for the dynamic-tools layer (Part 7).

A :class:`Tool` is any live lookup the orchestration can invoke when a query
needs data outside the static corpus (e.g. a film not in our 50-title set, or
the current IMDb rating). Each tool takes a short search term and returns a
:class:`ToolResult`; the runner formats results into the ``tool_results`` slot
of the chat state. Tools degrade gracefully — a network/parse failure yields an
``ok=False`` result rather than raising into the graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool: str          # tool name, e.g. "omdb"
    query: str         # the search term actually used
    ok: bool           # did the tool find something usable?
    content: str       # formatted, human-readable result (empty when ok=False)

    def render(self) -> str:
        return f"[{self.tool}] {self.content}".strip() if self.ok else ""


class Tool(ABC):
    """A single live data source exposed to the orchestrator."""

    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, query: str) -> ToolResult:
        """Look ``query`` up live and return a :class:`ToolResult`."""
        raise NotImplementedError

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = (text or "").strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
