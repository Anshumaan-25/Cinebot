"""OMDB live-search tool (Part 7).

Searches OMDB by title, then fetches full details for the best hit — so the
chatbot can answer about films outside our bounded corpus and report live
metadata (IMDb rating, genre, director, cast, plot). Reuses :class:`OMDBClient`
(disk-cached, throttled, key never persisted) against its own cache dir.
"""

from __future__ import annotations

from app.data.omdb_client import OMDBClient
from app.tools.base import Tool, ToolResult


class OMDBSearchTool(Tool):
    name = "omdb"
    description = (
        "Live structured movie facts (year, genre, director, cast, IMDb rating, "
        "plot) for a specific film, by title."
    )

    def __init__(self, client: OMDBClient, summary_chars: int = 1200) -> None:
        self._client = client
        self._summary_chars = summary_chars

    def run(self, query: str) -> ToolResult:
        details = self._resolve(query)
        if not OMDBClient.found(details):
            return ToolResult(self.name, query, ok=False, content="")
        return ToolResult(self.name, query, ok=True, content=self._format(details))

    # ----- internals -----
    def _resolve(self, query: str) -> dict:
        """Search first (handles loose phrasing), fall back to exact-title lookup."""
        results = self._client.search(query)
        if OMDBClient.found(results):
            hits = results.get("Search") or []
            if hits and hits[0].get("imdbID"):
                return self._client.by_imdb_id(hits[0]["imdbID"])
        return self._client.by_title(query)

    def _format(self, d: dict) -> str:
        title = d.get("Title", "?")
        year = d.get("Year", "")
        header = f"{title} ({year})".strip()
        lines = [f"{header} — {d.get('Genre', 'n/a')}"]
        if d.get("Director") and d["Director"] != "N/A":
            lines.append(f"Director: {d['Director']}")
        if d.get("Actors") and d["Actors"] != "N/A":
            lines.append(f"Cast: {d['Actors']}")
        if d.get("imdbRating") and d["imdbRating"] != "N/A":
            lines.append(f"IMDb rating: {d['imdbRating']}/10")
        if d.get("Released") and d["Released"] != "N/A":
            lines.append(f"Released: {d['Released']}")
        if d.get("Plot") and d["Plot"] != "N/A":
            lines.append("Plot: " + self._truncate(d["Plot"], self._summary_chars))
        return "\n".join(lines)
