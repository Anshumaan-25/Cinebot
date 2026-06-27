"""Wikipedia live-search tool (Part 7).

Full-text searches Wikipedia for the best-matching article, then returns its
lead-section summary — useful for films, people, or topics outside our corpus,
and for fresher context than the build-time snapshot. Requests carry a
descriptive User-Agent, are throttled (Wikimedia etiquette) and disk-cached.
"""

from __future__ import annotations

import httpx

from app.data.cache import DiskCache
from app.data.throttle import RateLimiter
from app.llm.retry import with_backoff
from app.tools.base import Tool, ToolResult


class WikipediaSearchTool(Tool):
    name = "wikipedia"
    description = "Live encyclopedic summary of a film, person, or topic."

    def __init__(
        self,
        api_url: str,
        cache_dir,
        user_agent: str,
        summary_chars: int = 1200,
        min_interval: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        self._api = api_url
        self._cache = DiskCache(cache_dir, "json")
        self._throttle = RateLimiter(min_interval)
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": user_agent})
        self._summary_chars = summary_chars

    def close(self) -> None:
        self._client.close()

    def run(self, query: str) -> ToolResult:
        title = self._search_title(query)
        if not title:
            return ToolResult(self.name, query, ok=False, content="")
        summary = self._extract(title)
        if not summary:
            return ToolResult(self.name, query, ok=False, content="")
        content = f"{title}: {self._truncate(summary, self._summary_chars)}"
        return ToolResult(self.name, query, ok=True, content=content)

    # ----- internals -----
    def _search_title(self, query: str) -> str | None:
        data = self._get(
            {"action": "query", "list": "search", "srsearch": query, "srlimit": 1},
            f"wiki_search:{query}",
        )
        hits = data.get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None

    def _extract(self, title: str) -> str:
        data = self._get(
            {
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
                "formatversion": 2,
            },
            f"wiki_extract:{title}",
        )
        pages = data.get("query", {}).get("pages", [])
        return (pages[0].get("extract", "") if pages else "").strip()

    @with_backoff()
    def _get(self, params: dict, cache_key: str) -> dict:
        cached = self._cache.get_json(cache_key)
        if cached is not None:
            return cached
        self._throttle.wait()
        resp = self._client.get(self._api, params={**params, "format": "json"})
        resp.raise_for_status()
        data = resp.json()
        self._cache.put_json(cache_key, data)
        return data
