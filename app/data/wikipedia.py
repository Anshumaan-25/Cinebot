"""Wikipedia source: resolve the article via Wikidata, then scrape it.

Title resolution follows the requested chain: TMDB ``external_ids`` gives a
``wikidata_id`` -> Wikidata sitelinks give the canonical en-wiki title -> we
fetch and parse that article. A title-search fallback covers the rare film with
no Wikidata sitelink.

Extraction uses a section **whitelist** (keep Plot/Cast/Production/Reception/…)
rather than a blacklist, so we only ingest substantive prose. All Wikimedia
requests carry a descriptive User-Agent and are throttled to <=1 req/sec.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

import httpx

from app.data.cache import DiskCache
from app.data.throttle import RateLimiter
from app.llm.retry import with_backoff

# Keep a section if its heading contains any of these (case-insensitive).
WHITELIST_KEYWORDS = (
    "plot", "synopsis", "premise", "storyline", "story",
    "cast", "character",
    "production", "development", "writing", "screenplay", "casting",
    "filming", "cinematography", "design", "effects",
    "music", "soundtrack", "score",
    "release", "marketing", "box office", "distribution",
    "reception", "critical", "review", "response",
    "accolade", "award", "legacy", "theme", "analysis", "background",
)

# Nodes removed entirely before extraction.
_NOISE_SELECTORS = (
    "sup.reference", "span.mw-editsection", "style", "sup.noprint", ".mw-empty-elt",
)
# A paragraph/list item is dropped if any ancestor carries one of these classes.
_NOISE_ANCESTOR_CLASSES = (
    "infobox", "navbox", "reference", "reflist", "metadata", "hatnote",
    "thumb", "gallery", "sidebar", "ambox", "mbox", "navigation-not-searchable",
)


class WikipediaSource:
    def __init__(
        self,
        api_url: str,
        wikidata_url: str,
        cache_dir,
        user_agent: str,
        min_interval: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        self._api = api_url
        self._wd = wikidata_url
        self._html_cache = DiskCache(cache_dir, "html")
        self._meta_cache = DiskCache(f"{cache_dir}/_meta", "json")
        self._throttle = RateLimiter(min_interval)
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": user_agent})
        self.cache_hits = 0
        self.cache_misses = 0

    def close(self) -> None:
        self._client.close()

    # ----- title resolution: Wikidata first, search fallback -----
    def resolve_title(self, wikidata_id: str | None, fallback_title: str | None = None):
        if wikidata_id:
            title = self._enwiki_title_from_wikidata(wikidata_id)
            if title:
                return title, "wikidata"
        if fallback_title:
            title = self._search_title(fallback_title)
            if title:
                return title, "search"
        return None, "none"

    @with_backoff()
    def _enwiki_title_from_wikidata(self, qid: str) -> str | None:
        key = f"wd:{qid}"
        data = self._meta_cache.get_json(key)
        if data is None:
            self._throttle.wait()
            resp = self._client.get(
                self._wd,
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "sitelinks",
                    "sitefilter": "enwiki",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._meta_cache.put_json(key, data)
        try:
            return data["entities"][qid]["sitelinks"]["enwiki"]["title"]
        except (KeyError, TypeError):
            return None

    @with_backoff()
    def _search_title(self, query: str) -> str | None:
        key = f"search:{query}"
        data = self._meta_cache.get_json(key)
        if data is None:
            self._throttle.wait()
            resp = self._client.get(
                self._api,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": f"{query} film",
                    "srlimit": 1,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._meta_cache.put_json(key, data)
        hits = data.get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None

    # ----- fetch + parse -----
    def fetch_sections(self, title: str) -> dict[str, str]:
        html = self._fetch_html(title)
        if not html:
            return {}
        return self._parse_sections(html)

    @with_backoff()
    def _fetch_html(self, title: str) -> str:
        cached = self._html_cache.get_text(title)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        self._throttle.wait()
        resp = self._client.get(
            self._api,
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        html = data.get("parse", {}).get("text", "")  # missing page -> {"error": ...} -> ""
        self._html_cache.put_text(title, html)
        return html

    def _parse_sections(self, html: str) -> dict[str, str]:
        soup = BeautifulSoup(html, "lxml")
        root = soup.select_one("div.mw-parser-output") or soup
        for selector in _NOISE_SELECTORS:
            for node in root.select(selector):
                node.decompose()

        sections: dict[str, list[str]] = {"Lead": []}
        current = "Lead"
        for el in root.find_all(["h2", "p", "li"]):
            if el.name == "h2":
                current = el.get_text(" ", strip=True) or current
                sections.setdefault(current, [])
                continue
            if self._is_noise(el):
                continue
            text = el.get_text(" ", strip=True)
            if text:
                sections.setdefault(current, []).append(text)

        out: dict[str, str] = {}
        for name, parts in sections.items():
            if name == "Lead" or self._is_whitelisted(name):
                joined = " ".join(parts).strip()
                if joined:
                    out[name] = joined
        return out

    @staticmethod
    def _is_whitelisted(heading: str) -> bool:
        low = heading.lower()
        return any(keyword in low for keyword in WHITELIST_KEYWORDS)

    @staticmethod
    def _is_noise(el) -> bool:
        for ancestor in el.parents:
            name = getattr(ancestor, "name", None)
            if name == "table":
                return True
            classes = ancestor.get("class") if hasattr(ancestor, "get") else None
            if classes and any(
                noise in cls for cls in classes for noise in _NOISE_ANCESTOR_CLASSES
            ):
                return True
        return False
