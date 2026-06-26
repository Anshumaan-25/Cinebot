"""Wikipedia source: the film list, the IMDb id, and the article text.

Flow for the OMDB-based pipeline:
  1. Scrape the 'List of highest-grossing films' page -> film article titles.
  2. For each film, get its IMDb id from Wikidata (property P345).
  3. Fetch and parse the film's Wikipedia article (section whitelist).

All Wikimedia requests carry a descriptive User-Agent, are throttled to
<=1 req/sec, and are cached to disk. Article extraction keeps only whitelisted
substantive sections (Plot/Cast/Production/Reception/…).
"""

from __future__ import annotations

from urllib.parse import unquote

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

# Wikipedia namespaces we never treat as a film link.
_NAMESPACES = (
    "File", "Image", "Category", "Help", "Wikipedia", "Template",
    "Portal", "Special", "Talk", "User", "Module", "Draft",
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


def _title_from_href(href: str) -> str:
    slug = href.split("/wiki/", 1)[1] if "/wiki/" in href else href
    slug = slug.split("#", 1)[0]
    return unquote(slug).replace("_", " ")


def _is_namespace_link(href: str) -> bool:
    if "/wiki/" not in href:
        return True
    tail = href.split("/wiki/", 1)[1]
    prefix = tail.split(":", 1)[0] if ":" in tail else ""
    return prefix in _NAMESPACES


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

    # ----- 1) the film list -----
    def scrape_film_list(self, page_title: str, limit: int) -> list[dict]:
        """Return up to ``limit`` films [{title, wiki_title}] from a Wikipedia
        list page (e.g. 'List of highest-grossing films')."""
        html = self._fetch_html(page_title)
        if not html:
            return []
        return self._parse_film_list(html, limit)

    def _parse_film_list(self, html: str, limit: int) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        for sup in soup.select("sup.reference"):
            sup.decompose()
        tables = soup.select("table.wikitable")
        # The page has several tables (timeline, by-year, ...). Target the ranked
        # "Highest-grossing films" table specifically: it's the one with a Rank
        # column alongside the gross column.
        target = None
        for table in tables:
            headers = " ".join(th.get_text(" ", strip=True).lower() for th in table.find_all("th"))
            if "rank" in headers and "gross" in headers:
                target = table
                break
        if target is None and tables:  # fallback: richest in film links
            target = max(tables, key=lambda t: len(self._film_rows(t)))
        return self._film_rows(target, limit) if target is not None else []

    @staticmethod
    def _film_rows(table, limit: int | None = None) -> list[dict]:
        listings: list[dict] = []
        seen: set[str] = set()
        for row in table.find_all("tr"):
            if not row.find("td"):  # skip pure header rows
                continue
            # Prefer the italicised title link (handles colon titles like
            # "Avengers: Endgame"); fall back to the first article link.
            anchor = row.select_one("td i a[href^='/wiki/'], th i a[href^='/wiki/']")
            if anchor is None:
                for cand in row.select("td a[href^='/wiki/']"):
                    href = cand.get("href", "")
                    if "#" not in href and not _is_namespace_link(href):
                        anchor = cand
                        break
            if anchor is None:
                continue
            wiki_title = anchor.get("title") or _title_from_href(anchor.get("href", ""))
            if not wiki_title or wiki_title in seen:
                continue
            seen.add(wiki_title)
            listings.append(
                {"title": anchor.get_text(strip=True) or wiki_title, "wiki_title": wiki_title}
            )
            if limit and len(listings) >= limit:
                break
        return listings

    # ----- 2) IMDb id via Wikidata (property P345) -----
    @with_backoff()
    def imdb_id_from_title(self, wiki_title: str) -> str | None:
        key = f"imdb_of:{wiki_title}"
        data = self._meta_cache.get_json(key)
        if data is None:
            self._throttle.wait()
            resp = self._client.get(
                self._wd,
                params={
                    "action": "wbgetentities",
                    "sites": "enwiki",
                    "titles": wiki_title,
                    "props": "claims",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._meta_cache.put_json(key, data)
        for qid, entity in data.get("entities", {}).items():
            if qid == "-1":
                continue
            claims = entity.get("claims", {}).get("P345")
            if claims:
                try:
                    return claims[0]["mainsnak"]["datavalue"]["value"]
                except (KeyError, IndexError, TypeError):
                    return None
        return None

    # ----- 3) article text -----
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
