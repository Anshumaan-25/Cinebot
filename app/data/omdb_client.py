"""OMDB API client (omdbapi.com) — structured film metadata.

Replaces TMDB as the structured data source (TMDB is geo-blocked in some
regions). Same shape as the previous client: every response is cached to disk,
calls are throttled, and the API key is sent only at request time (never written
into cache keys/filenames).
"""

from __future__ import annotations

import httpx

from app.data.cache import DiskCache
from app.data.throttle import RateLimiter
from app.llm.retry import with_backoff


class OMDBClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        cache_dir,
        min_interval: float = 0.2,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "OMDB_API_KEY is not set. Add it to your .env "
                "(get a free key at http://www.omdbapi.com/apikey.aspx)."
            )
        self._key = api_key
        self._base = base_url
        self._cache = DiskCache(cache_dir, "json")
        self._throttle = RateLimiter(min_interval)
        self._client = httpx.Client(timeout=timeout)
        self.cache_hits = 0
        self.cache_misses = 0

    def close(self) -> None:
        self._client.close()

    # ----- public API -----
    def by_imdb_id(self, imdb_id: str) -> dict:
        return self._get({"i": imdb_id, "plot": "full"})

    def by_title(self, title: str, year: int | None = None) -> dict:
        params = {"t": title, "plot": "full"}
        if year:
            params["y"] = str(year)
        return self._get(params)

    @staticmethod
    def found(response: dict) -> bool:
        return bool(response) and response.get("Response") == "True"

    # ----- internals -----
    def _get(self, params: dict) -> dict:
        cache_key = f"omdb?{sorted(params.items())}"  # api key intentionally excluded
        cached = self._cache.get_json(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        data = self._fetch(params)
        self._cache.put_json(cache_key, data)
        return data

    @with_backoff()
    def _fetch(self, params: dict) -> dict:
        self._throttle.wait()
        query = dict(params)
        query["apikey"] = self._key
        response = self._client.get(self._base, params=query)
        response.raise_for_status()
        return response.json()
