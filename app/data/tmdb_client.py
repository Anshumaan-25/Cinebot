"""TMDB API client (v3) — structured film metadata, external ids, and reviews.

All responses are cached to disk and calls are gently throttled. The API key is
sent only at request time and never written into cache keys/filenames.
"""

from __future__ import annotations

import httpx

from app.data.cache import DiskCache
from app.data.throttle import RateLimiter
from app.llm.retry import with_backoff


class TMDBClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        cache_dir,
        min_interval: float = 0.25,
        cast_limit: int = 10,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "TMDB_API_KEY is not set. Add it to your .env "
                "(get a free key at https://www.themoviedb.org/settings/api)."
            )
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._cache = DiskCache(cache_dir, "json")
        self._throttle = RateLimiter(min_interval)
        self._cast_limit = cast_limit
        self._client = httpx.Client(timeout=timeout)
        self.cache_hits = 0
        self.cache_misses = 0

    def close(self) -> None:
        self._client.close()

    # ----- public API -----
    def top_films(self, n: int, list_name: str = "top_rated") -> list[dict]:
        """Return up to ``n`` films from a TMDB list (deduped, order preserved)."""
        out: list[dict] = []
        seen: set[int] = set()
        page = 1
        while len(out) < n:
            data = self._get(f"/movie/{list_name}", {"page": page, "language": "en-US"})
            results = data.get("results", [])
            if not results:
                break
            for film in results:
                fid = film.get("id")
                if fid is not None and fid not in seen:
                    seen.add(fid)
                    out.append(film)
            if page >= int(data.get("total_pages", page)):
                break
            page += 1
        return out[:n]

    def film_details(self, tmdb_id: int) -> dict:
        """Full details incl. genres + credits (cast/crew) in a single call."""
        return self._get(
            f"/movie/{tmdb_id}",
            {"append_to_response": "credits", "language": "en-US"},
        )

    def external_ids(self, tmdb_id: int) -> dict:
        """imdb_id / wikidata_id / social ids."""
        return self._get(f"/movie/{tmdb_id}/external_ids")

    def film_reviews(self, tmdb_id: int, max_reviews: int = 10) -> list[str]:
        data = self._get(f"/movie/{tmdb_id}/reviews", {"language": "en-US", "page": 1})
        reviews = [r.get("content", "") for r in data.get("results", [])]
        return [r for r in reviews if r][:max_reviews]

    # ----- internals -----
    def _get(self, path: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        cache_key = f"{path}?{sorted(params.items())}"  # api_key intentionally excluded
        cached = self._cache.get_json(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        data = self._fetch(path, params)
        self._cache.put_json(cache_key, data)
        return data

    @with_backoff()
    def _fetch(self, path: str, params: dict) -> dict:
        self._throttle.wait()
        query = dict(params)
        query["api_key"] = self._key
        response = self._client.get(f"{self._base}{path}", params=query)
        response.raise_for_status()
        return response.json()
