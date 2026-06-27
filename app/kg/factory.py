"""Cached factory helpers for the knowledge-graph stack."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.data.cache import DiskCache
from app.kg.extraction import CastExtractor
from app.kg.graph_store import GraphStore


@lru_cache
def get_graph_store() -> GraphStore:
    s = get_settings()
    return GraphStore(s.neo4j_uri, s.neo4j_username, s.neo4j_password, s.neo4j_database)


@lru_cache
def get_cast_extractor() -> CastExtractor:
    s = get_settings()
    cache = DiskCache(s.cast_cache_dir, "json")
    return CastExtractor(
        s.google_api_key,
        s.gemini_model,
        cache=cache,
        min_interval=s.kg_min_interval,
    )
