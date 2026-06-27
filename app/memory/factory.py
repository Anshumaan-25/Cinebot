"""Cached factory helpers for the long-term memory stack."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.factory import get_router_llm
from app.memory.extractor import PreferenceExtractor
from app.memory.manager import MemoryManager
from app.memory.store import MemoryStore


@lru_cache
def get_memory_store() -> MemoryStore:
    return MemoryStore(get_settings().sqlite_path)


@lru_cache
def get_memory_manager() -> MemoryManager:
    s = get_settings()
    # Preference extraction uses the Groq router model (separate free quota).
    extractor = PreferenceExtractor(get_router_llm()) if s.has_groq else None
    return MemoryManager(get_memory_store(), extractor)
