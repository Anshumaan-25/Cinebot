"""Centralised configuration loaded from environment / `.env`.

All settings are read once via :func:`get_settings` (cached). API keys are
optional at load time so the app can be imported and tested without them; the
LLM/embedding clients raise a clear error only when actually used without a key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Provider API keys (free tier) ---
    google_api_key: Optional[str] = None  # Gemini 2.0 Flash + text-embedding-004
    groq_api_key: Optional[str] = None     # Llama 3.3 70B (routing)
    tmdb_api_key: Optional[str] = None      # real-time movie data (Part 7)

    # --- Neo4j Aura (free tier) — Part 3 ---
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = "neo4j"
    neo4j_password: Optional[str] = None

    # --- Models ---
    gemini_model: str = "gemini-2.0-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "text-embedding-004"
    embedding_dim: int = 768

    # --- Paths ---
    data_dir: Path = Path("./data")
    sqlite_path: Path = Path("./data/memory.db")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def embedding_cache_path(self) -> Path:
        return self.cache_dir / "embeddings.db"

    def ensure_dirs(self) -> None:
        """Create the data directory tree if it does not yet exist."""
        for p in (self.data_dir, self.raw_dir, self.processed_dir, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)

    # Convenience flags (never expose the key values themselves)
    @property
    def has_gemini(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_neo4j(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_password)

    @property
    def has_tmdb(self) -> bool:
        return bool(self.tmdb_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide settings instance."""
    return Settings()
