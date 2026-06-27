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
    omdb_api_key: Optional[str] = None      # structured film data (omdbapi.com)

    # --- Neo4j Aura (free tier) — Part 3 ---
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = "neo4j"
    neo4j_password: Optional[str] = None

    # --- Models ---
    gemini_model: str = "gemini-2.0-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "gemini-embedding-001"  # text-embedding-004 retired from AI Studio
    embedding_dim: int = 768

    # --- Data pipeline (Part 1) ---
    omdb_base_url: str = "https://www.omdbapi.com/"
    omdb_min_interval: float = 0.2          # seconds between OMDB calls
    omdb_cast_limit: int = 10               # OMDB returns ~4 main actors anyway
    highest_grossing_page: str = "List of highest-grossing films"  # Wikipedia film list
    n_films: int = 50                       # default corpus size (scale via --n)
    chunk_target_chars: int = 1800          # ~500-700 tokens
    chunk_overlap_chars: int = 200
    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"
    wikidata_api_url: str = "https://www.wikidata.org/w/api.php"
    wikipedia_min_interval: float = 1.0     # 1 req/sec (Wikimedia etiquette)
    wikipedia_user_agent: str = (
        "MemoryAugmentedChatbot/0.1 (educational internship project; "
        "contact: anshumaan.singh0099@gmail.com)"
    )

    # --- RAG / vector store (Part 2) ---
    chroma_collection: str = "film_chunks"
    rag_top_k: int = 5

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

    @property
    def omdb_cache_dir(self) -> Path:
        return self.raw_dir / "omdb"

    @property
    def wikipedia_cache_dir(self) -> Path:
        return self.raw_dir / "wikipedia"

    @property
    def films_path(self) -> Path:
        return self.processed_dir / "films.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def ensure_dirs(self) -> None:
        """Create the data directory tree if it does not yet exist."""
        for p in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.omdb_cache_dir,
            self.wikipedia_cache_dir,
            self.chroma_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    # Convenience flags (never expose the key values themselves)
    @property
    def has_gemini(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_omdb(self) -> bool:
        return bool(self.omdb_api_key)

    @property
    def has_neo4j(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_password)


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide settings instance."""
    return Settings()
