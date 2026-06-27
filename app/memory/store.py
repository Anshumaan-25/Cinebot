"""Long-term per-user memory store (SQLite).

Two tables: durable ``preferences`` (deduped per user) and ``interactions`` (the
running chat history). ``recall`` formats both into a context block ready to
inject into a prompt. Thread-safe so FastAPI worker threads can share it.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS ix_interactions_user ON interactions(user_id)",
    """CREATE TABLE IF NOT EXISTS preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at REAL NOT NULL,
        UNIQUE(user_id, content)
    )""",
)


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()

    def add_interaction(self, user_id: str, role: str, content: str) -> None:
        if not content:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO interactions (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (user_id, role, content, time.time()),
            )
            self._conn.commit()

    def add_preference(self, user_id: str, content: str) -> None:
        content = (content or "").strip()
        if not content:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO preferences (user_id, content, created_at) VALUES (?, ?, ?)",
                (user_id, content, time.time()),
            )
            self._conn.commit()

    def recent_interactions(self, user_id: str, limit: int = 6) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [(r["role"], r["content"]) for r in reversed(rows)]

    def preferences(self, user_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT content FROM preferences WHERE user_id = ? ORDER BY id", (user_id,)
            ).fetchall()
        return [r["content"] for r in rows]

    def recall(self, user_id: str, max_interactions: int = 6) -> str:
        """Format this user's preferences + recent history for prompt injection."""
        prefs = self.preferences(user_id)
        history = self.recent_interactions(user_id, max_interactions)
        parts: list[str] = []
        if prefs:
            parts.append("Known preferences for this user:\n" + "\n".join(f"- {p}" for p in prefs))
        if history:
            convo = "\n".join(f"{role.capitalize()}: {content}" for role, content in history)
            parts.append("Recent conversation:\n" + convo)
        return "\n\n".join(parts)

    def clear_user(self, user_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM interactions WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM preferences WHERE user_id = ?", (user_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
