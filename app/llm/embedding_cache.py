"""On-disk embedding cache (SQLite).

Embeddings are the most quota-expensive thing we compute against the free tier,
so every vector is cached keyed by (model, task_type, text). Re-runs read from
disk instead of re-spending quota. Stdlib + numpy only (no SDK import), so this
is cheap to unit-test in isolation.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._lock = threading.Lock()
        # check_same_thread=False: FastAPI may touch this from worker threads.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  key TEXT PRIMARY KEY,"
            "  dim INTEGER NOT NULL,"
            "  vector BLOB NOT NULL"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def make_key(model: str, task_type: str, text: str) -> str:
        raw = f"{model}\x00{task_type}\x00{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT dim, vector FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32).reshape(dim)

    def put(self, key: str, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32).ravel()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?, ?, ?)",
                (key, int(vec.shape[0]), vec.tobytes()),
            )
            self._conn.commit()

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        rows = [
            (key, int(np.asarray(v, dtype=np.float32).ravel().shape[0]),
             np.asarray(v, dtype=np.float32).ravel().tobytes())
            for key, v in items
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
