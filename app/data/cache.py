"""Tiny on-disk cache so every TMDB / Wikipedia fetch happens at most once.

Keys are hashed to safe filenames. Because all fetches are cached, re-running
the pipeline is free (no repeated network calls, no re-spent quota).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _safe_name(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path | str, ext: str = "json") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ext = ext.lstrip(".")

    def path(self, key: str) -> Path:
        return self.root / f"{_safe_name(key)}.{self.ext}"

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def get_json(self, key: str) -> Any | None:
        p = self.path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text("utf-8"))

    def put_json(self, key: str, obj: Any) -> None:
        self.path(key).write_text(json.dumps(obj, ensure_ascii=False), "utf-8")

    def get_text(self, key: str) -> str | None:
        p = self.path(key)
        return p.read_text("utf-8") if p.exists() else None

    def put_text(self, key: str, text: str) -> None:
        self.path(key).write_text(text, "utf-8")
