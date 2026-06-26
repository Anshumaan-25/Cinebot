"""A minimal rate limiter to respect free-tier / etiquette request limits."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Block until at least ``min_interval`` seconds have passed since the last
    call. Thread-safe so it also works under FastAPI worker threads later."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if self._last and elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last = time.monotonic()
