"""Minimal, consistent logging setup shared across the app and scripts."""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True
