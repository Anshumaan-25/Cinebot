"""Section-aware, sentence-boundary chunking with overlap.

Chunks stay within ``target_chars`` (~500-700 tokens) and carry a small
``overlap_chars`` tail from the previous chunk so context isn't lost at edges.
"""

from __future__ import annotations

import re

from app.data.schema import Chunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")[:40] or "section"


def split_text(text: str, target_chars: int = 1800, overlap_chars: int = 200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > target_chars:
            chunks.append(" ".join(current))
            # Carry a tail of recent sentences as overlap into the next chunk.
            overlap: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                if overlap_len + len(prev) > overlap_chars:
                    break
                overlap.insert(0, prev)
                overlap_len += len(prev) + 1
            current = list(overlap)
            current_len = overlap_len
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def make_chunks(
    tmdb_id: int,
    film_title: str,
    section: str,
    source: str,
    text: str,
    *,
    target_chars: int = 1800,
    overlap_chars: int = 200,
) -> list[Chunk]:
    pieces = split_text(text, target_chars=target_chars, overlap_chars=overlap_chars)
    return [
        Chunk(
            chunk_id=f"{tmdb_id}-{source}-{_slug(section)}-{i}",
            tmdb_id=tmdb_id,
            film_title=film_title,
            section=section,
            source=source,
            text=piece,
        )
        for i, piece in enumerate(pieces)
    ]
