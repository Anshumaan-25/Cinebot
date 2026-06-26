"""Text normalisation applied before chunking."""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[\d+\]")          # leftover [1] [23] citation markers
_EDIT = re.compile(r"\[edit\]", re.IGNORECASE)
_BRACKETED_NOTE = re.compile(r"\[(?:citation needed|clarification needed|note \d+)\]", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip citation/edit artefacts and collapse whitespace."""
    if not text:
        return ""
    text = _CITATION.sub("", text)
    text = _BRACKETED_NOTE.sub("", text)
    text = _EDIT.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()
