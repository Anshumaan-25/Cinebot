"""Structured-output schema for LLM cast extraction (Part 3)."""

from __future__ import annotations

from pydantic import BaseModel


class CastMember(BaseModel):
    actor: str
    character: str | None = None
