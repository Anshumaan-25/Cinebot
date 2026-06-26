"""Request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message.")
    user_id: str = Field("demo-user", description="Stable id for long-term memory (Part 5).")
    session_id: str | None = Field(None, description="Conversation/session id (Part 8).")


class ChatResponse(BaseModel):
    reply: str
    provider: str = Field(..., description="Which backend produced the reply.")
    note: str | None = Field(None, description="Scaffold / status note.")
