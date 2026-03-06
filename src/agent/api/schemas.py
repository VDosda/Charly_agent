from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ChatStreamRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = Field(default="backend-user", min_length=1)
    session_id: str = Field(default="backend-session", min_length=1)
