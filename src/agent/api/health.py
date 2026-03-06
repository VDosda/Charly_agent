from __future__ import annotations

from fastapi import APIRouter

from agent.api.schemas import HealthResponse


router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
