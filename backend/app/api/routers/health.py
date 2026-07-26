"""Pemeriksaan kesiapan API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_settings_dep
from backend.app.core.config import Settings
from backend.app.core.constants import DISCLAIMER

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings_dep)) -> dict:
    return {
        "status": "ok",
        "deployment_mode": settings.deployment_mode,
        "timezone": settings.timezone,
        "disclaimer": DISCLAIMER,
    }
