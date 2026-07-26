"""Titik masuk FastAPI untuk FurnaceGuard AI.

API ini hanya menyediakan endpoint advisory/read-only. Fase ini tidak memiliki
jalur tulis ke DCS/historian dan tidak boleh dipakai sebagai sistem kontrol.
"""

from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.routers.events import router as events_router
from backend.app.api.routers.health import router as health_router
from backend.app.api.routers.models import router as models_router
from backend.app.api.routers.risk import router as risk_router
from backend.app.core.constants import DISCLAIMER


def create_app() -> FastAPI:
    """Membuat instance aplikasi FastAPI."""
    app = FastAPI(
        title="FurnaceGuard AI",
        summary="ML-based early warning system for furnace blocking risk.",
        description=(
            "Decision-support API untuk analisis risiko blocking furnace PLTU "
            "Jeranjang Unit 1. Sistem bersifat read-only/advisory dan bukan "
            "pengganti proteksi boiler, interlock, SOP, atau keputusan operator."
        ),
        version="0.3.0",
        contact={"name": "PLTU Jeranjang Engineering"},
        license_info={"name": "MIT"},
        openapi_tags=[
            {"name": "health", "description": "Pemeriksaan kesiapan layanan."},
            {"name": "events", "description": "Event registry read-only."},
            {"name": "models", "description": "Model registry read-only."},
            {"name": "risk", "description": "Hybrid risk assessment read-only."},
        ],
    )
    app.state.disclaimer = DISCLAIMER
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(events_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(risk_router, prefix="/api/v1")
    return app


app = create_app()
