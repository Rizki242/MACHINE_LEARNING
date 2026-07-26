"""Registri model terlatih (`backend/app/models/registry.py`) lewat HTTP."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.deps import get_model_registry
from backend.app.core.constants import DISCLAIMER
from backend.app.models.registry import ModelRegistry
from backend.app.schemas.models import ModelBestResponse, ModelEntryOut, ModelListResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models(
    horizon: Optional[str] = None,
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelListResponse:
    entries = registry.entries
    if horizon is not None:
        entries = [entry for entry in entries if entry.horizon == horizon]

    return ModelListResponse(
        total=len(entries),
        items=[ModelEntryOut(**entry.to_dict()) for entry in entries],
        metadata=registry.metadata,
        disclaimer=DISCLAIMER,
    )


@router.get("/best", response_model=ModelBestResponse)
def best_model(
    horizon: str,
    metric: str = Query(default="pr_auc"),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelBestResponse:
    entry = registry.best(horizon, metric)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tidak ada model dengan horizon {horizon!r} dan metrik {metric!r}.",
        )
    return ModelBestResponse(entry=ModelEntryOut(**entry.to_dict()), disclaimer=DISCLAIMER)
