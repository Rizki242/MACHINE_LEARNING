"""Skema keluaran registri model (`backend/app/models/registry.py`)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ModelEntryOut(BaseModel):
    name: str
    horizon: str
    version: str
    data_source: str
    artifact_path: str
    feature_count: int
    train_rows: int
    trained_at: str
    threshold: float
    metrics: dict[str, Any] = {}
    parameters: dict[str, Any] = {}
    notes: list[str] = []


class ModelListResponse(BaseModel):
    total: int
    items: list[ModelEntryOut]
    metadata: dict[str, Any] = {}
    disclaimer: str


class ModelBestResponse(BaseModel):
    entry: Optional[ModelEntryOut]
    disclaimer: str
