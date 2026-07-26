"""Skema keluaran Event Registry (README §15)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class EventOut(BaseModel):
    event_id: str
    unit_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_type: Optional[str] = None
    event_location: Optional[str] = None
    severity: Optional[int] = None
    initial_symptom: Optional[str] = None
    operator_action: Optional[str] = None
    maintenance_action: Optional[str] = None
    derating_mw: Optional[float] = None
    trip_status: Optional[str] = None
    clinker_found: Optional[str] = None
    coal_source: Optional[str] = None
    coal_blending: Optional[str] = None
    notes: Optional[str] = None
    equipment_raw: Optional[str] = None
    equipment_canonical: Optional[str] = None
    status_raw: Optional[str] = None
    record_kind: Optional[str] = None
    duration_hours: Optional[float] = None
    mwh_lost: Optional[float] = None
    classification_method: Optional[str] = None
    classification_confidence: Optional[float] = None
    needs_review: Optional[bool] = None


class EventListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[EventOut]
    disclaimer: str
