"""Event Registry (README §15) lewat HTTP — hanya baca."""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.deps import get_event_registry
from backend.app.core.constants import DISCLAIMER
from backend.app.schemas.events import EventListResponse, EventOut

router = APIRouter(prefix="/events", tags=["events"])

_OUTPUT_COLUMNS = list(EventOut.model_fields)


def _row_to_event(row: pd.Series) -> EventOut:
    payload = {}
    for column in _OUTPUT_COLUMNS:
        value = row.get(column)
        payload[column] = None if pd.isna(value) else value
    return EventOut(**payload)


@router.get("", response_model=EventListResponse)
def list_events(
    event_type: Optional[str] = None,
    min_severity: Optional[int] = Query(default=None, ge=0, le=4),
    needs_review: Optional[bool] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    registry: pd.DataFrame = Depends(get_event_registry),
) -> EventListResponse:
    frame = registry

    if event_type is not None:
        frame = frame.loc[frame["event_type"] == event_type]
    if min_severity is not None:
        frame = frame.loc[frame["severity"].fillna(-1) >= min_severity]
    if needs_review is not None:
        frame = frame.loc[frame["needs_review"].fillna(False) == needs_review]
    if start is not None:
        frame = frame.loc[frame["start_time"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["start_time"] <= pd.Timestamp(end)]

    frame = frame.sort_values("start_time")
    total = len(frame)
    page = frame.iloc[offset : offset + limit]

    return EventListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_row_to_event(row) for _, row in page.iterrows()],
        disclaimer=DISCLAIMER,
    )


@router.get("/{event_id}", response_model=EventOut)
def get_event(
    event_id: str, registry: pd.DataFrame = Depends(get_event_registry)
) -> EventOut:
    matches = registry.loc[registry["event_id"] == event_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} tidak ditemukan.")
    return _row_to_event(matches.iloc[0])
