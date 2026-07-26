"""Hybrid Risk Engine (README §11, §13, §14) lewat HTTP.

Seluruh perhitungan dipakai ulang dari ``backend/app/reports/risk_demo.py``
dan ``backend/app/rules/risk_rules.py`` — router ini hanya menyusun
permintaan HTTP, memvalidasi parameter, dan membentuk skema respons.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.deps import (
    _cached_period_assessment,
    get_settings_dep,
    validate_model_and_horizon,
)
from backend.app.core.config import Settings
from backend.app.core.constants import (
    DATA_SOURCE_SYNTHETIC,
    DISCLAIMER,
    SYNTHETIC_METRIC_WARNING,
)
from backend.app.rules import risk_rules
from backend.app.schemas.risk import (
    RiskAssessmentResponse,
    RiskComponents,
    RiskDemoResponse,
    RiskPoint,
    RiskSnapshot,
)

router = APIRouter(prefix="/risk", tags=["risk"])

#: Rentang maksimum per permintaan — membatasi ukuran payload JSON. Biaya
#: komputasi didominasi pemuatan satu tahun penuh terlepas dari lebar
#: jendela, jadi batas ini bukan soal performa melainkan ukuran respons.
MAX_SPAN = timedelta(days=7)


def _series_points(scores: pd.DataFrame) -> list[RiskPoint]:
    return [
        RiskPoint(
            timestamp=row.timestamp,
            risk_score=float(row.risk_score),
            risk_score_persistent=float(row.risk_score_persistent),
            confidence_score=float(row.confidence_score),
            data_quality_score=float(row.data_quality_score),
        )
        for row in scores.itertuples()
    ]


def _snapshots(
    scores: pd.DataFrame, features: pd.DataFrame, positions: list[int], settings: Settings
) -> list[RiskSnapshot]:
    result = []
    for position in positions:
        row = scores.iloc[position]
        assessment = risk_rules.assess_row(
            features,
            position,
            risk_score=float(row["risk_score_persistent"]),
            model_probability=float(row["probability"]),
            data_quality=float(row["data_quality_score"]),
            rule_score=float(row["rule_score"]),
            anomaly_score=float(row["anomaly_score"]),
            settings=settings,
        )
        payload = assessment.to_dict()
        result.append(
            RiskSnapshot(
                timestamp=row["timestamp"],
                risk_score=payload["risk_score"],
                status=payload["status"],
                status_meaning=payload["status_meaning"],
                confidence_score=payload["confidence_score"],
                data_quality_score=payload["data_quality_score"],
                suspected_area=payload["suspected_area"],
                dominant_indicators=payload["dominant_indicators"],
                components=RiskComponents(**payload["components"]),
            )
        )
    return result


def _top_positions(
    scores: pd.DataFrame, top: int, before: Optional[pd.Timestamp] = None
) -> list[int]:
    """Posisi ``top`` skor tertinggi, opsional hanya sebelum suatu waktu."""
    candidates = scores
    if before is not None:
        windowed = scores.loc[scores["timestamp"] < before]
        if not windowed.empty:
            candidates = windowed
    ranked = candidates.sort_values("risk_score_persistent", ascending=False).head(top)
    return list(ranked.index)


@router.get("/assessment", response_model=RiskAssessmentResponse)
def risk_assessment(
    start: datetime,
    end: datetime,
    model: str = "xgboost",
    horizon: Optional[str] = None,
    top: int = Query(default=3, ge=1, le=10),
    settings: Settings = Depends(get_settings_dep),
) -> RiskAssessmentResponse:
    horizon = horizon or str(settings.model["primary_horizon"])
    validate_model_and_horizon(settings, model, horizon)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts <= start_ts:
        raise HTTPException(status_code=400, detail="end harus setelah start.")
    if end_ts - start_ts > MAX_SPAN:
        raise HTTPException(
            status_code=400,
            detail=f"Rentang maksimum {MAX_SPAN.days} hari per permintaan.",
        )

    scores, features, threshold = _cached_period_assessment(
        model, horizon, start_ts.isoformat(), end_ts.isoformat()
    )

    return RiskAssessmentResponse(
        model=model,
        horizon=horizon,
        threshold=threshold,
        window_start=start_ts,
        window_end=end_ts,
        series=_series_points(scores),
        snapshots=_snapshots(scores, features, _top_positions(scores, top), settings),
        data_source=DATA_SOURCE_SYNTHETIC,
        warning=SYNTHETIC_METRIC_WARNING,
        disclaimer=DISCLAIMER,
    )


@router.get("/demo", response_model=RiskDemoResponse)
def risk_demo_endpoint(
    model: str = "xgboost",
    horizon: Optional[str] = None,
    top: int = Query(default=3, ge=1, le=10),
    settings: Settings = Depends(get_settings_dep),
) -> RiskDemoResponse:
    """Jendela demo satu event blocking nyata dari tahun uji (README §14)."""
    from backend.app.reports.risk_demo import find_demo_window

    horizon = horizon or str(settings.model["primary_horizon"])
    validate_model_and_horizon(settings, model, horizon)

    horizon_minutes = next(
        int(h["minutes"]) for h in settings.model["horizons"] if h["name"] == horizon
    )
    try:
        start_ts, end_ts, event_time = find_demo_window(settings, horizon_minutes)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    scores, features, threshold = _cached_period_assessment(
        model, horizon, start_ts.isoformat(), end_ts.isoformat()
    )

    positions = _top_positions(scores, top, before=event_time)
    return RiskDemoResponse(
        model=model,
        horizon=horizon,
        threshold=threshold,
        window_start=start_ts,
        window_end=end_ts,
        series=_series_points(scores),
        snapshots=_snapshots(scores, features, positions, settings),
        data_source=DATA_SOURCE_SYNTHETIC,
        warning=SYNTHETIC_METRIC_WARNING,
        disclaimer=DISCLAIMER,
        event_time=event_time,
    )
