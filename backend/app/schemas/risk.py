"""Skema keluaran Hybrid Risk Engine (README §13, §14)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class RiskPoint(BaseModel):
    """Satu titik waktu, dipakai untuk menggambar grafik tren."""

    timestamp: datetime
    risk_score: float
    risk_score_persistent: float
    confidence_score: float
    data_quality_score: float


class RiskComponents(BaseModel):
    rule_engine: float
    ml_model: float
    anomaly: float


class RiskSnapshot(BaseModel):
    """Satu penilaian lengkap siap tampil untuk operator (README §14)."""

    timestamp: datetime
    risk_score: float
    status: str
    status_meaning: str
    confidence_score: float
    data_quality_score: float
    suspected_area: Optional[str] = None
    dominant_indicators: list[str] = []
    components: RiskComponents


class RiskAssessmentResponse(BaseModel):
    model: str
    horizon: str
    threshold: float
    window_start: datetime
    window_end: datetime
    series: list[RiskPoint]
    snapshots: list[RiskSnapshot]
    data_source: str
    warning: str
    disclaimer: str


class RiskDemoResponse(RiskAssessmentResponse):
    event_time: datetime
