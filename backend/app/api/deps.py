"""Dependensi FastAPI: pemuat data dan artefak, di-cache untuk umur proses.

Setiap fungsi di sini dipakai lewat ``Depends(...)`` dan sengaja dibuat
sebagai fungsi biasa (bukan lambda atau closure) supaya pengujian dapat
menimpanya lewat ``app.dependency_overrides`` tanpa perlu menyentuh berkas
data maupun model yang sesungguhnya.

Registri event dan model dimuat sekali lalu dipakai ulang selama proses
API berjalan — mengulang pipeline harus diikuti mulai ulang proses ini
untuk melihat data terbaru. Wajar untuk fase tanpa basis data atau
ingestion langsung.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import HTTPException

from backend.app.core.config import Settings, get_settings


def get_settings_dep() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def get_event_registry() -> pd.DataFrame:
    from backend.app.data.event_etl import load_registry

    return load_registry(get_settings())


@lru_cache(maxsize=1)
def get_model_registry():
    from backend.app.models.registry import ModelRegistry

    return ModelRegistry.load(get_settings())


@lru_cache(maxsize=1)
def _anomaly_detector() -> Any | None:
    import joblib

    path = get_settings().paths.models / "isolation_forest.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def valid_model_names(settings: Settings) -> list[str]:
    """Nama model klasifikasi yang boleh dipilih lewat API.

    ``isolation_forest`` dikecualikan — itu detektor anomali, bukan
    pengklasifikasi per horizon.
    """
    models = settings.model["models"]
    return [
        name
        for name, config in models.items()
        if name != "isolation_forest" and config.get("enabled", True)
    ]


def valid_horizon_names(settings: Settings) -> list[str]:
    return [str(h["name"]) for h in settings.model["horizons"]]


def validate_model_and_horizon(settings: Settings, model_name: str, horizon: str) -> None:
    models = valid_model_names(settings)
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"model tidak dikenal {model_name!r}. Pilihan: {models}.",
        )
    horizons = valid_horizon_names(settings)
    if horizon not in horizons:
        raise HTTPException(
            status_code=400,
            detail=f"horizon tidak dikenal {horizon!r}. Pilihan: {horizons}.",
        )


@lru_cache(maxsize=16)
def _cached_period_assessment(
    model_name: str, horizon: str, start_iso: str, end_iso: str
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Muat artefak, bangun fitur, dan hitung skor untuk satu jendela waktu.

    Di-cache per kombinasi persis (model, horizon, start, end) — permintaan
    berulang terhadap jendela yang sama (mis. dashboard yang me-refresh
    tampilan yang sedang dilihat) tidak mengulang pembangunan fitur satu
    tahun penuh dari awal.
    """
    from backend.app.reports.risk_demo import load_artifacts, prepare_period, score_period

    settings = get_settings()
    try:
        model, columns, threshold, baseline = load_artifacts(settings, model_name, horizon)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)

    manifest_path = settings.paths.synthetic / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Data sintetis belum dibangkitkan. Jalankan: "
                "python -m backend.app.data.synthetic"
            ),
        )
    import json

    manifest_years = {entry["year"] for entry in json.loads(manifest_path.read_text())["years"]}
    requested_years = {start.year, end.year}
    missing_years = requested_years - manifest_years
    if missing_years:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Tahun {sorted(missing_years)} tidak ada pada data sintetis "
                f"({sorted(manifest_years)} tersedia). Jalankan: "
                "python -m backend.app.data.synthetic --years <rentang>"
            ),
        )

    try:
        features, raw = prepare_period(settings, baseline, start, end)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    scores = score_period(
        features, raw, model, columns, settings, threshold, _anomaly_detector()
    )
    return scores, features, threshold
