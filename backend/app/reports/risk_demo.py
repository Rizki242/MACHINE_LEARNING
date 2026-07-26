"""Demonstrasi Hybrid Risk Engine dari ujung ke ujung.

Mengambil model terlatih, menjalankannya pada satu periode data, lalu
menyusun keluaran persis seperti contoh README §14: Risk Score, status,
Confidence Score, Data Quality Score, suspected area, indikator dominan,
dan rekomendasi pemeriksaan.

Inilah bentuk yang akan dilihat operator. Semua modul lain bermuara ke
sini.

PERINGATAN
----------
Keluaran demo ini dihitung dari DATA SINTETIS. Angkanya bukan kondisi
PLTU Jeranjang Unit 1 dan tidak boleh dijadikan dasar tindakan operasi.

Penggunaan
----------
    python -m backend.app.reports.risk_demo
    python -m backend.app.reports.risk_demo --model xgboost --top 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.constants import SYNTHETIC_METRIC_WARNING
from backend.app.core.logging import get_logger
from backend.app.explainability import shap_explainer
from backend.app.rules import risk_rules

LOGGER = get_logger(__name__)


def load_artifacts(
    settings: Settings, model_name: str, horizon: str
) -> tuple[Any, list[str], float, Any]:
    """Muat model, daftar kolom, ambang, dan baseline dari disk."""
    import joblib

    from backend.app.features.baseline import Baseline

    model_path = settings.paths.models / f"{model_name}_{horizon}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artefak {model_path.name} belum ada. Jalankan: "
            "python -m backend.app.models.train"
        )
    payload = joblib.load(model_path)

    baseline_path = settings.paths.models / "baseline_load_zone.json"
    baseline = Baseline.load(baseline_path)

    return payload["model"], payload["columns"], float(payload["threshold"]), baseline


def prepare_period(
    settings: Settings,
    baseline,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bangun fitur dan data mentah untuk satu periode waktu."""
    from backend.app.data.synthetic import load_synthetic
    from backend.app.features.baseline import assign_load_zone
    from backend.app.features.engineering import build_features

    years = sorted({start.year, end.year})
    raw = load_synthetic(settings, years=years)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    # Riwayat satu jam sebelum periode dibutuhkan agar fitur bergulir
    # terisi penuh di titik pertama yang ditampilkan.
    warmup = pd.Timedelta(minutes=90)
    window = raw.loc[raw["timestamp"].between(start - warmup, end)].reset_index(drop=True)
    if window.empty:
        raise ValueError(f"Tidak ada data sintetis pada {start} sampai {end}.")

    window["load_zone"] = assign_load_zone(window, settings)
    features = build_features(window, settings, baseline=baseline)

    keep = window["timestamp"] >= start
    return features.loc[keep].reset_index(drop=True), window.loc[keep].reset_index(
        drop=True
    )


def score_period(
    features: pd.DataFrame,
    raw: pd.DataFrame,
    model: Any,
    columns: list[str],
    settings: Settings,
    threshold: float,
    detector: Any | None = None,
) -> pd.DataFrame:
    """Hitung ketiga skor README §13 untuk seluruh periode."""
    from backend.app.data import quality
    from backend.app.models.train import _prepare_matrix, anomaly_score

    matrix = _prepare_matrix(features, columns)
    probability = pd.Series(model.predict_proba(matrix)[:, 1], index=features.index)
    model_risk = risk_rules.probability_to_risk(probability, threshold)

    _, rule_score = risk_rules.evaluate_rules(features, settings)
    anomaly = pd.Series(
        anomaly_score(detector, features, columns), index=features.index
    )

    risk = risk_rules.combine_scores(rule_score, model_risk, anomaly, settings)
    persistent = risk_rules.apply_persistence(risk, settings)

    data_quality = quality.rolling_quality(raw, settings, window_minutes=60)
    confidence = risk_rules.confidence_from_quality(probability, data_quality, settings)

    return pd.DataFrame(
        {
            "timestamp": raw["timestamp"],
            "unit_load_mw": raw["unit_load_mw"],
            "main_steam_flow": raw["main_steam_flow"],
            "probability": probability,
            "model_risk_score": model_risk,
            "rule_score": rule_score,
            "anomaly_score": anomaly,
            "risk_score": risk,
            "risk_score_persistent": persistent,
            "confidence_score": confidence,
            "data_quality_score": data_quality,
            "event_start": raw["event_start"],
        }
    )


def render_snapshot(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    position: int,
    horizon_minutes: int,
    settings: Settings,
) -> str:
    """Susun satu tampilan operator untuk satu titik waktu."""
    row = scores.iloc[position]
    triggers = risk_rules.rule_triggers_at(features, position, settings)
    area = risk_rules.suspected_area(triggers, settings)
    status, _ = risk_rules.status_for(float(row["risk_score_persistent"]), settings)

    indicators = [trigger.describe() for trigger in triggers[:5]]
    if not indicators:
        indicators = ["Tidak ada aturan teknik yang menyala pada titik ini"]

    return shap_explainer.format_operator_output(
        timestamp=row["timestamp"],
        load_mw=float(row["unit_load_mw"]),
        steam_flow=float(row["main_steam_flow"]),
        risk_score=float(row["risk_score_persistent"]),
        status=status,
        confidence=float(row["confidence_score"]),
        data_quality=float(row["data_quality_score"]),
        suspected_area=area,
        indicators=indicators,
        horizon_minutes=(horizon_minutes // 2, horizon_minutes),
        recommendations=shap_explainer.recommendations_for(area),
    )


def find_demo_window(
    settings: Settings, horizon_minutes: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Cari satu event blocking pada tahun uji untuk didemokan."""
    from backend.app.data.event_etl import load_registry

    registry = load_registry(settings)
    blocking = settings.event_taxonomy["blocking_event_types"]
    test_start = pd.Timestamp(settings.model["split"]["validation_end"])

    candidates = registry.loc[
        registry["event_type"].isin(blocking) & (registry["start_time"] > test_start)
    ].sort_values("start_time")

    if candidates.empty:
        raise ValueError("Tidak ada event blocking pada periode uji.")

    event_time = pd.Timestamp(candidates.iloc[0]["start_time"])
    start = event_time - pd.Timedelta(minutes=horizon_minutes * 4)
    end = event_time + pd.Timedelta(minutes=30)
    return start, end, event_time


def run(
    settings: Settings | None = None,
    model_name: str = "xgboost",
    horizon: str | None = None,
    top: int = 3,
) -> dict[str, Any]:
    """Jalankan demo dan tulis hasilnya ke berkas laporan."""
    import joblib

    settings = settings or get_settings()
    settings.paths.ensure()

    horizon = horizon or str(settings.model["primary_horizon"])
    horizon_minutes = next(
        int(h["minutes"]) for h in settings.model["horizons"] if h["name"] == horizon
    )

    model, columns, threshold, baseline = load_artifacts(settings, model_name, horizon)

    detector = None
    detector_path = settings.paths.models / "isolation_forest.joblib"
    if detector_path.exists():
        detector = joblib.load(detector_path)

    start, end, event_time = find_demo_window(settings, horizon_minutes)
    LOGGER.info(
        "Periode demo %s sampai %s (event pada %s)", start, end, event_time
    )

    features, raw = prepare_period(settings, baseline, start, end)
    scores = score_period(
        features, raw, model, columns, settings, threshold, detector
    )

    # Titik yang ditampilkan: skor tertinggi sebelum event terjadi.
    before_event = scores["timestamp"] < event_time
    ranked = (
        scores.loc[before_event]
        .sort_values("risk_score_persistent", ascending=False)
        .head(top)
    )

    snapshots = [
        render_snapshot(scores, features, int(position), horizon_minutes, settings)
        for position in ranked.index
    ]

    output_path = settings.paths.reports / "risk_engine_demo.txt"
    header = [
        "=" * 72,
        "DEMO HYBRID RISK ENGINE — FURNACEGUARD AI",
        "=" * 72,
        SYNTHETIC_METRIC_WARNING,
        "",
        f"Model            : {model_name}",
        f"Horizon          : {horizon} ({horizon_minutes} menit)",
        f"Ambang keputusan : {threshold:.5f}",
        f"Periode          : {start:%Y-%m-%d %H:%M} sampai {end:%Y-%m-%d %H:%M}",
        f"Event nyata      : {event_time:%Y-%m-%d %H:%M}",
        "",
    ]
    body = ("\n\n" + "=" * 72 + "\n\n").join(snapshots)
    output_path.write_text("\n".join(header) + body + "\n", encoding="utf-8")

    summary_path = settings.paths.reports / "risk_engine_demo.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "horizon": horizon,
                "threshold": threshold,
                "window_start": str(start),
                "window_end": str(end),
                "event_time": str(event_time),
                "peak_risk_score": float(scores["risk_score_persistent"].max()),
                "risk_at_event_minus_60m": float(
                    scores.loc[
                        scores["timestamp"]
                        <= event_time - pd.Timedelta(minutes=60),
                        "risk_score_persistent",
                    ].tail(1).squeeze()
                    if (scores["timestamp"] <= event_time - pd.Timedelta(minutes=60)).any()
                    else np.nan
                ),
                "median_data_quality": float(scores["data_quality_score"].median()),
                "warning": SYNTHETIC_METRIC_WARNING,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "scores": scores,
        "snapshots": snapshots,
        "output_path": output_path,
        "summary_path": summary_path,
        "event_time": event_time,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo Hybrid Risk Engine Unit 1")
    parser.add_argument("--model", default="xgboost", help="Nama model terlatih")
    parser.add_argument("--horizon", default=None, help="Horizon prediksi")
    parser.add_argument("--top", type=int, default=2, help="Jumlah tampilan")
    args = parser.parse_args(argv)

    outcome = run(get_settings(), args.model, args.horizon, args.top)

    print()
    for snapshot in outcome["snapshots"]:
        print(snapshot)
        print()
        print("=" * 72)
        print()
    print(f"Tersimpan: {outcome['output_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
