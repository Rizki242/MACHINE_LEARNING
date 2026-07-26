"""Penilaian kualitas data masukan.

README §13 mewajibkan tiga skor dilaporkan terpisah: Risk Score,
Confidence Score, dan Data Quality Score. Modul ini menghitung yang
ketiga, dan hanya yang ketiga.

Alasan skor ini berdiri sendiri: prediksi risiko yang dihitung dari data
setengah hilang atau dari sensor yang macet tetap menghasilkan angka yang
terlihat meyakinkan. Operator berhak tahu bahwa angka itu berdiri di atas
data yang buruk sebelum menindaklanjutinya.

Empat komponen (bobot dari ``config/thresholds.yaml``):
  kelengkapan          — berapa banyak nilai yang benar-benar ada
  sinyal macet         — berapa lama pembacaan membeku
  pelanggaran rentang  — nilai di luar batas fisik yang masuk akal
  keteraturan cap waktu— apakah jarak sampel sesuai yang diharapkan
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class QualityResult:
    """Hasil penilaian kualitas satu dataset atau satu jendela waktu."""

    score: float
    completeness: float
    stuck_score: float
    range_score: float
    timestamp_score: float
    row_count: int
    column_count: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_quality_score": round(self.score, 2),
            "completeness": round(self.completeness, 4),
            "stuck_score": round(self.stuck_score, 4),
            "range_score": round(self.range_score, 4),
            "timestamp_score": round(self.timestamp_score, 4),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "details": self.details,
        }

    def render(self) -> str:
        return (
            f"Data Quality Score : {self.score:.1f} / 100\n"
            f"  kelengkapan      : {self.completeness * 100:.1f}%\n"
            f"  sinyal tidak macet: {self.stuck_score * 100:.1f}%\n"
            f"  dalam rentang    : {self.range_score * 100:.1f}%\n"
            f"  cap waktu teratur: {self.timestamp_score * 100:.1f}%\n"
            f"  baris x kolom    : {self.row_count} x {self.column_count}"
        )


def _analog_columns(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in exclude
        and frame[column].dtype.kind == "f"
        and not column.startswith("ramp_")
    ]


def stuck_fraction(series: pd.Series, window: int, tolerance: float) -> float:
    """Bagian sampel yang berada di dalam rentetan nilai beku.

    Sebuah nilai dihitung macet bila ia bagian dari rentetan nilai identik
    yang panjangnya mencapai ``window``. Nilai yang kebetulan sama dua-tiga
    kali berturut-turut tidak dihitung — itu wajar pada sinyal tenang.
    """
    values = series.to_numpy()
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return 0.0

    changed = np.ones(values.size, dtype=bool)
    changed[1:] = ~(np.abs(np.diff(values)) <= tolerance)
    changed[~valid] = True

    # Nomor kelompok naik setiap kali nilai berubah.
    group = np.cumsum(changed)
    lengths = np.bincount(group)
    run_length = lengths[group]

    stuck = (run_length >= window) & valid
    return float(stuck.sum() / max(valid.sum(), 1))


def range_violation_fraction(
    series: pd.Series, bounds: tuple[float, float] | list[float]
) -> float:
    """Bagian sampel di luar rentang fisik yang masuk akal."""
    values = series.to_numpy()
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return 0.0
    low, high = float(bounds[0]), float(bounds[1])
    outside = valid & ((values < low) | (values > high))
    return float(outside.sum() / valid.sum())


def timestamp_regularity(
    index: pd.Series | pd.DatetimeIndex, expected_seconds: float
) -> tuple[float, dict[str, Any]]:
    """Seberapa dekat jarak antar-sampel dengan jarak yang diharapkan."""
    stamps = pd.to_datetime(pd.Series(index)).sort_values()
    if len(stamps) < 3:
        return 1.0, {"insufficient_samples": True}

    deltas = stamps.diff().dropna().dt.total_seconds()
    on_time = float(((deltas - expected_seconds).abs() <= expected_seconds * 0.1).mean())
    gaps = deltas[deltas > expected_seconds * 2]

    return on_time, {
        "median_interval_seconds": float(deltas.median()),
        "expected_interval_seconds": expected_seconds,
        "gap_count": int(len(gaps)),
        "largest_gap_seconds": float(gaps.max()) if len(gaps) else 0.0,
    }


def assess(
    frame: pd.DataFrame,
    settings: Settings | None = None,
    timestamp_column: str = "timestamp",
) -> QualityResult:
    """Hitung Data Quality Score untuk satu dataframe timeseries."""
    settings = settings or get_settings()
    config = settings.thresholds["data_quality"]
    weights = config["weights"]
    window = int(config["stuck_window_samples"])
    tolerance = float(config["stuck_tolerance"])
    expected_seconds = float(config["expected_interval_seconds"])
    ranges = settings.plausible_ranges

    exclude = {
        timestamp_column,
        "synthetic_seed",
        "is_running",
        "outage_active",
        "event_start",
    }
    columns = _analog_columns(frame, exclude)
    if not columns:
        raise ValueError("Tidak ada kolom analog yang bisa dinilai.")

    completeness = float(frame[columns].notna().to_numpy().mean())

    per_column: dict[str, dict[str, float]] = {}
    stuck_values: list[float] = []
    range_values: list[float] = []

    for column in columns:
        series = frame[column]
        stuck = stuck_fraction(series, window, tolerance)
        stuck_values.append(stuck)

        bounds = ranges.get(column)
        violation = range_violation_fraction(series, bounds) if bounds else 0.0
        if bounds:
            range_values.append(violation)

        per_column[column] = {
            "missing": round(float(series.isna().mean()), 5),
            "stuck": round(stuck, 5),
            "range_violation": round(violation, 5),
            "has_range_check": bounds is not None,
        }

    stuck_score = 1.0 - float(np.mean(stuck_values))
    range_score = 1.0 - (float(np.mean(range_values)) if range_values else 0.0)

    if timestamp_column in frame.columns:
        timestamp_score, timestamp_details = timestamp_regularity(
            frame[timestamp_column], expected_seconds
        )
    else:
        timestamp_score, timestamp_details = 1.0, {"timestamp_column_absent": True}

    score = 100.0 * (
        float(weights["completeness"]) * completeness
        + float(weights["stuck_signal"]) * stuck_score
        + float(weights["range_violation"]) * range_score
        + float(weights["timestamp_regularity"]) * timestamp_score
    )

    worst = sorted(
        per_column.items(),
        key=lambda item: item[1]["missing"] + item[1]["stuck"] + item[1]["range_violation"],
        reverse=True,
    )[:10]

    return QualityResult(
        score=float(np.clip(score, 0.0, 100.0)),
        completeness=completeness,
        stuck_score=stuck_score,
        range_score=range_score,
        timestamp_score=timestamp_score,
        row_count=int(len(frame)),
        column_count=len(columns),
        details={
            "timestamp": timestamp_details,
            "columns_without_range_check": [
                name for name, info in per_column.items() if not info["has_range_check"]
            ],
            "worst_columns": dict(worst),
        },
    )


def rolling_quality(
    frame: pd.DataFrame,
    settings: Settings | None = None,
    window_minutes: int = 60,
    timestamp_column: str = "timestamp",
) -> pd.Series:
    """Data Quality Score bergulir per baris.

    Dipakai saat menyajikan skor berdampingan dengan Risk Score: keduanya
    harus merujuk jendela waktu yang sama agar dapat dibandingkan.

    Hanya melihat ke belakang. Skor pada waktu ``t`` tidak boleh dipengaruhi
    data setelah ``t``.
    """
    settings = settings or get_settings()
    config = settings.thresholds["data_quality"]
    weights = config["weights"]

    exclude = {
        timestamp_column,
        "synthetic_seed",
        "is_running",
        "outage_active",
        "event_start",
    }
    columns = _analog_columns(frame, exclude)
    window = f"{window_minutes}min"

    indexed = frame.set_index(pd.to_datetime(frame[timestamp_column]))
    present = indexed[columns].notna().mean(axis=1)
    completeness = present.rolling(window, min_periods=1).mean()

    ranges = settings.plausible_ranges
    checked = [c for c in columns if c in ranges]
    if checked:
        inside = pd.DataFrame(
            {
                column: indexed[column].between(
                    ranges[column][0], ranges[column][1]
                )
                | indexed[column].isna()
                for column in checked
            },
            index=indexed.index,
        ).mean(axis=1)
        range_score = inside.rolling(window, min_periods=1).mean()
    else:
        range_score = pd.Series(1.0, index=indexed.index)

    # Komponen macet dan keteraturan cap waktu dihitung pada tingkat
    # dataset, bukan per baris — keduanya butuh jendela panjang agar
    # bermakna. Di sini keduanya dianggap sempurna dan dikoreksi oleh
    # pemanggil lewat `assess` bila diperlukan.
    weight_sum = float(weights["completeness"]) + float(weights["range_violation"])
    score = 100.0 * (
        float(weights["completeness"]) * completeness
        + float(weights["range_violation"]) * range_score
        + (1.0 - weight_sum)
    )
    score.index = frame.index
    return score.clip(0.0, 100.0).rename("data_quality_score")
