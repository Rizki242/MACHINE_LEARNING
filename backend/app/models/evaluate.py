"""Metrik evaluasi model peringatan dini (README §12).

Metrik ambang-bebas biasa saja tidak cukup untuk sistem peringatan dini.
Sebuah model bisa punya ROC-AUC 0,98 dan tetap tidak berguna di ruang
kontrol karena membangkitkan empat puluh alarm palsu setiap hari.

Karena itu modul ini menghitung dua kelompok:

Metrik per sampel — precision, recall, F1, PR-AUC, ROC-AUC, Brier score,
    calibration error. Menjawab: seberapa baik model memisahkan menit
    berisiko dari menit normal.

Metrik per event — false alarms per hari, missed event rate, event
    detection rate, warning horizon rata-rata dan median. Menjawab
    pertanyaan yang sebenarnya ditanyakan operator: berapa banyak event
    nyata yang tertangkap, berapa lama sebelumnya, dan berapa sering
    sistem berteriak tanpa sebab.

PR-AUC dan false alarms per hari adalah metrik utama. Kelas positif
sangat jarang, sehingga ROC-AUC akan terlihat bagus secara menyesatkan.
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
class EvaluationResult:
    """Kumpulan metrik satu model pada satu himpunan data."""

    model_name: str
    horizon: str
    dataset: str
    threshold: float
    sample_metrics: dict[str, float] = field(default_factory=dict)
    event_metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "horizon": self.horizon,
            "dataset": self.dataset,
            "threshold": round(self.threshold, 4),
            **{key: round(value, 4) for key, value in self.sample_metrics.items()},
            **{key: round(value, 4) for key, value in self.event_metrics.items()},
            "notes": self.notes,
        }


# --------------------------------------------------------------------
# Metrik per sampel
# --------------------------------------------------------------------
def calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Expected calibration error.

    Rata-rata selisih mutlak antara probabilitas yang diklaim model dan
    frekuensi kejadian sebenarnya, dibobot jumlah sampel per bin. Nilai
    nol berarti "model bilang 30 persen" memang terjadi 30 persen kali.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = labels.size
    if total == 0:
        return float("nan")

    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(
            float(probabilities[mask].mean()) - float(labels[mask].mean())
        )
    return error


def sample_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, bins: int = 10
) -> dict[str, float]:
    """Hitung seluruh metrik per sampel README §12."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (probabilities >= threshold).astype(int)
    positives = int(labels.sum())

    metrics: dict[str, float] = {
        "positive_rate": float(labels.mean()),
        "positive_count": float(positives),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "calibration_error": float(calibration_error(labels, probabilities, bins)),
    }

    # AUC tidak terdefinisi bila hanya ada satu kelas.
    if 0 < positives < labels.size:
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    return metrics


# --------------------------------------------------------------------
# Metrik per event
# --------------------------------------------------------------------
def event_metrics(
    timestamps: pd.Series,
    probabilities: np.ndarray,
    event_starts: pd.Series,
    threshold: float,
    detection_window_minutes: int,
    persistence_minutes: int = 1,
) -> dict[str, float]:
    """Hitung metrik berbasis event, bukan berbasis menit.

    Sebuah event dianggap terdeteksi bila ada alarm dalam jendela
    ``detection_window_minutes`` sebelum waktu mulainya. Warning horizon
    dihitung dari alarm PERTAMA di dalam jendela itu, karena itulah saat
    operator sebenarnya menerima peringatan.

    Alarm di luar jendela event mana pun dihitung sebagai alarm palsu.
    Alarm berturut-turut dalam satu rentetan dihitung satu kali — operator
    melihat satu alarm yang menyala terus, bukan ratusan alarm terpisah.
    """
    stamps = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    alarm = pd.Series(probabilities >= threshold).reset_index(drop=True)

    if persistence_minutes > 1:
        alarm = (
            alarm.rolling(persistence_minutes, min_periods=persistence_minutes)
            .min()
            .fillna(0)
            .astype(bool)
        )

    marks = pd.Series(event_starts).reset_index(drop=True).astype(bool)
    event_times = stamps[marks.to_numpy()]

    window = pd.Timedelta(minutes=detection_window_minutes)
    alarm_times = stamps[alarm.to_numpy()]

    detected = 0
    horizons: list[float] = []
    consumed = pd.Series(False, index=alarm_times.index)

    for event_time in event_times:
        in_window = alarm_times[
            (alarm_times >= event_time - window) & (alarm_times <= event_time)
        ]
        consumed.loc[in_window.index] = True
        if len(in_window):
            detected += 1
            first = in_window.min()
            horizons.append((event_time - first).total_seconds() / 60.0)

    # Alarm palsu: rentetan alarm yang tidak menyentuh event mana pun.
    #
    # Rentetan panjang TIDAK dihitung satu kali. Alarm yang menyala tiga jam
    # tanpa ada apa-apa bukanlah satu gangguan tunggal bagi operator, dan
    # menghitungnya begitu membuka celah yang merusak: ambang mendekati nol
    # akan membuat alarm menyala hampir sepanjang waktu, menghasilkan satu
    # rentetan raksasa, lalu tercatat sebagai "kurang dari satu alarm palsu
    # per hari" sambil mendeteksi setiap event. Setiap rentetan karena itu
    # dihitung sebanyak jendela deteksi yang dilaluinya.
    unclaimed = alarm_times[~consumed.to_numpy()]
    false_alarm_count = 0
    if len(unclaimed):
        new_run = unclaimed.diff() > pd.Timedelta(minutes=persistence_minutes * 2)
        run_id = new_run.cumsum()
        for _, run in unclaimed.groupby(run_id):
            minutes = (run.max() - run.min()).total_seconds() / 60.0
            false_alarm_count += max(
                1, int(np.ceil(minutes / max(detection_window_minutes, 1)))
            )

    span_days = max((stamps.iloc[-1] - stamps.iloc[0]).total_seconds() / 86400.0, 1e-9)
    event_count = int(len(event_times))
    duty_cycle = float(alarm.mean()) if len(alarm) else 0.0

    return {
        "event_count": float(event_count),
        "events_detected": float(detected),
        "event_detection_rate": float(detected / event_count) if event_count else float("nan"),
        "missed_event_rate": (
            float(1.0 - detected / event_count) if event_count else float("nan")
        ),
        "false_alarms_per_day": float(false_alarm_count / span_days),
        "alarm_duty_cycle": duty_cycle,
        "average_warning_horizon_minutes": (
            float(np.mean(horizons)) if horizons else float("nan")
        ),
        "median_warning_horizon_minutes": (
            float(np.median(horizons)) if horizons else float("nan")
        ),
        "evaluation_span_days": float(span_days),
    }


# --------------------------------------------------------------------
# Pemilihan ambang
# --------------------------------------------------------------------
def choose_threshold(
    timestamps: pd.Series,
    probabilities: np.ndarray,
    event_starts: pd.Series,
    settings: Settings | None = None,
) -> tuple[float, dict[str, Any]]:
    """Pilih ambang keputusan di data validasi.

    Ambang dipilih dari sasaran operasional, bukan dari nilai bulat yang
    kelihatan rapi: cari ambang dengan tingkat deteksi event tertinggi
    yang masih memenuhi anggaran alarm palsu harian. Bila tidak ada satu
    pun ambang yang memenuhinya, ambil yang alarm palsunya paling sedikit
    dan catat bahwa sasaran tidak tercapai — jangan diam-diam
    melonggarkannya.

    Pemilihan bertumpu pada metrik per event, bukan per sampel: yang
    dijanjikan sistem ini kepada operator adalah menangkap event dengan
    alarm palsu terbatas, bukan mengoptimalkan F1 per menit.
    """
    settings = settings or get_settings()
    alarm_config = settings.thresholds["alarm"]
    budget = float(alarm_config["target_false_alarms_per_day"])
    persistence = int(alarm_config["persistence_minutes"])
    detection_window = int(
        settings.model["evaluation"]["detection_window_minutes"]
    )

    max_duty = float(alarm_config.get("max_alarm_duty_cycle", 0.20))

    candidates = np.unique(np.quantile(probabilities, np.linspace(0.50, 0.9995, 40)))
    trials: list[dict[str, Any]] = []

    for threshold in candidates:
        metrics = event_metrics(
            timestamps,
            probabilities,
            event_starts,
            float(threshold),
            detection_window,
            persistence,
        )
        trials.append(
            {
                "threshold": float(threshold),
                "false_alarms_per_day": metrics["false_alarms_per_day"],
                "alarm_duty_cycle": metrics["alarm_duty_cycle"],
                "event_detection_rate": metrics["event_detection_rate"],
                "median_warning_horizon_minutes": metrics[
                    "median_warning_horizon_minutes"
                ],
            }
        )

    # Dua syarat, bukan satu. Anggaran alarm palsu saja masih menyisakan
    # celah untuk ambang yang membuat alarm menyala hampir sepanjang waktu;
    # batas duty cycle menutupnya.
    feasible = [
        t
        for t in trials
        if t["false_alarms_per_day"] <= budget and t["alarm_duty_cycle"] <= max_duty
    ]
    if feasible:
        best = max(feasible, key=lambda t: (t["event_detection_rate"], -t["threshold"]))
        note = (
            f"Ambang dipilih dengan anggaran {budget:.1f} alarm palsu per hari "
            f"dan duty cycle maksimum {max_duty:.0%}; tercapai "
            f"{best['false_alarms_per_day']:.2f} alarm/hari pada duty "
            f"{best['alarm_duty_cycle']:.1%}."
        )
    else:
        within_duty = [t for t in trials if t["alarm_duty_cycle"] <= max_duty]
        pool = within_duty or trials
        best = min(pool, key=lambda t: t["false_alarms_per_day"])
        note = (
            f"SASARAN TIDAK TERCAPAI: tidak ada ambang yang memenuhi "
            f"{budget:.1f} alarm palsu per hari sekaligus duty cycle "
            f"{max_duty:.0%}. Terbaik {best['false_alarms_per_day']:.2f} "
            f"alarm/hari pada duty {best['alarm_duty_cycle']:.1%}."
        )
        LOGGER.warning(note)

    return float(best["threshold"]), {"note": note, "trials": trials}


def evaluate(
    model_name: str,
    horizon: str,
    dataset: str,
    timestamps: pd.Series,
    probabilities: np.ndarray,
    labels: np.ndarray,
    event_starts: pd.Series,
    threshold: float,
    settings: Settings | None = None,
) -> EvaluationResult:
    """Hitung seluruh metrik README §12 untuk satu model dan satu himpunan."""
    settings = settings or get_settings()
    config = settings.model["evaluation"]
    persistence = int(settings.thresholds["alarm"]["persistence_minutes"])

    result = EvaluationResult(
        model_name=model_name,
        horizon=horizon,
        dataset=dataset,
        threshold=threshold,
        sample_metrics=sample_metrics(
            labels, probabilities, threshold, int(config["calibration_bins"])
        ),
        event_metrics=event_metrics(
            timestamps,
            probabilities,
            event_starts,
            threshold,
            int(config["detection_window_minutes"]),
            persistence,
        ),
    )

    if result.sample_metrics["positive_count"] < 30:
        result.notes.append(
            "Jumlah sampel positif sangat sedikit — metrik tidak stabil."
        )
    if result.event_metrics["event_count"] < 5:
        result.notes.append(
            "Jumlah event pada himpunan ini di bawah lima; metrik per event "
            "hanya indikatif."
        )
    return result


def comparison_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Susun tabel perbandingan seluruh model, diurut PR-AUC menurun."""
    frame = pd.DataFrame([result.to_dict() for result in results])
    columns = [
        "model",
        "horizon",
        "dataset",
        "threshold",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "positive_rate",
        "event_count",
        "events_detected",
        "event_detection_rate",
        "missed_event_rate",
        "false_alarms_per_day",
        "alarm_duty_cycle",
        "median_warning_horizon_minutes",
        "average_warning_horizon_minutes",
        "brier_score",
        "calibration_error",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].sort_values("pr_auc", ascending=False).reset_index(drop=True)
