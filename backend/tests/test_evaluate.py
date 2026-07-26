"""Pengujian metrik evaluasi peringatan dini.

Metrik per event adalah yang paling mudah dibuat terlihat bagus secara
keliru — menghitung alarm per menit, atau menghitung warning horizon dari
alarm terakhir alih-alih pertama, keduanya menghasilkan angka indah yang
tidak berarti apa-apa di ruang kontrol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.core.config import get_settings
from backend.app.models import evaluate


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def _timeline(minutes: int = 2880) -> pd.Series:
    return pd.Series(pd.date_range("2025-01-01", periods=minutes, freq="1min"))


# --------------------------------------------------------------------
# Metrik per sampel
# --------------------------------------------------------------------
def test_calibration_error_zero_for_perfect_model():
    labels = np.array([0, 0, 1, 1] * 250)
    probabilities = labels.astype(float)
    assert evaluate.calibration_error(labels, probabilities, bins=10) < 1e-9


def test_calibration_error_high_for_overconfident_model():
    rng = np.random.default_rng(0)
    labels = (rng.random(1000) < 0.1).astype(int)
    probabilities = np.full(1000, 0.9)  # klaim 90 %, kenyataan 10 %
    assert evaluate.calibration_error(labels, probabilities, bins=10) > 0.7


def test_sample_metrics_handle_single_class():
    """AUC tidak terdefinisi tanpa dua kelas; harus NaN, bukan galat."""
    labels = np.zeros(100, dtype=int)
    probabilities = np.full(100, 0.01)
    metrics = evaluate.sample_metrics(labels, probabilities, threshold=0.5)
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["roc_auc"])
    assert metrics["precision"] == 0.0


def test_pr_auc_exposes_what_roc_auc_hides():
    """Pada kelas sangat jarang, ROC-AUC menyanjung model yang lemah.

    Model di bawah ini menempatkan seluruh 40 sampel positif di sepersepuluh
    teratas — ROC-AUC di atas 0,9. Tetapi sepersepuluh teratas juga berisi
    sekitar 2.000 sampel negatif, sehingga presisi terbaiknya sekitar dua
    persen. Inilah keadaan yang sesungguhnya dihadapi operator, dan hanya
    PR-AUC yang menunjukkannya.
    """
    rng = np.random.default_rng(1)
    size = 20_000
    labels = np.zeros(size, dtype=int)
    positives = rng.choice(size, 40, replace=False)
    labels[positives] = 1

    # Negatif tersebar merata 0..1, jadi sekitar 2.000 di antaranya ikut
    # menempati sepersepuluh teratas bersama seluruh sampel positif.
    probabilities = rng.random(size)
    probabilities[positives] = 0.9 + rng.random(40) * 0.1

    metrics = evaluate.sample_metrics(labels, probabilities, threshold=0.9)
    assert metrics["roc_auc"] > 0.9
    assert metrics["pr_auc"] < 0.2
    assert metrics["precision"] < 0.05


# --------------------------------------------------------------------
# Metrik per event
# --------------------------------------------------------------------
def test_event_detected_when_alarm_precedes_it():
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))
    event_starts.iloc[1000] = 1

    probabilities = np.zeros(len(stamps))
    probabilities[900:1000] = 0.9  # alarm mulai 100 menit sebelum event

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.5,
        detection_window_minutes=180,
    )
    assert metrics["event_count"] == 1.0
    assert metrics["events_detected"] == 1.0
    assert metrics["event_detection_rate"] == 1.0
    assert metrics["missed_event_rate"] == 0.0
    # Warning horizon dari alarm PERTAMA, bukan terakhir.
    assert metrics["median_warning_horizon_minutes"] == pytest.approx(100.0)


def test_event_missed_when_alarm_comes_too_late():
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))
    event_starts.iloc[1000] = 1

    probabilities = np.zeros(len(stamps))
    probabilities[1010:1100] = 0.9  # alarm baru setelah event

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.5,
        detection_window_minutes=180,
    )
    assert metrics["events_detected"] == 0.0
    assert metrics["missed_event_rate"] == 1.0


def test_event_missed_when_alarm_outside_detection_window():
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))
    event_starts.iloc[1000] = 1

    probabilities = np.zeros(len(stamps))
    probabilities[500:600] = 0.9  # 400 menit sebelum event

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.5,
        detection_window_minutes=180,
    )
    assert metrics["events_detected"] == 0.0


def test_short_burst_counted_once():
    """Operator melihat satu alarm menyala terus, bukan ratusan alarm.

    Rentetan yang lebih pendek dari satu jendela deteksi dihitung sekali.
    """
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))

    probabilities = np.zeros(len(stamps))
    probabilities[100:200] = 0.9  # 100 menit, di bawah jendela 180 menit

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.5,
        detection_window_minutes=180,
    )
    span_days = metrics["evaluation_span_days"]
    assert metrics["false_alarms_per_day"] == pytest.approx(1.0 / span_days)


def test_long_burst_counted_per_detection_window():
    """Alarm yang menyala berjam-jam bukan satu gangguan tunggal.

    Menghitungnya satu kali membuka celah: ambang mendekati nol membuat
    alarm menyala nyaris sepanjang waktu, tercatat sebagai satu rentetan,
    lalu tampak seperti sistem yang sangat tenang.
    """
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))

    probabilities = np.zeros(len(stamps))
    probabilities[100:820] = 0.9  # 720 menit menyala terus

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.5,
        detection_window_minutes=180,
    )
    span_days = metrics["evaluation_span_days"]
    # 719 menit dibagi jendela 180 menit membulat ke atas menjadi empat.
    assert metrics["false_alarms_per_day"] == pytest.approx(4.0 / span_days)


def test_always_on_alarm_has_high_duty_cycle():
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))
    probabilities = np.full(len(stamps), 0.9)

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.001,
        detection_window_minutes=180,
    )
    assert metrics["alarm_duty_cycle"] > 0.9


def test_threshold_rejects_always_on_alarm(settings):
    """Ambang yang membuat alarm menyala terus harus ditolak.

    Model di bawah ini mendeteksi setiap event — tetapi hanya dengan cara
    membunyikan alarm nyaris sepanjang waktu.
    """
    rng = np.random.default_rng(11)
    stamps = _timeline(6000)
    event_starts = pd.Series(np.zeros(len(stamps)))
    for position in (1500, 3500, 5500):
        event_starts.iloc[position] = 1

    # Skor tinggi hampir di mana-mana: ambang rendah akan menyala terus.
    probabilities = 0.4 + rng.random(len(stamps)) * 0.5

    threshold, info = evaluate.choose_threshold(
        stamps, probabilities, event_starts, settings
    )
    max_duty = float(settings.thresholds["alarm"]["max_alarm_duty_cycle"])
    achieved = evaluate.event_metrics(
        stamps,
        probabilities,
        event_starts,
        threshold,
        int(settings.model["evaluation"]["detection_window_minutes"]),
        int(settings.thresholds["alarm"]["persistence_minutes"]),
    )
    assert achieved["alarm_duty_cycle"] <= max_duty + 1e-9


def test_separate_bursts_counted_separately():
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))

    probabilities = np.zeros(len(stamps))
    probabilities[100:200] = 0.9
    probabilities[1000:1100] = 0.9
    probabilities[2000:2100] = 0.9

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.5,
        detection_window_minutes=180,
    )
    span_days = metrics["evaluation_span_days"]
    assert metrics["false_alarms_per_day"] == pytest.approx(3.0 / span_days)


def test_alarms_near_event_are_not_false_alarms():
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))
    event_starts.iloc[1000] = 1

    probabilities = np.zeros(len(stamps))
    probabilities[900:1000] = 0.9

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.5,
        detection_window_minutes=180,
    )
    assert metrics["false_alarms_per_day"] == 0.0


def test_persistence_suppresses_short_bursts():
    """Rentetan yang lebih pendek dari syarat ketahanan tidak jadi alarm."""
    stamps = _timeline()
    event_starts = pd.Series(np.zeros(len(stamps)))

    probabilities = np.zeros(len(stamps))
    probabilities[500:503] = 0.9  # hanya tiga menit

    metrics = evaluate.event_metrics(
        stamps, probabilities, event_starts, threshold=0.5,
        detection_window_minutes=180, persistence_minutes=10,
    )
    assert metrics["false_alarms_per_day"] == 0.0


# --------------------------------------------------------------------
# Pemilihan ambang
# --------------------------------------------------------------------
def test_threshold_respects_false_alarm_budget(settings):
    rng = np.random.default_rng(2)
    stamps = _timeline(10_000)
    event_starts = pd.Series(np.zeros(len(stamps)))
    for position in (2000, 5000, 8000):
        event_starts.iloc[position] = 1

    probabilities = rng.random(len(stamps)) * 0.2
    for position in (2000, 5000, 8000):
        probabilities[position - 90 : position] += 0.7

    threshold, info = evaluate.choose_threshold(
        stamps, probabilities, event_starts, settings
    )
    budget = float(settings.thresholds["alarm"]["target_false_alarms_per_day"])

    achieved = evaluate.event_metrics(
        stamps,
        probabilities,
        event_starts,
        threshold,
        int(settings.model["evaluation"]["detection_window_minutes"]),
        int(settings.thresholds["alarm"]["persistence_minutes"]),
    )
    assert achieved["false_alarms_per_day"] <= budget
    assert 0.0 < threshold < 1.0
    assert "anggaran" in info["note"].lower() or "sasaran" in info["note"].lower()


def test_threshold_reports_when_budget_unreachable(settings):
    """Sasaran yang tidak tercapai harus dikatakan, bukan dilonggarkan."""
    rng = np.random.default_rng(3)
    stamps = _timeline(3000)
    event_starts = pd.Series(np.zeros(len(stamps)))
    event_starts.iloc[1500] = 1

    # Model tak berguna: skor acak tanpa hubungan dengan event.
    probabilities = rng.random(len(stamps))

    _, info = evaluate.choose_threshold(stamps, probabilities, event_starts, settings)
    assert isinstance(info["note"], str) and info["note"]


def test_comparison_table_sorted_by_pr_auc(settings):
    results = []
    for name, pr_auc in (("a", 0.1), ("b", 0.5), ("c", 0.3)):
        result = evaluate.EvaluationResult(
            model_name=name, horizon="blocking_next_60m", dataset="test",
            threshold=0.5,
        )
        result.sample_metrics = {"pr_auc": pr_auc, "roc_auc": 0.9}
        result.event_metrics = {"false_alarms_per_day": 1.0}
        results.append(result)

    table = evaluate.comparison_table(results)
    assert list(table["model"]) == ["b", "c", "a"]
