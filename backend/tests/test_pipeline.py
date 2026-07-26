"""Pengujian pelabelan, pembagian waktu, kualitas data, dan rule engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.core.config import get_settings
from backend.app.data import quality
from backend.app.features.baseline import assign_load_zone, fit_baseline
from backend.app.models.train import build_labels, build_split_plan, stratified_positions
from backend.app.rules import risk_rules


@pytest.fixture(scope="module")
def settings():
    return get_settings()


# --------------------------------------------------------------------
# Pelabelan
# --------------------------------------------------------------------
def test_labels_look_forward_only():
    """Label bernilai 1 sebelum event, bukan sesudahnya."""
    size = 200
    event_start = pd.Series(np.zeros(size))
    event_start.iloc[100] = 1.0

    labels = build_labels(event_start, [{"name": "next_30m", "minutes": 30}])
    values = labels["next_30m"]

    # 30 menit sebelum event: positif.
    assert values[70] == 1
    assert values[99] == 1
    # Tepat pada saat event dan sesudahnya: bukan lagi prediksi.
    assert values[100] == 0
    assert values[120] == 0
    # Jauh sebelum jendela: negatif.
    assert values[60] == 0


def test_longer_horizon_contains_shorter():
    size = 500
    event_start = pd.Series(np.zeros(size))
    event_start.iloc[300] = 1.0

    labels = build_labels(
        event_start,
        [
            {"name": "next_30m", "minutes": 30},
            {"name": "next_60m", "minutes": 60},
            {"name": "next_180m", "minutes": 180},
        ],
    )
    assert labels["next_30m"].sum() == 30
    assert labels["next_60m"].sum() == 60
    assert labels["next_180m"].sum() == 180
    # Setiap positif horizon pendek juga positif di horizon panjang.
    assert np.all(labels["next_60m"][labels["next_30m"] == 1] == 1)


def test_labels_empty_without_events():
    labels = build_labels(
        pd.Series(np.zeros(100)), [{"name": "next_60m", "minutes": 60}]
    )
    assert labels["next_60m"].sum() == 0


# --------------------------------------------------------------------
# Pembagian waktu
# --------------------------------------------------------------------
def test_split_is_temporal_and_embargoed(settings):
    """Himpunan tidak boleh tumpang tindih, dan harus dipisah embargo."""
    try:
        plan = build_split_plan(settings)
    except FileNotFoundError:
        pytest.skip("Data sintetis belum dibangkitkan.")

    embargo = pd.Timedelta(hours=float(settings.model["split"]["embargo_hours"]))
    assert plan.validation_start - plan.train_end >= 2 * embargo
    assert plan.test_start - plan.validation_end >= 2 * embargo
    assert set(plan.train_years).isdisjoint(plan.validation_years)
    assert set(plan.validation_years).isdisjoint(plan.test_years)
    assert max(plan.train_years) < min(plan.validation_years)
    assert max(plan.validation_years) < min(plan.test_years)


def test_stratified_positions_keeps_all_positives():
    labels = np.zeros(10_000, dtype=np.int8)
    labels[:120] = 1
    rng = np.random.default_rng(0)
    positions = stratified_positions(labels, 1000, rng)

    assert labels[positions].sum() == 120
    assert len(positions) <= 1000 + 120
    assert np.all(np.diff(positions) > 0)


# --------------------------------------------------------------------
# Zona beban dan baseline
# --------------------------------------------------------------------
def test_assign_load_zone_matches_settings(settings):
    frame = pd.DataFrame({"unit_load_mw": [0.0, 5.0, 7.0, 15.0, 20.0, 24.0]})
    zones = assign_load_zone(frame, settings)
    assert list(zones) == [
        "startup",
        "startup",
        "low_load",
        "medium_load",
        "high_load",
        "near_rated_load",
    ]


def test_baseline_uses_robust_statistics(settings):
    """Pencilan besar tidak boleh menggeser pusat baseline."""
    size = 4000
    rng = np.random.default_rng(1)
    values = rng.normal(9.0, 0.2, size)
    values[:200] = 90.0  # pencilan ekstrem

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=size, freq="1min"),
            "unit_load_mw": 20.0,
            "bed_differential_pressure": values,
            "is_running": 1,
        }
    )
    baseline = fit_baseline(frame, ["bed_differential_pressure"], settings)
    centre = baseline.statistics["high_load"]["bed_differential_pressure"]["centre"]
    assert 8.5 < centre < 9.5


def test_baseline_corrects_for_load_within_zone(settings):
    """Sinyal normal di tepi zona tidak boleh terlihat menyimpang.

    Zona medium membentang 10 sampai 17 MW, dan temperatur bed bergerak
    terus mengikuti beban di sepanjang rentang itu. Dengan satu nilai
    tengah per zona, unit yang berjalan di tepi bawah zona akan tampak
    menyimpang jauh ke bawah pada hampir semua sinyal — dan rule engine
    akan menyala menunjuk peralatan yang sehat.
    """
    rng = np.random.default_rng(5)
    size = 8000
    load = rng.uniform(10.1, 17.0, size)
    # Sinyal yang murni mengikuti beban, ditambah derau kecil.
    signal = 700.0 + 12.0 * load + rng.normal(0.0, 3.0, size)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=size, freq="1min"),
            "unit_load_mw": load,
            "main_bed_temperature": signal,
            "is_running": 1,
        }
    )
    baseline = fit_baseline(frame, ["main_bed_temperature"], settings)
    frame["load_zone"] = assign_load_zone(frame, settings)
    deviation = baseline.deviation(frame)["main_bed_temperature_dev"]

    edge = frame["unit_load_mw"] < 10.6
    middle = frame["unit_load_mw"].between(13.0, 14.0)

    assert abs(float(deviation[edge].median())) < 0.5
    assert abs(float(deviation[middle].median())) < 0.5


def test_baseline_still_flags_genuine_deviation(settings):
    """Koreksi beban tidak boleh menutupi penyimpangan yang sungguhan."""
    rng = np.random.default_rng(6)
    size = 8000
    load = rng.uniform(10.1, 17.0, size)
    signal = 700.0 + 12.0 * load + rng.normal(0.0, 3.0, size)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=size, freq="1min"),
            "unit_load_mw": load,
            "main_bed_temperature": signal,
            "is_running": 1,
        }
    )
    baseline = fit_baseline(frame, ["main_bed_temperature"], settings)

    # Kondisi abnormal: 40 derajat di atas yang seharusnya pada beban itu.
    abnormal = frame.head(50).copy()
    abnormal["main_bed_temperature"] += 40.0
    abnormal["load_zone"] = assign_load_zone(abnormal, settings)

    deviation = baseline.deviation(abnormal)["main_bed_temperature_dev"]
    assert float(deviation.median()) > 5.0


def test_baseline_falls_back_when_load_is_constant(settings):
    """Beban tetap membuat tren tidak dapat dipasang; harus mundur aman."""
    size = 3000
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=size, freq="1min"),
            "unit_load_mw": 20.0,
            "bed_differential_pressure": np.linspace(8.0, 10.0, size),
            "is_running": 1,
        }
    )
    baseline = fit_baseline(frame, ["bed_differential_pressure"], settings)
    entry = baseline.statistics["high_load"]["bed_differential_pressure"]
    assert entry["slope"] == 0.0
    assert entry["intercept"] == pytest.approx(entry["centre"])


def test_baseline_deviation_is_zero_at_centre(settings):
    size = 3000
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=size, freq="1min"),
            "unit_load_mw": 20.0,
            "bed_differential_pressure": np.linspace(8.0, 10.0, size),
            "is_running": 1,
        }
    )
    baseline = fit_baseline(frame, ["bed_differential_pressure"], settings)
    frame["load_zone"] = assign_load_zone(frame, settings)
    deviation = baseline.deviation(frame)["bed_differential_pressure_dev"]

    assert abs(float(deviation.iloc[size // 2])) < 0.05
    assert float(deviation.iloc[0]) < -1.0
    assert float(deviation.iloc[-1]) > 1.0


# --------------------------------------------------------------------
# Kualitas data
# --------------------------------------------------------------------
def _quality_frame(size: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=size, freq="1min"),
            "unit_load_mw": rng.normal(20.0, 0.3, size),
            "bed_differential_pressure": rng.normal(9.0, 0.2, size),
            "oxygen_o2": rng.normal(3.8, 0.1, size),
        }
    )


def test_quality_score_high_for_clean_data(settings):
    result = quality.assess(_quality_frame(), settings)
    assert result.score > 95.0
    assert result.completeness == pytest.approx(1.0)


def test_quality_score_drops_with_missing_data(settings):
    """Skor harus turun sebanding porsi data yang hilang, bukan runtuh.

    Diuji relatif terhadap data bersih: kelengkapan berbobot 0,40, jadi
    kehilangan 11 persen nilai menurunkan skor sekitar 4,5 poin. Angka
    mutlaknya bergantung bobot di thresholds.yaml dan tidak layak
    dipatok di dalam pengujian.
    """
    clean = quality.assess(_quality_frame(), settings)

    frame = _quality_frame()
    frame.loc[100:300, "bed_differential_pressure"] = np.nan
    degraded = quality.assess(frame, settings)

    assert degraded.completeness < 0.9
    assert degraded.score < clean.score
    weight = float(settings.thresholds["data_quality"]["weights"]["completeness"])
    expected_drop = 100.0 * weight * (clean.completeness - degraded.completeness)
    assert degraded.score == pytest.approx(clean.score - expected_drop, abs=0.5)


def test_stuck_signal_detected(settings):
    frame = _quality_frame()
    frame.loc[100:300, "oxygen_o2"] = 3.8
    result = quality.assess(frame, settings)
    assert result.stuck_score < 0.95


def test_range_violation_detected(settings):
    frame = _quality_frame()
    frame.loc[100:200, "oxygen_o2"] = 95.0  # di luar 0-21 persen
    result = quality.assess(frame, settings)
    assert result.range_score < 1.0


def test_stuck_fraction_ignores_short_runs():
    """Nilai sama dua-tiga kali berturut-turut wajar pada sinyal tenang."""
    values = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0, 4.0] * 20)
    assert quality.stuck_fraction(values, window=30, tolerance=1e-9) == 0.0


# --------------------------------------------------------------------
# Rule engine
# --------------------------------------------------------------------
def _rule_frame(size: int = 10) -> pd.DataFrame:
    settings = get_settings()
    columns: set[str] = set()
    for spec in settings.thresholds["rules"].values():
        for condition in spec["conditions"]:
            columns.add(str(condition["feature"]))
    return pd.DataFrame({column: np.zeros(size) for column in sorted(columns)})


def test_rule_engine_quiet_when_nothing_deviates(settings):
    frame = _rule_frame()
    _, score = risk_rules.evaluate_rules(frame, settings)
    assert float(score.max()) == 0.0


def test_agglomeration_rule_fires(settings):
    frame = _rule_frame()
    frame.loc[5, "bed_differential_pressure_dev"] = 4.0
    frame.loc[5, "main_bed_air_resistance_dev"] = 4.0
    frame.loc[5, "bed_temp_spread"] = 90.0

    per_rule, score = risk_rules.evaluate_rules(frame, settings)
    assert per_rule.loc[5, "main_bed_agglomeration"] > 0
    assert float(score.iloc[5]) > 0
    assert float(score.iloc[0]) == 0.0


def test_triggers_report_reason_and_area(settings):
    frame = _rule_frame()
    frame.loc[2, "coal_feeder_imbalance"] = 0.6
    frame.loc[2, "coal_feeder_command_deviation"] = 0.4

    triggers = risk_rules.rule_triggers_at(frame, 2, settings)
    assert triggers
    assert all(trigger.reason for trigger in triggers)
    assert risk_rules.suspected_area(triggers, settings) == "coal_feeder_unknown"


def test_combined_score_stays_in_range(settings):
    index = range(5)
    rule = pd.Series([0.0, 25.0, 50.0, 75.0, 100.0], index=index)
    model = pd.Series([100.0, 75.0, 50.0, 25.0, 0.0], index=index)
    anomaly = pd.Series([50.0] * 5, index=index)

    combined = risk_rules.combine_scores(rule, model, anomaly, settings)
    assert combined.between(0.0, 100.0).all()


def test_combined_score_reweights_when_model_absent(settings):
    rule = pd.Series([100.0, 0.0])
    combined = risk_rules.combine_scores(rule, None, None, settings)
    # Rule engine sendirian harus dapat mencapai 100.
    assert combined.iloc[0] == pytest.approx(100.0)
    assert combined.iloc[1] == pytest.approx(0.0)


def test_persistence_suppresses_single_spike(settings):
    values = pd.Series([0.0] * 20)
    values.iloc[10] = 95.0
    smoothed = risk_rules.apply_persistence(values, settings)
    assert float(smoothed.iloc[10]) == 0.0


def test_persistence_keeps_sustained_high_score(settings):
    minutes = int(settings.thresholds["alarm"]["persistence_minutes"])
    values = pd.Series([0.0] * 10 + [90.0] * (minutes + 5))
    smoothed = risk_rules.apply_persistence(values, settings)
    assert float(smoothed.iloc[-1]) == pytest.approx(90.0)


def test_confidence_penalised_by_poor_data(settings):
    probability = pd.Series([0.95, 0.95])
    data_quality = pd.Series([99.0, 40.0])
    confidence = risk_rules.confidence_from_quality(probability, data_quality, settings)
    assert confidence.iloc[0] > confidence.iloc[1]


def test_confidence_low_when_model_undecided(settings):
    probability = pd.Series([0.5])
    confidence = risk_rules.confidence_from_quality(
        probability, pd.Series([100.0]), settings
    )
    assert float(confidence.iloc[0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("score", "label"),
    [(10, "Normal"), (30, "Early Warning"), (60, "Warning"), (81, "High Risk"), (95, "Critical")],
)
def test_status_mapping(settings, score, label):
    assert risk_rules.status_for(score, settings)[0] == label


# --------------------------------------------------------------------
# Pemetaan probabilitas ke Risk Score
# --------------------------------------------------------------------
def test_probability_at_threshold_maps_to_fifty():
    """Ambang keputusan harus jatuh tepat di batas bawah status Warning."""
    threshold = 0.0217
    score = risk_rules.probability_to_risk(pd.Series([threshold]), threshold)
    assert float(score.iloc[0]) == pytest.approx(50.0, abs=1e-6)


def test_probability_mapping_is_monotonic():
    threshold = 0.02
    probabilities = pd.Series([1e-6, 1e-4, 0.005, 0.02, 0.1, 0.5, 0.95])
    score = risk_rules.probability_to_risk(probabilities, threshold)
    assert score.is_monotonic_increasing
    assert score.between(0.0, 100.0).all()


def test_probability_mapping_separates_rare_event_scale():
    """Perbedaan pada skala peristiwa jarang harus tetap terbaca.

    Tanpa pemetaan log-odds, 0,001 dan 0,01 sama-sama membulat menjadi
    "sekitar nol persen" dan model yang benar terlihat seperti tidak
    menemukan apa pun.
    """
    threshold = 0.02
    low = float(risk_rules.probability_to_risk(pd.Series([0.001]), threshold).iloc[0])
    high = float(risk_rules.probability_to_risk(pd.Series([0.01]), threshold).iloc[0])
    assert high - low > 10.0


def test_probability_mapping_handles_extremes():
    score = risk_rules.probability_to_risk(pd.Series([0.0, 1.0]), 0.02)
    assert float(score.iloc[0]) >= 0.0
    assert float(score.iloc[1]) <= 100.0
    assert np.isfinite(score.to_numpy()).all()


def test_every_rule_feature_is_produced_by_engineering(settings):
    """Setiap fitur yang dirujuk thresholds.yaml harus benar-benar dibuat.

    Fitur yang dirujuk tetapi tidak pernah terbentuk membuat aturannya
    kehilangan sebagian syarat tanpa suara — aturan tetap berjalan, tetapi
    dengan bobot yang lebih kecil dari yang dimaksudkan.
    """
    from backend.app.features.engineering import BASELINE_DERIVED_COLUMNS

    produced: set[str] = set()

    # Fitur turunan dan bergulir yang dihasilkan build_features.
    base_columns = settings.model["features"]["rolling_base_columns"]
    for column in base_columns:
        produced.add(f"{column}_dev")
        for window in settings.model["features"]["rolling_std_windows_minutes"]:
            produced.add(f"{column}_rolling_std_{window}m")
        for window in settings.model["features"]["slope_windows_minutes"]:
            produced.add(f"{column}_slope_{window}m")
    for column in BASELINE_DERIVED_COLUMNS:
        produced.add(column)
        produced.add(f"{column}_dev")
    produced.update(
        {
            "bed_temp_spread",
            "main_aux_temp_spread",
            "bed_pressure_imbalance",
            "coal_feeder_imbalance",
            "coal_feeder_command_deviation",
            "aux_air_flow_imbalance",
            "air_to_fuel_ratio",
            "slag_discharge_failure",
            "ash_cooler_current_slope",
            "combustion_instability_index",
            "return_system_disturbance",
            "coal_feeder_current_max",
            "coal_feeder_current_spread",
            "aux_bed_pressure_rolling_std_15m",
        }
    )

    referenced = {
        str(condition["feature"])
        for spec in settings.thresholds["rules"].values()
        for condition in spec["conditions"]
    }
    missing = sorted(referenced - produced)
    assert not missing, (
        "Fitur berikut dirujuk thresholds.yaml tetapi tidak pernah dibentuk: "
        f"{missing}"
    )
