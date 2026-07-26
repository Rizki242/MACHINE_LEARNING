"""Pengujian rekayasa fitur.

Pengujian terpenting di berkas ini adalah yang memeriksa kebocoran masa
depan. Kebocoran itu gagal secara senyap: tidak ada galat, hanya metrik
bagus yang menyesatkan. Satu-satunya cara menangkapnya adalah menguji
sifatnya secara langsung.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.core.config import get_settings
from backend.app.features import engineering as fe


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


# --------------------------------------------------------------------
# Kebocoran masa depan
# --------------------------------------------------------------------
def test_rolling_mean_does_not_see_future():
    """Sinyal berundak: rata-rata sebelum undakan tidak boleh terpengaruh."""
    values = _series([0.0] * 50 + [100.0] * 50)
    result = fe.rolling_mean(values, 10)
    # Tepat sebelum undakan, seluruh jendela masih nol.
    assert result.iloc[49] == pytest.approx(0.0)
    # Setelah undakan, nilainya naik.
    assert result.iloc[59] == pytest.approx(100.0)


def test_rolling_std_does_not_see_future():
    values = _series([5.0] * 60 + [80.0] * 40)
    result = fe.rolling_std(values, 15)
    assert result.iloc[59] == pytest.approx(0.0, abs=1e-9)
    assert result.iloc[65] > 0.0


def test_rolling_slope_does_not_see_future():
    values = _series([0.0] * 60 + list(np.arange(1, 41, dtype=float)))
    result = fe.rolling_slope(values, 30)
    assert result.iloc[59] == pytest.approx(0.0, abs=1e-9)
    assert result.iloc[75] > 0.0


def test_rolling_slope_matches_least_squares():
    """Bentuk tertutup harus sama dengan regresi kuadrat terkecil biasa."""
    rng = np.random.default_rng(0)
    values = pd.Series(np.cumsum(rng.normal(0, 1, 200)))
    window = 20
    result = fe.rolling_slope(values, window)

    for position in (50, 100, 150):
        segment = values.iloc[position - window + 1 : position + 1].to_numpy()
        expected = np.polyfit(np.arange(window), segment, 1)[0]
        assert result.iloc[position] == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_rate_of_change_uses_past_only():
    values = _series(list(range(100)))
    result = fe.rate_of_change(values, 5)
    assert result.iloc[10] == pytest.approx(5.0)
    assert pd.isna(result.iloc[0])


def test_no_feature_correlates_with_future_step(settings):
    """Uji menyeluruh: undakan di masa depan tidak boleh terlihat sebelumnya.

    Dibangun satu dataframe yang seluruh sinyalnya datar lalu melonjak di
    tengah. Setiap kolom fitur yang dihasilkan harus tetap konstan pada
    seluruh baris sebelum lonjakan. Kolom mana pun yang berubah sebelum
    lonjakan berarti melihat ke depan.
    """
    size = 400
    step_at = 300
    stamps = pd.date_range("2025-01-01", periods=size, freq="1min")

    base = {
        "timestamp": stamps,
        "unit_load_mw": 20.0,
        "main_steam_flow": 104.0,
        "main_steam_pressure": 9.0,
        "main_steam_temperature": 535.0,
        "coal_flow_total": 21.0,
        "main_bed_temperature": 870.0,
        "front_aux_bed_temperature": 852.0,
        "rear_aux_bed_temperature": 848.0,
        "main_bed_pressure": 10.5,
        "bed_differential_pressure": 9.0,
        "aux_bed_pressure": 6.5,
        "furnace_pressure": -100.0,
        "main_bed_air_flow": 48000.0,
        "auxiliary_bed_air_flow": 22000.0,
        "front_aux_bed_air_flow": 11000.0,
        "rear_aux_bed_air_flow": 11000.0,
        "primary_air_pressure": 11.0,
        "secondary_air_flow": 32000.0,
        "oxygen_o2": 3.8,
        "carbon_monoxide_co": 90.0,
        "id_fan_current": 95.0,
        "ash_cooler_motor_current": 25.0,
        "bottom_ash_discharge_status": 1,
        "bottom_ash_flow": 2.0,
        "return_leg_temperature": 825.0,
        "return_leg_pressure": 3.9,
        "u_valve_air_pressure": 9.0,
        "separator_differential_pressure": 1.6,
        "coal_feeder_1_flow": 5.25,
        "coal_feeder_2_flow": 5.25,
        "coal_feeder_3_flow": 5.25,
        "coal_feeder_4_flow": 5.25,
        "coal_feeder_1_command": 5.25,
        "coal_feeder_2_command": 5.25,
        "coal_feeder_3_command": 5.25,
        "coal_feeder_4_command": 5.25,
        "coal_feeder_1_current": 19.0,
        "coal_feeder_2_current": 19.0,
        "coal_feeder_3_current": 19.0,
        "coal_feeder_4_current": 19.0,
    }
    frame = pd.DataFrame(base)

    # Lonjakan besar pada beberapa sinyal kunci, mulai dari `step_at`.
    for column, delta in (
        ("bed_differential_pressure", 6.0),
        ("main_bed_temperature", 60.0),
        ("carbon_monoxide_co", 700.0),
        ("primary_air_pressure", 4.0),
        ("coal_feeder_1_flow", -3.0),
    ):
        frame.loc[step_at:, column] = frame.loc[step_at:, column] + delta

    features = fe.build_features(frame, settings, baseline=None)

    # Baris 120 sampai tepat sebelum lonjakan: seluruh jendela terpanjang
    # (60 menit) sudah terisi dan isinya konstan.
    window = slice(120, step_at)
    offenders: list[str] = []
    for column in features.columns:
        if column == "timestamp":
            continue
        segment = features[column].iloc[window].dropna()
        if segment.empty:
            continue
        if float(segment.max() - segment.min()) > 1e-6:
            offenders.append(column)

    assert not offenders, (
        "Fitur berikut berubah sebelum lonjakan terjadi — indikasi "
        f"kebocoran masa depan: {offenders[:10]}"
    )


# --------------------------------------------------------------------
# Formula README §9
# --------------------------------------------------------------------
def test_bed_temperature_spread_matches_readme():
    frame = pd.DataFrame(
        {
            "main_bed_temperature": [900.0, 800.0],
            "front_aux_bed_temperature": [850.0, 860.0],
            "rear_aux_bed_temperature": [820.0, 870.0],
        }
    )
    result = fe.bed_temperature_spread(frame)
    assert result.iloc[0] == pytest.approx(80.0)
    assert result.iloc[1] == pytest.approx(70.0)


def test_feeder_imbalance_matches_readme():
    """(max - min) / rata-rata, sesuai formula README §9."""
    frame = pd.DataFrame(
        {
            "coal_feeder_1_flow": [6.0],
            "coal_feeder_2_flow": [4.0],
            "coal_feeder_3_flow": [5.0],
            "coal_feeder_4_flow": [5.0],
        }
    )
    result = fe.feeder_imbalance(frame)
    assert result.iloc[0] == pytest.approx((6.0 - 4.0) / 5.0)


def test_air_resistance_matches_readme():
    pressure = _series([12.0])
    flow = _series([48000.0])
    assert fe.air_resistance(pressure, flow).iloc[0] == pytest.approx(12.0 / 48000.0)


def test_slag_discharge_failure_requires_both_conditions():
    """Katup terbuka DAN abu tidak keluar — bukan salah satunya saja."""
    frame = pd.DataFrame(
        {
            "bottom_ash_discharge_status": [1, 1, 0, 0],
            "bottom_ash_flow": [0.05, 2.0, 0.05, 2.0],
        }
    )
    result = fe.slag_discharge_failure(frame)
    assert list(result) == [1.0, 0.0, 0.0, 0.0]


def test_feeder_command_deviation_detects_gap():
    frame = pd.DataFrame(
        {
            "coal_feeder_1_flow": [3.0, 5.0],
            "coal_feeder_2_flow": [5.0, 5.0],
            "coal_feeder_1_command": [5.0, 5.0],
            "coal_feeder_2_command": [5.0, 5.0],
        }
    )
    result = fe.feeder_command_deviation(frame)
    assert result.iloc[0] == pytest.approx(0.4)
    assert result.iloc[1] == pytest.approx(0.0)


def test_air_resistance_survives_zero_flow():
    """Aliran nol tidak boleh menghasilkan pembagian dengan nol."""
    result = fe.air_resistance(_series([10.0]), _series([0.0]))
    assert np.isfinite(result.iloc[0])


def test_build_features_rejects_forward_looking_config(settings):
    """Konfigurasi yang mematikan backward_only harus ditolak.

    Penjaga ini penting: mengubah satu baris YAML tidak boleh cukup untuk
    diam-diam menghidupkan jendela yang melihat ke depan.
    """
    import dataclasses

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="1min"),
            "main_bed_temperature": np.arange(10.0),
        }
    )
    broken_model = {
        **settings.model,
        "features": {**settings.model["features"], "backward_only": False},
    }
    broken = dataclasses.replace(settings, model=broken_model)

    with pytest.raises(ValueError, match="backward_only"):
        fe.build_features(frame, broken)
