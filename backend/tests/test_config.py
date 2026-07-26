"""Pengujian pemuatan dan validasi konfigurasi."""

from __future__ import annotations

import pytest

from backend.app.core.config import get_settings


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def test_load_zones_contiguous(settings):
    """Zona beban harus bersambung tanpa celah (README §10)."""
    zones = settings.load_zones
    assert len(zones) == 5
    for previous, current in zip(zones, zones[1:]):
        assert previous["max_mw"] == current["min_mw"]
    assert zones[0]["min_mw"] == 0
    assert zones[-1]["max_mw"] == settings.units["unit"]["capacity_mw"]


@pytest.mark.parametrize(
    ("load_mw", "expected"),
    [
        (0.0, "startup"),
        (5.0, "startup"),
        (5.1, "low_load"),
        (10.0, "low_load"),
        (10.1, "medium_load"),
        (17.0, "medium_load"),
        (17.1, "high_load"),
        (22.0, "high_load"),
        (22.1, "near_rated_load"),
        (25.0, "near_rated_load"),
        (26.0, "near_rated_load"),
    ],
)
def test_load_zone_boundaries(settings, load_mw, expected):
    """Batas zona mengikuti README §10: bawah eksklusif kecuali zona pertama."""
    assert settings.load_zone_for(load_mw) == expected


def test_risk_bands_cover_full_range(settings):
    """Band risiko harus menutup 0-100 tanpa celah (README §13)."""
    bands = settings.risk_bands
    assert bands[0]["min"] == 0
    assert bands[-1]["max"] == 100
    for previous, current in zip(bands, bands[1:]):
        assert current["min"] == previous["max"] + 1


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (0, "Normal"),
        (24, "Normal"),
        (25, "Early Warning"),
        (49, "Early Warning"),
        (50, "Warning"),
        (74, "Warning"),
        (75, "High Risk"),
        (89, "High Risk"),
        (90, "Critical"),
        (100, "Critical"),
    ],
)
def test_risk_band_labels(settings, score, label):
    assert settings.risk_band_for(score)["label"] == label


def test_hybrid_weights_sum_to_one(settings):
    total = sum(float(v) for v in settings.thresholds["hybrid_weights"].values())
    assert abs(total - 1.0) < 1e-9


def test_data_quality_weights_sum_to_one(settings):
    weights = settings.thresholds["data_quality"]["weights"]
    assert abs(sum(float(v) for v in weights.values()) - 1.0) < 1e-9


def test_no_dcs_tag_mapped_yet(settings):
    """Fase 0 belum punya pemetaan tag DCS; semuanya harus masih kosong.

    Bila pengujian ini gagal, artinya seseorang sudah mengisi tag asli.
    Itu kabar baik, tetapi pengujian ini harus diperbarui bersama
    verifikasi engineer instrumentasi.
    """
    assert settings.dcs_tag_lookup() == {}


def test_priority_a_tags_present(settings):
    """Seluruh tag Priority A README §7 harus terdaftar."""
    names = set(settings.standard_names("priority_a"))
    required = {
        "timestamp",
        "unit_load_mw",
        "main_steam_flow",
        "main_steam_pressure",
        "main_steam_temperature",
        "coal_flow_total",
        "main_bed_temperature",
        "front_aux_bed_temperature",
        "rear_aux_bed_temperature",
        "main_bed_pressure",
        "bed_differential_pressure",
        "furnace_pressure",
        "main_bed_air_flow",
        "auxiliary_bed_air_flow",
        "primary_air_pressure",
        "secondary_air_flow",
        "oxygen_o2",
        "carbon_monoxide_co",
        "id_fan_current",
        "bottom_ash_discharge_status",
        "ash_cooler_motor_current",
        "operator_event",
    }
    assert required.issubset(names)


def test_deployment_mode_is_offline(settings):
    """Fase 0 tidak boleh tersambung ke historian atau DCS (README §19, §20)."""
    assert settings.deployment_mode == "offline"


def test_rule_thresholds_reference_known_operators(settings):
    from backend.app.rules.risk_rules import OPERATORS

    for rule_name, spec in settings.thresholds["rules"].items():
        for condition in spec["conditions"]:
            assert condition["operator"] in OPERATORS, (
                f"Aturan {rule_name} memakai operator tak dikenal: "
                f"{condition['operator']}"
            )
            assert condition["points"] > 0
            assert condition["reason"]
