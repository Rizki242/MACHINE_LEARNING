"""Pengujian generator timeseries sintetis.

Yang diuji bukan "apakah datanya terlihat bagus", melainkan apakah data
itu layak dipakai melatih model: rentangnya masuk akal secara fisik,
event nyata benar-benar tersalin ke timeline, degradasi muncul SEBELUM
event, dan penanda sumber data tidak pernah hilang.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.core.config import get_settings
from backend.app.core.constants import DATA_SOURCE_COLUMN, DATA_SOURCE_SYNTHETIC
from backend.app.data import synthetic


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def registry(settings):
    from backend.app.data.event_etl import load_registry

    try:
        return load_registry(settings)
    except FileNotFoundError:
        pytest.skip("Event Registry belum dibangun.")


@pytest.fixture(scope="module")
def chunk(settings, registry):
    """Satu potongan 30 hari, cukup untuk menguji sifat-sifat penting."""
    index = pd.date_range("2024-11-01", "2024-11-30 23:59", freq="1min", name="timestamp")
    spec = synthetic.GenerationSpec(
        start_year=2024, end_year=2024, freq="1min", seed=12345
    )
    rng = np.random.default_rng(spec.seed)
    return synthetic.generate_chunk(index, registry, settings, spec, rng)


# --------------------------------------------------------------------
# Penanda sumber data
# --------------------------------------------------------------------
def test_every_row_labelled_synthetic(chunk):
    """README §21: data sintetis dan data aktual tidak boleh tertukar."""
    assert (chunk[DATA_SOURCE_COLUMN] == DATA_SOURCE_SYNTHETIC).all()
    assert chunk["synthetic_seed"].nunique() == 1


def test_generation_is_reproducible(settings, registry):
    index = pd.date_range("2024-06-01", "2024-06-03", freq="1min", name="timestamp")
    spec = synthetic.GenerationSpec(2024, 2024, "1min", seed=777)

    first = synthetic.generate_chunk(
        index, registry, settings, spec, np.random.default_rng(spec.seed)
    )
    second = synthetic.generate_chunk(
        index, registry, settings, spec, np.random.default_rng(spec.seed)
    )
    pd.testing.assert_series_equal(
        first["main_bed_temperature"], second["main_bed_temperature"]
    )


# --------------------------------------------------------------------
# Kewajaran fisik
# --------------------------------------------------------------------
def test_all_signals_inside_plausible_ranges(settings, chunk):
    """Setiap sinyal harus berada di rentang fisik pada units.yaml."""
    offenders: list[str] = []
    for column, bounds in settings.plausible_ranges.items():
        if column not in chunk.columns:
            continue
        values = chunk[column].dropna()
        if values.empty:
            continue
        if values.min() < bounds[0] or values.max() > bounds[1]:
            offenders.append(
                f"{column}: [{values.min():.1f}, {values.max():.1f}] "
                f"di luar {bounds}"
            )
    assert not offenders, offenders


def test_bed_temperature_realistic_while_running(chunk):
    """CFB beroperasi pada 800-950 °C; di luar itu tidak masuk akal."""
    running = chunk.loc[chunk["is_running"] == 1, "main_bed_temperature"].dropna()
    assert 780.0 < running.quantile(0.01)
    assert running.quantile(0.99) < 960.0


def test_auxiliary_bed_cooler_than_main_bed(chunk):
    """Auxiliary bed lebih rendah 500 mm dan udaranya terpisah (README §3)."""
    running = chunk.loc[chunk["is_running"] == 1]
    difference = (
        running["main_bed_temperature"] - running["front_aux_bed_temperature"]
    ).dropna()
    assert difference.median() > 0


def test_coal_flow_consistent_with_energy_balance(settings, chunk):
    """Aliran batubara harus sepadan dengan produksi steam dan nilai kalor."""
    running = chunk.loc[(chunk["is_running"] == 1) & (chunk["unit_load_mw"] > 18)]
    ratio = (running["coal_flow_total"] / running["main_steam_flow"]).dropna()
    # Sekitar 0,2 t batubara per ton steam untuk batubara 3.611 kcal/kg.
    assert 0.15 < ratio.median() < 0.30


def test_feeder_flows_sum_to_total(chunk):
    feeder_columns = [c for c in chunk.columns if c.endswith("_flow") and "feeder" in c]
    total = chunk[feeder_columns].sum(axis=1)
    difference = (total - chunk["coal_flow_total"]).abs()
    assert float(difference.median()) < 0.05


def test_all_load_zones_represented(settings):
    """Baseline per zona hanya bisa dihitung bila tiap zona pernah terisi."""
    from backend.app.features.baseline import assign_load_zone

    try:
        frame = synthetic.load_synthetic(
            settings, years=[2024], columns=["timestamp", "unit_load_mw", "is_running"]
        )
    except FileNotFoundError:
        pytest.skip("Data sintetis belum dibangkitkan.")

    zones = assign_load_zone(frame, settings).value_counts()
    for zone in (z["name"] for z in settings.load_zones):
        assert zones.get(zone, 0) > 0, f"Zona {zone} tidak pernah terisi."


def test_normal_feeder_imbalance_below_rule_threshold(settings, chunk):
    """Ketidakseimbangan saat normal harus jauh di bawah ambang aturan.

    Kalau tidak, fitur itu akan menyala terus dan kehilangan artinya.
    """
    from backend.app.features.engineering import feeder_imbalance

    threshold = next(
        float(condition["value"])
        for condition in settings.thresholds["rules"]["coal_feeder_blocking"][
            "conditions"
        ]
        if condition["feature"] == "coal_feeder_imbalance"
    )
    quiet = chunk.loc[
        (chunk["is_running"] == 1) & (chunk["ramp_coal_feeder_blocking"] < 0.01)
    ]
    assert float(feeder_imbalance(quiet).quantile(0.95)) < threshold


# --------------------------------------------------------------------
# Keselarasan dengan event nyata
# --------------------------------------------------------------------
def test_degradation_precedes_event(settings, registry):
    """Ramp degradasi harus naik SEBELUM event, bukan sesudahnya."""
    blocking = settings.event_taxonomy["blocking_event_types"]
    candidates = registry.loc[
        registry["event_type"].isin(blocking)
        & registry["start_time"].between("2024-01-01", "2024-12-01")
    ].sort_values("start_time")
    if candidates.empty:
        pytest.skip("Tidak ada event blocking pada rentang uji.")

    event = candidates.iloc[0]
    start = pd.Timestamp(event["start_time"])
    index = pd.date_range(
        start - pd.Timedelta(hours=8), start + pd.Timedelta(hours=1),
        freq="1min",
        name="timestamp",
    )
    spec = synthetic.GenerationSpec(2024, 2024, "1min", seed=999)
    frame = synthetic.generate_chunk(
        index, registry, settings, spec, np.random.default_rng(spec.seed)
    )

    ramp_column = f"ramp_{event['event_type']}"
    ramp = frame.set_index("timestamp")[ramp_column]
    before = ramp.loc[: start - pd.Timedelta(minutes=1)]

    assert float(before.max()) > 0.3, "Tidak ada degradasi sebelum event."
    # Delapan jam sebelum event harus masih tenang: ramp terpanjang 240 menit.
    assert float(ramp.iloc[:60].max()) < 0.2


def test_outage_periods_have_zero_load(settings, registry):
    """Selama outage nyata, unit harus benar-benar berhenti."""
    outages = registry.loc[
        (registry["record_kind"] == "outage")
        & registry["start_time"].between("2024-01-01", "2024-11-01")
        & registry["end_time"].notna()
    ]
    outages = outages.loc[
        (outages["end_time"] - outages["start_time"]) > pd.Timedelta(hours=6)
    ]
    if outages.empty:
        pytest.skip("Tidak ada outage panjang pada rentang uji.")

    event = outages.iloc[0]
    index = pd.date_range(
        event["start_time"] + pd.Timedelta(hours=1),
        event["start_time"] + pd.Timedelta(hours=4),
        freq="1min",
        name="timestamp",
    )
    spec = synthetic.GenerationSpec(2024, 2024, "1min", seed=555)
    frame = synthetic.generate_chunk(
        index, registry, settings, spec, np.random.default_rng(spec.seed)
    )
    assert float(frame["unit_load_mw"].max()) < 1.0
    assert int(frame["is_running"].sum()) == 0


# --------------------------------------------------------------------
# Cacat data yang disengaja
# --------------------------------------------------------------------
def test_data_defects_present(chunk):
    """Modul kualitas data tidak akan teruji bila datanya sempurna."""
    analog = [
        column
        for column in chunk.columns
        if chunk[column].dtype == np.float32 and not column.startswith("ramp_")
    ]
    missing = chunk[analog].isna().to_numpy().mean()
    assert 0.0 < missing < 0.05, (
        "Cacat data harus ada tetapi tidak boleh mendominasi; "
        f"proporsi hilang saat ini {missing:.5f}"
    )
