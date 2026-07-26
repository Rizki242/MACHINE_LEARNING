"""Pengujian ETL dan klasifikasi Event Registry.

Angka acuan berasal dari audit langsung berkas sumber:
sheet ``UNIT 1 Derating (Jurnal) `` berisi 302 baris data dan sheet
``UNIT 1 Outage (Jurnal) `` berisi 127 baris data, seluruhnya dalam
rentang Januari 2016 sampai Mei 2026.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.core.config import get_settings
from backend.app.core.constants import (
    RECORD_KIND_DERATING,
    RECORD_KIND_OUTAGE,
    SHEET_DERATING,
    SHEET_OUTAGE,
)
from backend.app.data import event_etl
from backend.app.rules import event_classifier

EXPECTED_DERATING_ROWS = 302
EXPECTED_OUTAGE_ROWS = 127
EXPECTED_TOTAL = EXPECTED_DERATING_ROWS + EXPECTED_OUTAGE_ROWS


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def registry(settings):
    """Muat registry hasil olahan; bangun bila belum ada."""
    try:
        return event_etl.load_registry(settings)
    except FileNotFoundError:  # pragma: no cover - jalur pertama kali
        frame, _ = event_etl.build_event_registry(settings)
        event_etl.write_registry(frame, settings)
        return frame


# --------------------------------------------------------------------
# Bentuk sumber
# --------------------------------------------------------------------
def test_sheet_names_keep_trailing_space():
    """Nama sheet sumber memang berspasi di ujung; jangan dirapikan.

    Bila konstanta ini pernah di-strip(), pd.read_excel akan gagal
    menemukan sheet-nya dan seluruh ETL berhenti.
    """
    assert SHEET_DERATING.endswith(" ")
    assert SHEET_OUTAGE.endswith(" ")


def test_row_counts_match_source_audit(registry):
    """Jumlah baris harus persis sama dengan hasil audit berkas sumber."""
    counts = registry["record_kind"].value_counts()
    assert counts[RECORD_KIND_DERATING] == EXPECTED_DERATING_ROWS
    assert counts[RECORD_KIND_OUTAGE] == EXPECTED_OUTAGE_ROWS
    assert len(registry) == EXPECTED_TOTAL


def test_date_range_matches_source(registry):
    assert registry["start_time"].min() >= pd.Timestamp("2016-01-01")
    assert registry["start_time"].max() <= pd.Timestamp("2026-06-01")


def test_event_ids_unique_and_formatted(registry):
    assert registry["event_id"].is_unique
    assert registry["event_id"].str.startswith("JRG-U1-").all()


def test_columns_absent_from_source_stay_empty(registry):
    """Kolom yang tidak ada di sumber tidak boleh ditebak (README §15)."""
    for column in ("coal_source", "coal_blending", "clinker_found"):
        assert registry[column].isna().all(), (
            f"Kolom {column} tidak ada di berkas sumber, tetapi terisi. "
            "Nilai yang ditebak akan menyesatkan analisis kualitas batubara."
        )


def test_severity_within_readme_range(registry):
    assert registry["severity"].between(0, 4).all()


def test_end_time_never_precedes_start(registry):
    both = registry["start_time"].notna() & registry["end_time"].notna()
    assert (registry.loc[both, "end_time"] >= registry.loc[both, "start_time"]).all()


def test_duration_mismatch_is_recorded_not_hidden(registry):
    """Selisih durasi harus tercatat, bukan ditimpa diam-diam."""
    assert "duration_mismatch_hours" in registry.columns
    assert "duration_hours_source" in registry.columns


# --------------------------------------------------------------------
# Pembersihan teks
# --------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Coal   Feeder 1A ", "Coal Feeder 1A"),
        ("GENERATOR�OUTPUT�BREAKER", "GENERATOR OUTPUT BREAKER"),
        ("nan", None),
        ("", None),
        (None, None),
        ("-", None),
    ],
)
def test_clean_text(raw, expected):
    assert event_etl.clean_text(raw) == expected


def test_mojibake_repaired_in_registry(registry):
    """Karakter pengganti tidak boleh tersisa di kolom mana pun."""
    for column in ("equipment_raw", "initial_symptom", "notes", "maintenance_action"):
        values = registry[column].dropna().astype(str)
        assert not values.str.contains("�").any()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Coal Feeder 1A", "COAL_FEEDER"),
        ("Coalfeeder 1A", "COAL_FEEDER"),
        ("COAL FEEDER", "COAL_FEEDER"),
        ("Furnace", "FURNACE"),
        ("FURNACE", "FURNACE"),
        ("Embedded Tube", "EMBEDDED_TUBE"),
        ("CONDENSOR", "CONDENSER"),
        ("Kondensor", "CONDENSER"),
        ("CWP 1B", "CWP"),
        ("Bunker 1A; Bunker 1B", "COAL_HANDLING"),
    ],
)
def test_equipment_normalisation(settings, raw, expected):
    assert event_etl.canonical_equipment(raw, settings.event_taxonomy) == expected


# --------------------------------------------------------------------
# Klasifikasi
# --------------------------------------------------------------------
@pytest.mark.parametrize(
    ("gangguan", "penyebab", "equipment", "expected"),
    [
        (
            "Gangguan Indikasi Aglomerasi Furnace",
            "Indikasi slagging/aglomerasi di furnace",
            "Furnace",
            "furnace_agglomeration",
        ),
        (
            "Temperatur Furnace High (Indikasi Slagging Area Furnace)",
            "Indikasi Slagging Area Furnace",
            "Boiler",
            "furnace_agglomeration",
        ),
        (
            "Blocking pada coal feeder",
            "Terjadinya blocking batubara pada coal feeder",
            "Coal Feeder",
            "coal_feeder_blocking",
        ),
        (
            "Blocking pada chute coal bunker to coalfeeder",
            "Batubara ngeblock di coal bunker",
            "Bunker 1A; Bunker 1B",
            "coal_feeder_blocking",
        ),
        (
            "Indikasi blocking di Sealpot Cyclone",
            "Refractory rontok dan menutupi area sealpot",
            "CYCLONE",
            "return_system_disturbance",
        ),
        (
            "Kebocoran Embedded Wall Tube Rear Boiler nomor 10",
            "Bed Material Erosion",
            "EMBEDDED TUBE",
            "tube_leak_bed_erosion",
        ),
        (
            "Furnace PDT Tinggi",
            "Bukaan Damper IDF sudah Maksimal",
            "BOILER",
            "air_distribution_disturbance",
        ),
        (
            "Cleaning condensor dan debris filter",
            "Sampah",
            "CONDENSER",
            "not_furnace_related",
        ),
        (
            "CWP trip Level air laut low",
            "Sedimentasi",
            "CWP",
            "not_furnace_related",
        ),
    ],
)
def test_keyword_classification(settings, gangguan, penyebab, equipment, expected):
    """Kasus nyata dari berkas sumber harus terklasifikasi benar."""
    canonical = event_etl.canonical_equipment(equipment, settings.event_taxonomy)
    verdict = event_classifier.match_keywords(
        gangguan, penyebab, equipment, canonical, settings.event_taxonomy
    )
    assert verdict.event_type == expected


def test_typo_tolerance_via_char_ngrams(settings):
    """Salah ketik di sumber (``funace``, ``slaging``) tetap tertangkap."""
    canonical = event_etl.canonical_equipment("Furnace", settings.event_taxonomy)
    verdict = event_classifier.match_keywords(
        "Pengaturan pembebeban akibat Indikasi slagging pada funace",
        "Indikasi slagging/aglomerasi di furnace",
        "Furnace",
        canonical,
        settings.event_taxonomy,
    )
    assert verdict.event_type == "furnace_agglomeration"


def test_no_furnace_event_labelled_unrelated(registry):
    """Tidak boleh ada event bertanda furnace yang tersaring keluar lingkup."""
    unrelated = registry.loc[registry["event_type"] == "not_furnace_related"]
    pattern = r"furnace|funace|\bbed\b|feeder|slag|aglomer|agglomer"
    hits = unrelated["initial_symptom"].astype(str).str.lower().str.contains(
        pattern, regex=True, na=False
    )
    assert not hits.any(), (
        "Event berikut menyebut furnace tetapi dilabeli di luar lingkup: "
        f"{unrelated.loc[hits, 'initial_symptom'].head().tolist()}"
    )


def test_zero_severity_types_have_severity_zero(settings, registry):
    zero_types = settings.event_taxonomy["zero_severity_event_types"]
    subset = registry.loc[registry["event_type"].isin(zero_types)]
    assert (subset["severity"] == 0).all()


def test_blocking_events_have_nonzero_severity(settings, registry):
    blocking = settings.event_taxonomy["blocking_event_types"]
    subset = registry.loc[registry["event_type"].isin(blocking)]
    assert (subset["severity"] >= 2).all()


def test_outage_severity_at_least_three(registry):
    """Outage berarti unit berhenti — minimal severity 3 (README §15)."""
    outages = registry.loc[
        (registry["record_kind"] == RECORD_KIND_OUTAGE) & (registry["severity"] > 0)
    ]
    assert (outages["severity"] >= 3).all()


def test_classification_coverage(registry):
    """Sisa yang belum terklasifikasi harus kecil dan selalu ditandai."""
    unclassified = registry["event_type"] == "unclassified"
    assert unclassified.sum() <= 25
    assert registry.loc[unclassified, "needs_review"].all()


def test_blocking_event_count_meets_audit_floor(settings, registry):
    """Audit manual menemukan minimal 81 baris berkata kunci blocking."""
    blocking = settings.event_taxonomy["blocking_event_types"]
    assert int(registry["event_type"].isin(blocking).sum()) >= 81
