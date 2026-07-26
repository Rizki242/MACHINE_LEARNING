"""ETL Event Registry PLTU Jeranjang Unit 1.

Membaca berkas PARETO riwayat gangguan (XLSX) dan mengubahnya menjadi
Event Registry terstruktur sesuai skema README §15.

Bentuk sumber yang harus ditangani
----------------------------------
* Nama sheet mengandung spasi di ujung — lihat ``core.constants``.
* Header tersebar pada dua baris dengan sel gabungan; data mulai baris
  ke-4 (``skiprows=3``).
* Sebelas kolom pertama identik pada sheet derating dan outage, tetapi
  kolom sesudahnya berbeda: derating punya ``MW DERATING`` lalu
  ``MWh HILANG``; outage langsung ``MWh HILANG``.
* Nama equipment punya 95 varian ejaan dan kapitalisasi.
* Sebagian sel mengandung mojibake dari karakter non-breaking yang rusak.

Prinsip
-------
Kolom yang tidak ada di sumber (``coal_source``, ``coal_blending``,
``clinker_found``) diisi kosong. Tidak ditebak.

Durasi dihitung ulang dari ``selesai - mulai`` dan dibandingkan dengan
kolom durasi sumber. Selisihnya dicatat, bukan ditimpa diam-diam.

Penggunaan
----------
    python -m backend.app.data.event_etl
    python -m backend.app.data.event_etl --cross-check
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.constants import (
    COMMON_COLUMNS,
    DERATING_EXTRA_COLUMNS,
    EVENT_ID_PREFIX,
    OUTAGE_EXTRA_COLUMNS,
    RECORD_KIND_DERATING,
    RECORD_KIND_OUTAGE,
    SHEET_DERATING,
    SHEET_OUTAGE,
    SOURCE_SKIPROWS,
    UNIT_ID,
)
from backend.app.core.logging import get_logger
from backend.app.data.schemas import (
    ValidationReport,
    coerce_event_registry,
    validate_event_registry,
)

LOGGER = get_logger(__name__)

#: Berkas sumber utama — snapshot terbaru.
PRIMARY_SOURCE_GLOB = "05_PARETO*.xlsx"
#: Snapshot bulan sebelumnya, dipakai untuk uji konsistensi sumber.
SECONDARY_SOURCE_GLOB = "04_PARETO*.xlsx"

#: Karakter pengganti dan non-breaking yang muncul akibat mojibake.
_BROKEN_CHARS = re.compile(r"[� ​  ]+")
_WHITESPACE = re.compile(r"\s+")

#: Status yang menandakan unit berhenti tanpa rencana.
_FORCED_OUTAGE_PREFIXES = ("FO", "FOL")


# --------------------------------------------------------------------
# Pembersihan teks
# --------------------------------------------------------------------
def clean_text(value: Any) -> str | None:
    """Rapikan sel teks sumber.

    Memperbaiki mojibake (mis. ``GENERATOR�OUTPUT�BREAKER``),
    menormalkan bentuk Unicode, dan meratakan spasi berlebih.
    Mengembalikan ``None`` untuk sel kosong.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if text.strip().lower() in {"", "nan", "nat", "none", "-"}:
        return None
    text = unicodedata.normalize("NFKC", text)
    text = _BROKEN_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text or None


def canonical_equipment(raw: str | None, taxonomy: dict[str, Any]) -> str | None:
    """Petakan nama equipment mentah ke daftar kanonik.

    Pencocokan substring tanpa memandang huruf besar-kecil. Urutan entri
    di ``event_taxonomy.yaml`` menentukan pemenang — entri pertama yang
    cocok dipakai.
    """
    if not raw:
        return None
    haystack = raw.casefold()
    for entry in taxonomy.get("equipment_canonical", []) or []:
        for pattern in entry.get("patterns", []) or []:
            if pattern.casefold() in haystack:
                return str(entry["name"])
    return None


# --------------------------------------------------------------------
# Pembacaan sumber
# --------------------------------------------------------------------
def find_source_file(directory: Path, pattern: str) -> Path:
    """Cari satu berkas sumber yang cocok dengan pola."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Tidak ada berkas yang cocok dengan {pattern!r} di {directory}. "
            "Letakkan berkas PARETO riwayat gangguan di direktori data."
        )
    if len(matches) > 1:
        LOGGER.warning(
            "Ditemukan %d berkas cocok %s; memakai yang terbaru: %s",
            len(matches),
            pattern,
            matches[-1].name,
        )
    return matches[-1]


def read_source_sheet(path: Path, sheet: str, record_kind: str) -> pd.DataFrame:
    """Baca satu sheet jurnal dan kembalikan baris data yang bersih.

    Baris judul, baris header sisa, dan baris total tersaring lewat dua
    syarat: kolom ``gangguan`` terisi dan kolom ``mulai`` berupa tanggal
    yang sah.
    """
    extra = (
        DERATING_EXTRA_COLUMNS
        if record_kind == RECORD_KIND_DERATING
        else OUTAGE_EXTRA_COLUMNS
    )
    needed = len(COMMON_COLUMNS) + len(extra)

    raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=SOURCE_SKIPROWS)

    # Nomor baris pada berkas Excel asli, untuk penelusuran balik.
    raw = raw.assign(source_row=raw.index + SOURCE_SKIPROWS + 1)
    source_row = raw["source_row"]

    if raw.shape[1] < needed:
        raise ValueError(
            f"{path.name} sheet {sheet!r}: hanya {raw.shape[1]} kolom, "
            f"minimal {needed} dibutuhkan. Struktur berkas berubah."
        )

    frame = raw.iloc[:, : len(COMMON_COLUMNS)].copy()
    frame.columns = COMMON_COLUMNS
    for index, name in extra.items():
        frame[name] = raw.iloc[:, index]
    frame["source_row"] = source_row

    frame["mulai"] = pd.to_datetime(frame["mulai"], errors="coerce")
    frame["selesai"] = pd.to_datetime(frame["selesai"], errors="coerce")

    before = len(frame)
    keep = frame["gangguan"].notna() & frame["mulai"].notna()
    frame = frame.loc[keep].copy()
    LOGGER.info(
        "%s | %-26s | %3d baris data dari %3d baris mentah",
        path.name[:2],
        sheet.strip(),
        len(frame),
        before,
    )

    frame["record_kind"] = record_kind
    frame["source_sheet"] = sheet
    frame["source_file"] = path.name
    return frame


# --------------------------------------------------------------------
# Transformasi
# --------------------------------------------------------------------
def _infer_trip_status(status: str | None, record_kind: str) -> str:
    """Turunkan ``trip_status`` dari kode status sumber.

    Forced outage (``FO``, ``FOL``) berarti unit berhenti tanpa rencana.
    Ini turunan, bukan pembacaan langsung dari log proteksi — kolom ini
    tidak boleh dipakai sebagai bukti terjadinya trip.
    """
    if record_kind != RECORD_KIND_OUTAGE or not status:
        return "no"
    code = status.upper().strip()
    return "yes" if code.startswith(_FORCED_OUTAGE_PREFIXES) else "no"


def _to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalise_records(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Ubah baris sumber mentah menjadi baris Event Registry.

    Belum berisi ``event_type``, ``event_location``, dan ``severity`` —
    ketiganya diisi oleh :mod:`backend.app.rules.event_classifier`.
    """
    taxonomy = settings.event_taxonomy
    records: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        gangguan = clean_text(row.gangguan)
        equipment_raw = clean_text(row.equipment)
        penyebab = clean_text(row.penyebab)
        penyelesaian = clean_text(row.penyelesaian)
        status = clean_text(row.status)

        start = row.mulai
        end = row.selesai if pd.notna(row.selesai) else pd.NaT

        duration_source = _to_float(row.durasi_jam)
        duration_computed: float | None = None
        if pd.notna(start) and pd.notna(end):
            duration_computed = (end - start).total_seconds() / 3600.0

        mismatch: float | None = None
        if duration_computed is not None and duration_source is not None:
            mismatch = duration_computed - duration_source

        records.append(
            {
                # Skema README §15
                "event_id": None,  # diberikan setelah pengurutan waktu
                "unit_id": UNIT_ID,
                "start_time": start,
                "end_time": end,
                "event_type": None,
                "event_location": None,
                "severity": None,
                "initial_symptom": gangguan,
                "operator_action": None,  # tidak dibedakan di sumber
                "maintenance_action": penyelesaian,
                "derating_mw": _to_float(getattr(row, "derating_mw", None)),
                "trip_status": _infer_trip_status(status, row.record_kind),
                "clinker_found": None,  # tidak ada di sumber
                "coal_source": None,  # tidak ada di sumber
                "coal_blending": None,  # tidak ada di sumber
                "notes": penyebab,
                # Kolom audit
                "source_file": row.source_file,
                "source_sheet": row.source_sheet,
                "source_row": int(row.source_row),
                "equipment_raw": equipment_raw,
                "equipment_canonical": canonical_equipment(equipment_raw, taxonomy),
                "status_raw": status,
                "record_kind": row.record_kind,
                "duration_hours": duration_computed,
                "duration_hours_source": duration_source,
                "duration_mismatch_hours": mismatch,
                "mwh_lost": _to_float(getattr(row, "mwh_lost", None)),
                "classification_method": None,
                "classification_confidence": None,
                "needs_review": False,
            }
        )

    result = pd.DataFrame.from_records(records)
    return result.sort_values("start_time", kind="stable").reset_index(drop=True)


def assign_event_ids(frame: pd.DataFrame, taxonomy: dict[str, Any]) -> pd.DataFrame:
    """Beri ``event_id`` berformat ``JRG-U1-<KODE>-<NNN>`` (README §15).

    Penomoran berjalan per kode jenis event, urut waktu mulai. Event yang
    tidak berhubungan dengan furnace memakai kode ``MISC`` agar tetap
    dapat ditelusuri tanpa mengotori penomoran kelas blocking.
    """
    codes = {
        name: str(spec.get("code", name[:6].upper()))
        for name, spec in (taxonomy.get("event_types") or {}).items()
    }

    result = frame.sort_values("start_time", kind="stable").copy()
    counters: dict[str, int] = {}
    event_ids: list[str] = []

    for event_type in result["event_type"]:
        code = codes.get(str(event_type), "MISC")
        counters[code] = counters.get(code, 0) + 1
        event_ids.append(f"{EVENT_ID_PREFIX}-{code}-{counters[code]:03d}")

    result["event_id"] = event_ids
    return result.reset_index(drop=True)


# --------------------------------------------------------------------
# Alur utama
# --------------------------------------------------------------------
def extract_registry(
    source_path: Path, settings: Settings | None = None
) -> pd.DataFrame:
    """Baca kedua sheet Unit 1 dari satu berkas dan normalisasi isinya."""
    settings = settings or get_settings()
    frames = [
        read_source_sheet(source_path, SHEET_DERATING, RECORD_KIND_DERATING),
        read_source_sheet(source_path, SHEET_OUTAGE, RECORD_KIND_OUTAGE),
    ]
    combined = pd.concat(frames, ignore_index=True)
    return normalise_records(combined, settings)


def build_event_registry(
    settings: Settings | None = None,
    source_path: Path | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Bangun Event Registry lengkap: ekstraksi, klasifikasi, validasi."""
    settings = settings or get_settings()
    source_path = source_path or find_source_file(
        settings.paths.raw_data, PRIMARY_SOURCE_GLOB
    )
    LOGGER.info("Sumber: %s", source_path)

    registry = extract_registry(source_path, settings)

    # Impor di sini supaya modul rules tidak dimuat saat hanya ekstraksi
    # yang dibutuhkan (mis. dari pengujian unit).
    from backend.app.rules.event_classifier import classify_registry

    registry = classify_registry(registry, settings)
    registry = assign_event_ids(registry, settings.event_taxonomy)
    registry = coerce_event_registry(registry)

    report = validate_event_registry(registry)
    return registry, report


def cross_check_sources(settings: Settings | None = None) -> dict[str, Any]:
    """Bandingkan snapshot April dan Mei 2026.

    Setiap event yang ada di snapshot lama harus muncul identik di
    snapshot baru. Selisih apa pun dilaporkan sebagai temuan konsistensi
    sumber, bukan diperbaiki diam-diam.
    """
    settings = settings or get_settings()
    new_path = find_source_file(settings.paths.raw_data, PRIMARY_SOURCE_GLOB)
    old_path = find_source_file(settings.paths.raw_data, SECONDARY_SOURCE_GLOB)

    new = extract_registry(new_path, settings)
    old = extract_registry(old_path, settings)

    def key(frame: pd.DataFrame) -> pd.Series:
        return (
            frame["record_kind"].astype(str)
            + "|"
            + frame["start_time"].astype(str)
            + "|"
            + frame["initial_symptom"].fillna("").astype(str).str.casefold()
        )

    old_keys = set(key(old))
    new_keys = set(key(new))
    missing = sorted(old_keys - new_keys)
    added = sorted(new_keys - old_keys)

    result = {
        "old_file": old_path.name,
        "new_file": new_path.name,
        "old_rows": int(len(old)),
        "new_rows": int(len(new)),
        "rows_added": len(added),
        "rows_missing_from_new": len(missing),
        "missing_sample": missing[:10],
        "added_sample": added[:10],
        "consistent": not missing,
    }

    if missing:
        LOGGER.warning(
            "Konsistensi sumber: %d event ada di %s tetapi hilang di %s",
            len(missing),
            old_path.name,
            new_path.name,
        )
    else:
        LOGGER.info(
            "Konsistensi sumber: seluruh %d event snapshot lama ada di snapshot baru",
            len(old),
        )
    return result


def write_registry(registry: pd.DataFrame, settings: Settings) -> dict[str, Path]:
    """Tulis registry ke CSV dan Parquet."""
    settings.paths.ensure()
    csv_path = settings.paths.processed / "event_registry_unit1.csv"
    parquet_path = settings.paths.processed / "event_registry_unit1.parquet"

    registry.to_csv(csv_path, index=False, encoding="utf-8")
    try:
        registry.to_parquet(parquet_path, index=False)
    except (ImportError, ValueError) as exc:  # pragma: no cover
        LOGGER.warning("Parquet dilewati: %s", exc)
        parquet_path = csv_path

    return {"csv": csv_path, "parquet": parquet_path}


def load_registry(settings: Settings | None = None) -> pd.DataFrame:
    """Muat Event Registry yang sudah diproses dari disk."""
    settings = settings or get_settings()
    parquet_path = settings.paths.processed / "event_registry_unit1.parquet"
    csv_path = settings.paths.processed / "event_registry_unit1.csv"

    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "Event Registry belum dibangun. Jalankan: "
            "python -m backend.app.data.event_etl"
        )
    return coerce_event_registry(frame)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ETL Event Registry PLTU Jeranjang Unit 1"
    )
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="Bandingkan snapshot April dan Mei 2026 untuk uji konsistensi",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    registry, report = build_event_registry(settings)
    paths = write_registry(registry, settings)

    print()
    print("=" * 72)
    print("EVENT REGISTRY — PLTU JERANJANG UNIT 1")
    print("=" * 72)
    print(f"Total event         : {len(registry)}")
    for kind, count in registry["record_kind"].value_counts().items():
        print(f"  {kind:<16}: {count}")
    print(
        f"Rentang waktu       : {registry['start_time'].min():%Y-%m-%d} "
        f"s/d {registry['start_time'].max():%Y-%m-%d}"
    )
    print()
    print("Jenis event:")
    for event_type, count in registry["event_type"].value_counts().items():
        print(f"  {event_type:<32}: {count:>4}")
    print()
    print("Severity (README §15):")
    for severity, count in registry["severity"].value_counts().sort_index().items():
        print(f"  {severity}: {count}")
    print()
    print(report.render())
    print()
    print(f"CSV     : {paths['csv']}")
    print(f"Parquet : {paths['parquet']}")

    if args.cross_check:
        result = cross_check_sources(settings)
        out_path = settings.paths.reports / "source_cross_check.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print()
        print("-" * 72)
        print("UJI KONSISTENSI SUMBER")
        print("-" * 72)
        print(f"{result['old_file']}: {result['old_rows']} event")
        print(f"{result['new_file']}: {result['new_rows']} event")
        print(f"Bertambah di snapshot baru : {result['rows_added']}")
        print(f"Hilang dari snapshot baru  : {result['rows_missing_from_new']}")
        print(f"Konsisten                  : {'ya' if result['consistent'] else 'TIDAK'}")
        print(f"Rincian: {out_path}")

    return 0 if report.is_clean else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
