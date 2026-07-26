"""Konstanta tetap FurnaceGuard AI.

Berisi nilai yang tidak boleh berubah tanpa perubahan kode: nama sheet
sumber, kode identitas unit, dan skema kolom keluaran.

Ambang batas dan parameter model TIDAK ada di sini — semuanya di
``config/*.yaml`` supaya engineer bisa mengubahnya tanpa menyentuh Python.
"""

from __future__ import annotations

# --------------------------------------------------------------------
# Identitas unit
# --------------------------------------------------------------------
UNIT_ID = "UNIT_1"
PLANT_CODE = "JRG"
EVENT_ID_PREFIX = f"{PLANT_CODE}-U1"

# --------------------------------------------------------------------
# Sumber data mentah
#
# PERINGATAN: nama sheet di bawah mengandung spasi di ujung. Itu apa
# adanya di file Excel sumber. Jangan di-strip(), jangan dirapikan —
# pd.read_excel mencocokkan nama sheet secara harfiah.
# --------------------------------------------------------------------
SHEET_DERATING = "UNIT 1 Derating (Jurnal) "
SHEET_OUTAGE = "UNIT 1 Outage (Jurnal) "

#: Baris data pertama pada kedua sheet. Baris 0 judul, 1-2 header
#: bertingkat dengan sel gabungan.
SOURCE_SKIPROWS = 3

#: Sebelas kolom pertama identik pada kedua sheet.
COMMON_COLUMNS = [
    "no",
    "no_urut",
    "gangguan",
    "no_asset_maximo",
    "equipment",
    "penyebab",
    "penyelesaian",
    "status",
    "mulai",
    "selesai",
    "durasi_jam",
]

#: Kolom setelah indeks 10 BERBEDA antar sheet.
#: Derating: kolom 11 = MW DERATING, kolom 12 = MWh HILANG.
#: Outage  : kolom 11 = MWh HILANG (tidak ada kolom MW derating).
DERATING_EXTRA_COLUMNS = {11: "derating_mw", 12: "mwh_lost"}
OUTAGE_EXTRA_COLUMNS = {11: "mwh_lost"}

# --------------------------------------------------------------------
# Skema keluaran event registry (README §15 + kolom asal untuk audit)
# --------------------------------------------------------------------
EVENT_REGISTRY_COLUMNS = [
    # Skema README §15
    "event_id",
    "unit_id",
    "start_time",
    "end_time",
    "event_type",
    "event_location",
    "severity",
    "initial_symptom",
    "operator_action",
    "maintenance_action",
    "derating_mw",
    "trip_status",
    "clinker_found",
    "coal_source",
    "coal_blending",
    "notes",
    # Kolom tambahan untuk audit dan penelusuran ke sumber
    "source_file",
    "source_sheet",
    "source_row",
    "equipment_raw",
    "equipment_canonical",
    "status_raw",
    "record_kind",
    "duration_hours",
    "duration_hours_source",
    "duration_mismatch_hours",
    "mwh_lost",
    "classification_method",
    "classification_confidence",
    "needs_review",
]

#: Label jenis catatan sumber.
RECORD_KIND_DERATING = "derating"
RECORD_KIND_OUTAGE = "outage"

#: Nilai event_type untuk baris yang tidak berhubungan dengan furnace.
EVENT_TYPE_UNRELATED = "not_furnace_related"

#: Nilai event_type ketika tidak ada aturan yang cocok sama sekali.
EVENT_TYPE_UNCLASSIFIED = "unclassified"

# --------------------------------------------------------------------
# Penanda sumber data — data sintetis dan data pembangkit tidak boleh
# tercampur dalam satu berkas (README §21).
# --------------------------------------------------------------------
DATA_SOURCE_COLUMN = "data_source"
DATA_SOURCE_SYNTHETIC = "synthetic"
DATA_SOURCE_ACTUAL = "actual"

#: Disclaimer wajib pada setiap keluaran laporan (README §24).
DISCLAIMER = (
    "FurnaceGuard AI adalah alat analisis dan pendukung keputusan. "
    "Hasilnya bukan perintah operasi, bukan pengganti SOP, operator, "
    "engineer boiler, maupun interlock dan proteksi. Tidak menjamin "
    "blocking akan atau tidak akan terjadi. Wajib diverifikasi dengan "
    "kondisi aktual. Keputusan akhir berada pada operator dan engineer "
    "yang berwenang."
)

#: Peringatan wajib pada metrik yang dihitung di atas data sintetis.
SYNTHETIC_METRIC_WARNING = (
    "Angka unjuk kerja berikut dihitung di atas DATA SINTETIS, bukan data "
    "operasi PLTU Jeranjang Unit 1. Angka ini TIDAK BOLEH dikutip sebagai "
    "unjuk kerja lapangan. Tujuannya hanya memvalidasi bahwa pipeline "
    "berjalan benar dari ujung ke ujung."
)
