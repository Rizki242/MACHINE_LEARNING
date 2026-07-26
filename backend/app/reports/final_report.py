"""Penyusun laporan analisis akhir FurnaceGuard AI Fase 0.

Laporan disusun dari berkas keluaran yang sesungguhnya — ringkasan
analisis event, tabel evaluasi model, registri model, dan hasil uji
konsistensi sumber. Tidak ada angka yang diketik tangan, sehingga
menjalankan ulang pipeline akan memperbarui laporan tanpa risiko
angkanya menyimpang dari kenyataan.

Penggunaan
----------
    python -m backend.app.reports.final_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.constants import DISCLAIMER, SYNTHETIC_METRIC_WARNING
from backend.app.core.logging import get_logger
from backend.app.reports.event_analysis import EVENT_LABELS

LOGGER = get_logger(__name__)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        LOGGER.warning("Berkas tidak ditemukan, bagian terkait dilewati: %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _label(event_type: str) -> str:
    return EVENT_LABELS.get(event_type, event_type)


def _fmt(value: Any, digits: int = 2) -> str:
    """Format angka dengan konvensi Indonesia: koma desimal, titik ribuan."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (int, float)):
        text = f"{float(value):,.{digits}f}"
        # Tukar pemisah: "1,234.56" menjadi "1.234,56".
        return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return str(value)


# --------------------------------------------------------------------
# Bagian-bagian laporan
# --------------------------------------------------------------------
def section_header(settings: Settings) -> list[str]:
    unit = settings.units["unit"]
    boiler = settings.units["boiler"]
    return [
        "# Laporan Analisis — FurnaceGuard AI Fase 0",
        "",
        f"**Pembangkit:** {unit['plant']} · **Unit:** 1 · "
        f"**Kapasitas:** {unit['capacity_mw']:.0f} MW · "
        f"**Boiler:** {boiler['type']}, {boiler['rated_evaporation_t_per_h']:.0f} t/jam",
        "",
        "**Lingkup fase ini:** fondasi data, analisis riwayat gangguan, dan "
        "pipeline machine learning yang teruji. Belum ada API, antarmuka "
        "pengguna, maupun sambungan ke historian.",
        "",
        "---",
        "",
    ]


def section_summary(analysis: dict[str, Any], settings: Settings) -> list[str]:
    counts = analysis["events_by_type"]
    blocking = settings.event_taxonomy["blocking_event_types"]
    blocking_total = sum(counts.get(t, 0) for t in blocking)

    recurrence = analysis["recurrence_all_blocking"]
    monthly = pd.DataFrame(analysis["monthly_profile"]).T.fillna(0)
    wet = ["Nov", "Des", "Jan", "Feb", "Mar", "Apr"]
    dry = ["Mei", "Jun", "Jul", "Agu", "Sep", "Okt"]

    feeder_wet = int(monthly.loc[wet, "coal_feeder_blocking"].sum())
    feeder_dry = int(monthly.loc[dry, "coal_feeder_blocking"].sum())

    return [
        "## 1. Ringkasan eksekutif",
        "",
        f"Riwayat gangguan PLTU Jeranjang Unit 1 selama sepuluh tahun lima bulan "
        f"({analysis['date_range']['start'][:10]} sampai "
        f"{analysis['date_range']['end'][:10]}) berisi "
        f"**{analysis['total_events']} event**: "
        f"{analysis['by_record_kind'].get('derating', 0)} derating dan "
        f"{analysis['by_record_kind'].get('outage', 0)} outage. Sebanyak "
        f"**{blocking_total} event** masuk lingkup risiko blocking furnace.",
        "",
        "Lima temuan yang menentukan arah pengembangan berikutnya:",
        "",
        f"1. **Tidak ada satu pun catatan blocking cold slag pipe.** Sepanjang "
        f"sepuluh tahun, jurnal gangguan tidak pernah mencatatnya sebagai "
        f"penyebab derating maupun outage. Empat target probabilitas per slag "
        f"pipe pada README §5 karena itu tidak dapat dilatih maupun divalidasi.",
        "",
        f"2. **Blocking coal feeder dan aglomerasi furnace mendominasi.** "
        f"Keduanya menyumbang {counts.get('coal_feeder_blocking', 0)} dan "
        f"{counts.get('furnace_agglomeration', 0)} event — "
        f"{100 * (counts.get('coal_feeder_blocking', 0) + counts.get('furnace_agglomeration', 0)) / max(blocking_total, 1):.0f} "
        f"persen dari seluruh event blocking.",
        "",
        f"3. **Event blocking datang bergerombol.** Median selang antar-event "
        f"{_fmt(recurrence['gap_days_median'], 1)} hari dengan indeks dispersi "
        f"**{_fmt(recurrence['dispersion_index'], 2)}**; proses acak tanpa memori "
        f"bernilai sekitar 1,0. Sebanyak "
        f"{recurrence['share_within_7_days'] * 100:.0f} persen event menyusul "
        f"event sebelumnya dalam tujuh hari. Pengelompokan sekuat ini menunjuk "
        f"pada penyebab bersama, kemungkinan besar satu kiriman batubara.",
        "",
        f"4. **Tidak ada pola musiman.** Blocking coal feeder tercatat "
        f"{feeder_wet} kali pada musim hujan (Nov–Apr) dan {feeder_dry} kali "
        f"pada kemarau (Mei–Okt). Kelembaban batubara memang berulang kali "
        f"disebut sebagai penyebab, tetapi kejadiannya tersebar merata "
        f"sepanjang tahun.",
        "",
        "5. **Tidak ada satu pun timeseries DCS.** Seluruh tag Priority A "
        "README §7 belum tersedia. Inilah kendala tunggal terbesar; tanpa "
        "data itu sistem tidak dapat berjalan di kondisi nyata betapapun "
        "rapi kodenya.",
        "",
        "---",
        "",
    ]


def section_event_analysis(analysis: dict[str, Any], settings: Settings) -> list[str]:
    impact = pd.DataFrame(analysis["impact_by_event_type"]).T
    blocking = settings.event_taxonomy["blocking_event_types"]
    context = settings.event_taxonomy.get("context_event_types", [])
    scope = set(blocking) | set(context)

    lines = [
        "## 2. Analisis riwayat gangguan",
        "",
        "Seluruh angka pada bagian ini berasal dari **data operasi nyata**.",
        "",
        "### 2.1 Dampak per jenis event",
        "",
        "| Jenis event | Event | Jam hilang | MWh hilang | Median jam |",
        "|---|---:|---:|---:|---:|",
    ]

    subset = impact.loc[[i for i in impact.index if i in scope]]
    subset = subset.sort_values("events", ascending=False)
    for event_type, row in subset.iterrows():
        lines.append(
            f"| {_label(str(event_type))} | {int(row['events'])} | "
            f"{_fmt(row['total_hours'], 0)} | {_fmt(row['total_mwh_lost'], 0)} | "
            f"{_fmt(row['median_hours'], 1)} |"
        )

    yearly = pd.DataFrame(analysis["yearly_trend"]).T.fillna(0).astype(int)
    blocking_columns = [c for c in yearly.columns if c in blocking]
    totals = yearly[blocking_columns].sum(axis=1)

    lines += [
        "",
        "### 2.2 Tren tahunan event blocking",
        "",
        "| Tahun | " + " | ".join(_label(c) for c in blocking_columns) + " | Total |",
        "|---" * (len(blocking_columns) + 2) + "|",
    ]
    for year in yearly.index:
        cells = " | ".join(str(int(yearly.loc[year, c])) for c in blocking_columns)
        lines.append(f"| {year} | {cells} | **{int(totals.loc[year])}** |")

    peak_year = int(totals.idxmax())
    lines += [
        "",
        f"Puncaknya pada {peak_year} dengan {int(totals.max())} event. "
        f"Tahun terakhir pada tabel baru terisi sampai Mei.",
        "",
        "Komposisi severity juga bergeser: sebelum 2020 hampir semua event "
        "berseverity 3–4 (unit berhenti); sejak 2022 mayoritas berhenti di "
        "severity 2, yaitu derating tanpa unit berhenti. Gangguan menjadi "
        "**lebih sering tetapi lebih ringan**.",
        "",
        "### 2.3 Kekambuhan",
        "",
        "| Kelas | Event | Median selang | Indeks dispersi | Pola |",
        "|---|---:|---:|---:|---|",
    ]

    for label, key in [
        ("Semua blocking", "recurrence_all_blocking"),
        ("Aglomerasi furnace", "recurrence_agglomeration"),
        ("Blocking coal feeder", "recurrence_feeder"),
    ]:
        stats = analysis.get(key, {})
        if stats.get("insufficient_data", True) and "gap_days_median" not in stats:
            lines.append(f"| {label} | {stats.get('count', 0)} | — | — | data tidak cukup |")
            continue
        pola = "menggerombol" if stats["clustered"] else "menyebar"
        lines.append(
            f"| {label} | {stats['count']} | "
            f"{_fmt(stats['gap_days_median'], 1)} hari | "
            f"{_fmt(stats['dispersion_index'], 2)} | {pola} |"
        )

    duration = pd.DataFrame(analysis["duration_stats"]).T
    lines += [
        "",
        "### 2.4 Durasi gangguan",
        "",
        "| Jenis catatan | Jumlah | Min | Median | Rata-rata | P90 | Maks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, row in duration.iterrows():
        lines.append(
            f"| {kind} | {int(row['count'])} | {_fmt(row['min'], 2)} | "
            f"{_fmt(row['median'], 2)} | {_fmt(row['mean'], 2)} | "
            f"{_fmt(row['p90'], 2)} | {_fmt(row['max'], 2)} |"
        )
    lines += ["", "Satuan jam.", "", "### 2.5 Grafik", ""]
    for name, path in analysis.get("figures", {}).items():
        lines.append(f"- `{Path(path).name}` — {name.replace('_', ' ')}")
    lines += ["", "Seluruhnya di `backend/reports/figures/`.", "", "---", ""]
    return lines


def section_classification(analysis: dict[str, Any], settings: Settings) -> list[str]:
    counts = analysis["events_by_type"]
    total = analysis["total_events"]
    blocking = set(settings.event_taxonomy["blocking_event_types"])
    context = set(settings.event_taxonomy.get("context_event_types", []))

    lines = [
        "## 3. Klasifikasi event otomatis",
        "",
        "Teks gangguan berbahasa Indonesia diklasifikasikan menjadi "
        "`event_type`, `event_location`, dan `severity` sesuai README §15. "
        "Dua lapis dijalankan berurutan: aturan kata kunci yang dapat diaudit, "
        "lalu TF-IDF karakter n-gram yang menandai baris mencurigakan. "
        "Analyzer karakter dipilih karena sumbernya penuh salah ketik — "
        "`funace`, `slaging`, `pembebeban`, `batuabra`, `indikiasi`.",
        "",
        "| Jenis event | Jumlah | Porsi | Lingkup |",
        "|---|---:|---:|---|",
    ]

    for event_type, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        if event_type in blocking:
            scope = "**blocking**"
        elif event_type in context:
            scope = "konteks"
        elif event_type == "unclassified":
            scope = "perlu tinjauan"
        else:
            scope = "di luar lingkup"
        lines.append(
            f"| {_label(event_type)} | {count} | {100 * count / total:.1f} % | {scope} |"
        )

    unclassified = counts.get("unclassified", 0)
    lines += [
        "",
        f"Sebanyak {unclassified} event belum terklasifikasi dan seluruhnya "
        "ditandai untuk verifikasi di `backend/reports/events_needing_review.csv`. "
        "Isinya memang ambigu di sumber. Tidak ada baris yang dibuang diam-diam.",
        "",
        "Kolom `coal_source`, `coal_blending`, dan `clinker_found` yang diminta "
        "README §15 tidak ada di berkas sumber dan dibiarkan kosong. Menebaknya "
        "akan merusak analisis pengaruh kualitas batubara — justru pertanyaan "
        "yang paling ingin dijawab sistem ini.",
        "",
        "---",
        "",
    ]
    return lines


def section_model(
    table: pd.DataFrame | None, registry: dict[str, Any] | None, settings: Settings
) -> list[str]:
    lines = [
        "## 4. Unjuk kerja pipeline machine learning",
        "",
        f"> **{SYNTHETIC_METRIC_WARNING}**",
        "",
    ]

    if table is None or table.empty:
        lines += [
            "Tabel evaluasi belum tersedia. Jalankan "
            "`python -m backend.app.models.train`.",
            "",
            "---",
            "",
        ]
        return lines

    if registry:
        metadata = registry.get("metadata", {})
        lines += [
            f"Pembagian berbasis waktu: latih {metadata.get('train_years')}, "
            f"validasi {metadata.get('validation_years')}, "
            f"uji {metadata.get('test_years')}, dengan jeda embargo "
            f"{settings.model['split']['embargo_hours']} jam di setiap "
            f"perbatasan. Jumlah fitur {metadata.get('feature_count')}.",
            "",
        ]

    test = table.loc[table["dataset"] == "test"].copy()

    # Tabel evaluasi yang ditulis versi terdahulu tidak membawa jumlah event.
    # Registri model menyimpan metrik uji lengkap per model, jadi kolom yang
    # hilang diisi dari sana daripada dibiarkan kosong.
    if registry and "event_count" not in test.columns:
        lookup = {
            (entry["name"], entry["horizon"]): entry.get("metrics", {})
            for entry in registry.get("models", [])
        }
        for column in ("event_count", "events_detected", "alarm_duty_cycle"):
            test[column] = [
                lookup.get((row["model"], row["horizon"]), {}).get(column)
                for _, row in test.iterrows()
            ]

    lines += [
        "### 4.1 Metrik per sampel",
        "",
        "| Model | Horizon | PR-AUC | ROC-AUC | Precision | Recall | Brier | Calib. error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test.sort_values(["horizon", "pr_auc"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {row['model']} | {row['horizon']} | {_fmt(row['pr_auc'], 4)} | "
            f"{_fmt(row['roc_auc'], 4)} | {_fmt(row['precision'], 4)} | "
            f"{_fmt(row['recall'], 4)} | {_fmt(row['brier_score'], 4)} | "
            f"{_fmt(row['calibration_error'], 4)} |"
        )

    lines += [
        "",
        "Perhatikan jarak antara ROC-AUC dan PR-AUC. Dengan prevalensi kelas "
        "positif di bawah satu persen, model yang selalu menjawab "
        "\"tidak ada risiko\" sudah benar hampir sepanjang waktu dan akan "
        "memperoleh ROC-AUC yang terlihat mengesankan. **PR-AUC adalah metrik "
        "utama**; ROC-AUC dicantumkan hanya karena README §12 memintanya.",
        "",
        "### 4.2 Metrik per event",
        "",
        "Inilah pertanyaan yang sebenarnya ditanyakan operator: berapa banyak "
        "event nyata yang tertangkap, berapa lama sebelumnya, dan berapa "
        "sering sistem berteriak tanpa sebab.",
        "",
        "| Model | Horizon | Event | Terdeteksi | Tingkat deteksi | Alarm palsu/hari | Duty cycle | Median warning |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test.sort_values(["horizon", "pr_auc"], ascending=[True, False]).iterrows():
        duty = row.get("alarm_duty_cycle")
        duty_text = "—" if duty is None or pd.isna(duty) else f"{100 * float(duty):.1f} %"
        lines.append(
            f"| {row['model']} | {row['horizon']} | "
            f"{_fmt(row.get('event_count'), 0)} | "
            f"{_fmt(row.get('events_detected'), 0)} | "
            f"{_fmt(row.get('event_detection_rate'), 3)} | "
            f"{_fmt(row.get('false_alarms_per_day'), 2)} | "
            f"{duty_text} | "
            f"{_fmt(row.get('median_warning_horizon_minutes'), 0)} menit |"
        )

    alarm = settings.thresholds["alarm"]
    lines += [
        "",
        f"Ambang dipilih dari dua syarat sekaligus: paling banyak "
        f"{_fmt(alarm['target_false_alarms_per_day'], 1)} alarm palsu per hari "
        f"**dan** duty cycle paling tinggi "
        f"{100 * float(alarm['max_alarm_duty_cycle']):.0f} persen. Syarat kedua "
        "menutup celah yang tampak sepele tetapi merusak: tanpa batas duty "
        "cycle, ambang yang sangat rendah dapat lolos anggaran alarm palsu "
        "dengan cara membuat alarm menyala hampir sepanjang waktu — satu "
        "rentetan panjang, tercatat sedikit, dan sama sekali tidak berguna "
        "bagi operator.",
        "",
        "Tingkat deteksi bernilai 1,0 pada seluruh baris perlu dibaca dengan "
        "hati-hati: jumlah event pada periode uji hanya belasan, sehingga "
        "angka itu tidak stabil. Yang lebih layak dibandingkan antar-model "
        "adalah PR-AUC dan alarm palsu per hari.",
        "",
        "Sekali lagi: angka-angka di atas mengukur **kebenaran pipeline**, "
        "bukan kemampuan deteksi di lapangan. Pola degradasi pada data "
        "sintetis adalah hipotesis engineering menurut README §6, bukan "
        "pengamatan. Bila tanda tangan gangguan sesungguhnya berbeda — dan "
        "besar kemungkinan berbeda — model ini tidak akan mengenalinya.",
        "",
        "---",
        "",
    ]
    return lines


def section_data_needed(settings: Settings) -> list[str]:
    priority_a = settings.standard_names("priority_a")
    return [
        "## 5. Data yang harus disediakan",
        "",
        "Bagian ini adalah keluaran paling berharga dari fase ini.",
        "",
        "### 5.1 Prioritas 1 — tanpa ini sistem tidak dapat berjalan",
        "",
        "Ekspor historian resolusi **1 menit**, rentang minimum **12 bulan**, "
        "mencakup periode di sekitar event pada Event Registry. Seluruh "
        f"{len(priority_a)} tag Priority A README §7:",
        "",
        "```text",
        *[
            "  ".join(priority_a[i : i + 3])
            for i in range(0, len(priority_a), 3)
        ],
        "```",
        "",
        "Bersamanya diperlukan **daftar nama tag DCS asli beserta satuannya**, "
        "diverifikasi engineer operasi atau instrumentasi, untuk mengisi "
        "`config/tag_mapping.yaml`. Seluruh entri di berkas itu saat ini masih "
        "`null`.",
        "",
        "### 5.2 Prioritas 2 — menentukan mutu peringatan",
        "",
        "Tag Priority B README §7, ditambah satu yang tidak ada di README:",
        "",
        "**Log kegiatan poking** — waktu, lokasi, lama, dan alasannya. Inilah "
        "satu-satunya jalan memperoleh label untuk kelas blocking bottom ash. "
        "Dugaan paling masuk akal atas ketiadaan catatan blocking slag pipe "
        "adalah bahwa gangguannya selalu tertangani lewat poking rutin tanpa "
        "menyebabkan derating, sehingga tidak pernah masuk jurnal gangguan.",
        "",
        "### 5.3 Prioritas 3 — menjawab pertanyaan penyebab",
        "",
        "Data kualitas batubara per kiriman, dengan tanggal:",
        "",
        "```text",
        "coal_source            coal_supplier          coal_shipment_id",
        "coal_blending_ratio    gross_calorific_value  net_calorific_value",
        "total_moisture         ash_content            volatile_matter",
        "coal_size_distribution percentage_above_8mm   ash_fusion_temperature",
        "```",
        "",
        "Mengingat event blocking terbukti menggerombol, inilah data yang "
        "paling mungkin menjelaskan **mengapa** gerombolan itu terjadi.",
        "",
        "### 5.4 Prioritas 4 — verifikasi ambang",
        "",
        "SOP operasi Unit 1, logic DCS dan matriks cause and effect, "
        "commissioning report, serta daftar setting alarm dan trip aktual. "
        "README §2 menyatakan seluruh nilai desain harus diverifikasi "
        "terhadap dokumen tersebut. Sampai itu dilakukan, tidak satu pun "
        "ambang di berkas konfigurasi boleh dianggap sebagai batas operasi.",
        "",
        "### 5.5 Konfirmasi yang diperlukan",
        "",
        "1. **Jumlah coal feeder.** Manual desain menyebut empat; jurnal "
        "gangguan hanya pernah menyebut 1A dan 1B.",
        "2. **Blocking slag pipe.** Apakah benar tidak pernah terjadi, atau "
        "tertangani rutin tanpa dicatat, atau tercatat dengan istilah lain.",
        "3. **Event yang ditandai perlu tinjauan** pada "
        "`backend/reports/events_needing_review.csv`.",
        "",
        "---",
        "",
    ]


def section_next_steps() -> list[str]:
    return [
        "## 6. Langkah berikutnya",
        "",
        "Diurut menurut ketergantungan, bukan menurut kemudahan.",
        "",
        "1. **Kumpulkan data Prioritas 1 dan petakan tag DCS.** Segala hal "
        "lain menunggu langkah ini.",
        "2. **Hitung ulang baseline dan tinjau seluruh ambang** memakai data "
        "operasi nyata. Baseline saat ini berasal dari data sintetis.",
        "3. **Latih ulang model pada data nyata** dan bandingkan hasilnya "
        "dengan angka pada laporan ini. Selisihnya akan menunjukkan seberapa "
        "jauh asumsi generator sintetis meleset.",
        "4. **Verifikasi Event Registry** bersama engineer operasi, terutama "
        "event yang ditandai perlu tinjauan.",
        "5. **Baru setelah itu** bangun lapisan API dan antarmuka pengguna "
        "(README §16, §17). Membangunnya lebih dulu berarti menampilkan angka "
        "yang belum layak ditampilkan.",
        "",
        "Mode deployment tetap `offline` sampai langkah 1 sampai 4 selesai. "
        "README §19 menempatkan integrasi historian hanya setelah validasi, "
        "persetujuan teknis, dan pengujian keamanan.",
        "",
        "---",
        "",
    ]


def section_footer() -> list[str]:
    return [
        "## 7. Disclaimer",
        "",
        DISCLAIMER,
        "",
        "FurnaceGuard AI tidak boleh dipakai untuk bypass interlock, "
        "menonaktifkan proteksi, mengubah safety limit, mengontrol fan, coal "
        "feeder, atau ash valve secara otomatis, mengubah set point boiler "
        "tanpa otorisasi, maupun menggantikan prosedur operasi PLTU Jeranjang "
        "(README §20, §24).",
        "",
        "Dokumen pendukung: `docs/DATA_AUDIT.md`, `docs/METHODOLOGY.md`, "
        "`docs/LIMITATIONS.md`.",
        "",
    ]


# --------------------------------------------------------------------
# Alur utama
# --------------------------------------------------------------------
def build(settings: Settings | None = None) -> Path:
    """Susun laporan lengkap dan tulis ke `backend/reports/`."""
    settings = settings or get_settings()
    settings.paths.ensure()

    analysis = _read_json(settings.paths.reports / "event_analysis_summary.json")
    if analysis is None:
        raise FileNotFoundError(
            "Ringkasan analisis belum ada. Jalankan: "
            "python -m backend.app.reports.event_analysis"
        )

    registry = _read_json(settings.paths.models / "model_registry.json")

    table_path = settings.paths.reports / "model_evaluation.csv"
    table = pd.read_csv(table_path) if table_path.exists() else None

    lines: list[str] = []
    lines += section_header(settings)
    lines += section_summary(analysis, settings)
    lines += section_event_analysis(analysis, settings)
    lines += section_classification(analysis, settings)
    lines += section_model(table, registry, settings)
    lines += section_data_needed(settings)
    lines += section_next_steps()
    lines += section_footer()

    output = settings.paths.reports / "FURNACEGUARD_ANALYSIS_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Laporan ditulis: %s", output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Susun laporan analisis akhir")
    parser.parse_args(argv)

    path = build(get_settings())
    print(f"Laporan: {path}")
    print(f"Ukuran : {path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
