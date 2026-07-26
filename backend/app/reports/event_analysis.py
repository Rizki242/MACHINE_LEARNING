"""Analisis Event Registry PLTU Jeranjang Unit 1 (2016-01 s/d 2026-05).

Seluruh angka pada modul ini berasal dari data operasi nyata, bukan dari
data sintetis. Inilah bagian laporan yang boleh dikutip apa adanya.

Analisis yang dihasilkan
------------------------
* Pareto gangguan per equipment dan per jenis event, berbobot jumlah,
  durasi, dan MWh hilang.
* Tren tahunan, khususnya kelas aglomerasi dan blocking feeder.
* Musiman bulanan, diuji terhadap hipotesis batubara lembab musim hujan.
* Kekambuhan: sebaran waktu antar-event blocking dan indikasi
  pengelompokan.
* Statistik durasi dan pencilan.

Penggunaan
----------
    python -m backend.app.reports.event_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.reports import viz

LOGGER = get_logger(__name__)

SOURCE_NOTE = (
    "Sumber: PARETO Riwayat Gangguan PLTU Jeranjang Unit 1, Jan 2016 - Mei 2026 "
    "(429 event: 302 derating, 127 outage). Data operasi nyata."
)

#: Label ramah untuk jenis event pada grafik.
EVENT_LABELS = {
    "furnace_agglomeration": "Aglomerasi / slagging furnace",
    "coal_feeder_blocking": "Blocking coal feeder",
    "bottom_ash_blocking": "Blocking bottom ash",
    "return_system_disturbance": "Gangguan return material",
    "air_distribution_disturbance": "Gangguan distribusi udara",
    "tube_leak_bed_erosion": "Kebocoran tube erosi bed",
    "coal_quality_derating": "Derating kualitas batubara",
    "coal_supply_constraint": "Pembatasan pasokan batubara",
    "boiler_tube_leak_other": "Kebocoran tube lain",
    "boiler_auxiliary_other": "Peralatan bantu boiler",
    "fan_disturbance": "Gangguan fan udara",
    "startup_normalisation": "Penormalan pasca sinkron",
    "not_furnace_related": "Di luar lingkup furnace",
    "unclassified": "Belum terklasifikasi",
}

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
    "Jul", "Agu", "Sep", "Okt", "Nov", "Des",
]


def _label(event_type: Any) -> str:
    return EVENT_LABELS.get(str(event_type), str(event_type))


# --------------------------------------------------------------------
# Tabel analisis
# --------------------------------------------------------------------
def pareto_table(
    registry: pd.DataFrame, group_column: str, value_column: str | None = None
) -> pd.DataFrame:
    """Tabel Pareto: jumlah atau bobot per kelompok, plus kumulatif persen.

    ``value_column`` ``None`` berarti menghitung jumlah event. Isi dengan
    ``duration_hours`` atau ``mwh_lost`` untuk Pareto berbobot dampak.
    """
    frame = registry.dropna(subset=[group_column]).copy()
    if value_column is None:
        series = frame.groupby(group_column).size()
    else:
        series = frame.groupby(group_column)[value_column].sum(min_count=1)

    table = (
        series.sort_values(ascending=False)
        .rename("value")
        .to_frame()
        .assign(
            share_pct=lambda f: 100.0 * f["value"] / f["value"].sum(),
            cumulative_pct=lambda f: f["share_pct"].cumsum(),
        )
    )
    return table


def yearly_trend(registry: pd.DataFrame, event_types: list[str]) -> pd.DataFrame:
    """Jumlah event per tahun untuk jenis event terpilih."""
    frame = registry.loc[registry["event_type"].isin(event_types)].copy()
    frame["year"] = frame["start_time"].dt.year
    table = (
        frame.pivot_table(
            index="year", columns="event_type", values="event_id", aggfunc="count"
        )
        .reindex(columns=event_types)
        .fillna(0)
        .astype(int)
    )
    full_years = range(
        int(registry["start_time"].dt.year.min()),
        int(registry["start_time"].dt.year.max()) + 1,
    )
    return table.reindex(full_years, fill_value=0)


def monthly_profile(registry: pd.DataFrame, event_types: list[str]) -> pd.DataFrame:
    """Sebaran event per bulan kalender, digabung lintas tahun."""
    frame = registry.loc[registry["event_type"].isin(event_types)].copy()
    frame["month"] = frame["start_time"].dt.month
    table = (
        frame.pivot_table(
            index="month", columns="event_type", values="event_id", aggfunc="count"
        )
        .reindex(index=range(1, 13), columns=event_types)
        .fillna(0)
        .astype(int)
    )
    table.index = MONTH_LABELS
    return table


def recurrence_stats(registry: pd.DataFrame, event_types: list[str]) -> dict[str, Any]:
    """Sebaran selang waktu antar-event dan uji pengelompokan.

    Indeks dispersi memakai koefisien variasi selang waktu. Untuk proses
    Poisson tanpa memori nilainya mendekati 1. Nilai jauh di atas 1
    berarti event menggerombol — indikasi penyebab bersama seperti satu
    kiriman batubara yang buruk, bukan kerusakan acak.
    """
    frame = (
        registry.loc[registry["event_type"].isin(event_types)]
        .sort_values("start_time")
        .copy()
    )
    if len(frame) < 3:
        return {"count": int(len(frame)), "insufficient_data": True}

    gaps = frame["start_time"].diff().dropna().dt.total_seconds() / 86400.0
    gaps = gaps[gaps > 0]
    if gaps.empty:
        return {"count": int(len(frame)), "insufficient_data": True}

    mean_gap = float(gaps.mean())
    std_gap = float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0
    dispersion = std_gap / mean_gap if mean_gap else float("nan")

    # Proporsi event yang menyusul event lain dalam 7 hari.
    clustered = float((gaps <= 7).mean())

    return {
        "count": int(len(frame)),
        "gap_days_mean": mean_gap,
        "gap_days_median": float(gaps.median()),
        "gap_days_p90": float(gaps.quantile(0.90)),
        "gap_days_min": float(gaps.min()),
        "gap_days_max": float(gaps.max()),
        "dispersion_index": dispersion,
        "share_within_7_days": clustered,
        "clustered": bool(dispersion > 1.2),
        "insufficient_data": False,
    }


def duration_stats(registry: pd.DataFrame) -> pd.DataFrame:
    """Statistik durasi per jenis catatan sumber."""
    frame = registry.dropna(subset=["duration_hours"])
    return (
        frame.groupby("record_kind")["duration_hours"]
        .agg(
            count="count",
            min="min",
            median="median",
            mean="mean",
            p90=lambda s: s.quantile(0.90),
            max="max",
        )
        .round(2)
    )


def impact_by_event_type(registry: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan dampak per jenis event: jumlah, jam hilang, MWh hilang."""
    return (
        registry.groupby("event_type")
        .agg(
            events=("event_id", "count"),
            total_hours=("duration_hours", "sum"),
            total_mwh_lost=("mwh_lost", "sum"),
            median_hours=("duration_hours", "median"),
            max_severity=("severity", "max"),
        )
        .sort_values("total_mwh_lost", ascending=False)
        .round(1)
    )


def cause_resolution_pairs(
    registry: pd.DataFrame, event_types: list[str], top_n: int = 12
) -> pd.DataFrame:
    """Pasangan penyebab dan tindakan perbaikan yang paling sering."""
    frame = registry.loc[
        registry["event_type"].isin(event_types)
        & registry["maintenance_action"].notna()
    ].copy()
    frame["action"] = (
        frame["maintenance_action"].astype(str).str.strip().str.casefold().str[:70]
    )
    return (
        frame.groupby("action")
        .agg(
            occurrences=("event_id", "count"),
            median_hours=("duration_hours", "median"),
            total_mwh_lost=("mwh_lost", "sum"),
        )
        .sort_values("occurrences", ascending=False)
        .head(top_n)
        .round(1)
    )


# --------------------------------------------------------------------
# Grafik
# --------------------------------------------------------------------
def figure_pareto_equipment(registry: pd.DataFrame, path: Path) -> Path:
    """Pareto equipment berbobot MWh hilang — bar horizontal, satu hue."""
    import matplotlib.pyplot as plt

    table = pareto_table(registry, "equipment_canonical", "mwh_lost").head(12)
    labels = list(table.index)[::-1]
    values = [float(v) for v in table["value"]][::-1]
    colors = viz.sequential_colors(len(values))

    figure, axes = plt.subplots(figsize=(9.5, 5.6))
    bars = axes.barh(labels, values, color=colors, height=0.62, zorder=3)
    viz.label_bars_h(axes, bars, values, fmt="{:,.0f}")
    viz.style_axes(
        axes,
        title="Boiler menyumbang energi hilang terbesar setelah kejadian selingkup unit",
        subtitle=(
            "Total MWh hilang per kelompok peralatan, Unit 1, Jan 2016 - Mei 2026. "
            "Kelompok UNIT berisi overhaul dan kejadian yang mencakup seluruh unit, "
            "sehingga bukan satu peralatan tertentu."
        ),
        xlabel="MWh hilang (kumulatif 10 tahun)",
        grid_axis="x",
    )
    axes.set_xlim(0, max(values) * 1.16)
    axes.spines["left"].set_color(viz.BASELINE)
    viz.add_source_note(figure, SOURCE_NOTE)
    return viz.save(figure, path)


def figure_pareto_event_type(registry: pd.DataFrame, path: Path) -> Path:
    """Pareto jenis event dalam lingkup furnace, berbobot jumlah kejadian."""
    import matplotlib.pyplot as plt

    settings = get_settings()
    scope = (settings.event_taxonomy.get("blocking_event_types") or []) + (
        settings.event_taxonomy.get("context_event_types") or []
    )
    frame = registry.loc[registry["event_type"].isin(scope)]
    table = pareto_table(frame, "event_type")

    labels = [_label(index) for index in table.index][::-1]
    values = [float(v) for v in table["value"]][::-1]
    colors = viz.sequential_colors(len(values))

    figure, axes = plt.subplots(figsize=(9.5, 4.4))
    bars = axes.barh(labels, values, color=colors, height=0.6, zorder=3)
    viz.label_bars_h(axes, bars, values, fmt="{:,.0f}")
    viz.style_axes(
        axes,
        title="Blocking coal feeder dan aglomerasi furnace mendominasi gangguan bed",
        subtitle=(
            "Jumlah event dalam lingkup risiko blocking furnace, Unit 1, "
            "Jan 2016 - Mei 2026"
        ),
        xlabel="Jumlah event",
        grid_axis="x",
    )
    axes.set_xlim(0, max(values) * 1.16)
    axes.spines["left"].set_color(viz.BASELINE)
    viz.add_source_note(figure, SOURCE_NOTE)
    return viz.save(figure, path)


def figure_yearly_trend(registry: pd.DataFrame, path: Path) -> Path:
    """Tren tahunan tiga kelas gangguan bed — garis, label langsung."""
    import matplotlib.pyplot as plt

    types = [
        "coal_feeder_blocking",
        "furnace_agglomeration",
        "tube_leak_bed_erosion",
    ]
    table = yearly_trend(registry, types)

    figure, axes = plt.subplots(figsize=(9.5, 5.0))
    for index, event_type in enumerate(types):
        colour = viz.CATEGORICAL[index]
        series = table[event_type]
        axes.plot(
            series.index,
            series.to_numpy(),
            color=colour,
            marker="o",
            markersize=5,
            markeredgecolor=viz.SURFACE,
            markeredgewidth=1.5,
            label=_label(event_type),
            zorder=3,
        )
        last_year = series.index[-1]
        axes.annotate(
            _label(event_type),
            xy=(last_year, series.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=viz.TEXT_SECONDARY,
            fontsize=9,
            va="center",
        )

    viz.style_axes(
        axes,
        title="Aglomerasi furnace dan blocking coal feeder naik bersama sejak 2022",
        subtitle=(
            "Jumlah event per tahun, Unit 1. Lonjakan aglomerasi 2020 berdiri "
            "sendiri; tahun 2026 baru terisi sampai Mei."
        ),
        xlabel="Tahun",
        ylabel="Jumlah event",
    )
    viz.integer_axis(axes, "y")
    axes.set_xlim(table.index.min() - 0.4, table.index.max() + 3.4)
    axes.set_xticks(list(table.index))
    # Legend di bawah sumbu: ketiga deret sudah diberi label langsung di
    # ujung garis, sehingga legend di dalam plot hanya akan bertabrakan.
    figure.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncols=3,
        columnspacing=2.0,
        handlelength=1.8,
    )
    viz.add_source_note(figure, SOURCE_NOTE)
    return viz.save(figure, path)


def figure_monthly_profile(registry: pd.DataFrame, path: Path) -> Path:
    """Profil bulanan blocking feeder versus aglomerasi — batang berdampingan."""
    import matplotlib.pyplot as plt
    import numpy as np

    types = ["coal_feeder_blocking", "furnace_agglomeration"]
    table = monthly_profile(registry, types)

    wet = ["Nov", "Des", "Jan", "Feb", "Mar", "Apr"]
    dry = ["Mei", "Jun", "Jul", "Agu", "Sep", "Okt"]
    feeder_wet = int(table.loc[wet, "coal_feeder_blocking"].sum())
    feeder_dry = int(table.loc[dry, "coal_feeder_blocking"].sum())
    agglo_wet = int(table.loc[wet, "furnace_agglomeration"].sum())
    agglo_dry = int(table.loc[dry, "furnace_agglomeration"].sum())

    positions = np.arange(len(table.index))
    width = 0.38
    figure, axes = plt.subplots(figsize=(9.5, 4.8))

    for index, event_type in enumerate(types):
        values = table[event_type].to_numpy(dtype=float)
        offset = (index - 0.5) * (width + 0.02)
        bars = axes.bar(
            positions + offset,
            values,
            width=width,
            color=viz.CATEGORICAL[index],
            label=_label(event_type),
            zorder=3,
        )
        viz.label_bars_v(axes, bars, list(values))

    viz.style_axes(
        axes,
        title="Tidak ada pola musiman pada blocking feeder maupun aglomerasi",
        subtitle=(
            f"Jumlah event per bulan kalender digabung 2016-2026, Unit 1. "
            f"Musim hujan (Nov-Apr) vs kemarau (Mei-Okt): "
            f"feeder {feeder_wet} vs {feeder_dry}, "
            f"aglomerasi {agglo_wet} vs {agglo_dry}. "
            "Hipotesis batubara lembab musiman tidak terdukung data."
        ),
        xlabel="Bulan",
        ylabel="Jumlah event",
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(table.index)
    viz.integer_axis(axes, "y")
    axes.set_ylim(0, float(table.to_numpy().max()) * 1.32)
    axes.legend(loc="upper right", ncols=2, columnspacing=1.6, handlelength=1.4)
    viz.add_source_note(figure, SOURCE_NOTE)
    return viz.save(figure, path)


def figure_recurrence(registry: pd.DataFrame, path: Path) -> Path:
    """Sebaran selang waktu antar-event blocking — histogram satu hue."""
    import matplotlib.pyplot as plt

    settings = get_settings()
    types = settings.event_taxonomy.get("blocking_event_types") or []
    frame = (
        registry.loc[registry["event_type"].isin(types)].sort_values("start_time")
    )
    gaps = frame["start_time"].diff().dropna().dt.total_seconds() / 86400.0
    gaps = gaps[gaps > 0]

    figure, axes = plt.subplots(figsize=(9.5, 4.6))
    bins = np.logspace(np.log10(max(gaps.min(), 0.01)), np.log10(gaps.max()), 26)
    axes.hist(gaps, bins=bins, color=viz.SEQUENTIAL[4], edgecolor=viz.SURFACE,
              linewidth=1.0, zorder=3)
    axes.set_xscale("log")

    median = float(gaps.median())
    dispersion = float(gaps.std(ddof=1) / gaps.mean())
    within_week = float((gaps <= 7).mean())

    axes.axvline(median, color=viz.STATUS["critical"], linewidth=2.0, zorder=4)
    axes.annotate(
        f"median {median:.1f} hari",
        xy=(median, axes.get_ylim()[1] * 0.88),
        xytext=(8, 0),
        textcoords="offset points",
        color=viz.STATUS["critical"],
        fontsize=9.5,
        fontweight="600",
    )

    viz.style_axes(
        axes,
        title="Event blocking datang bergerombol, bukan tersebar merata",
        subtitle=(
            f"Selang waktu antar-event blocking berurutan, skala logaritmik. "
            f"Indeks dispersi {dispersion:.2f} — di atas 1,0 berarti menggerombol; "
            f"proses acak tanpa memori bernilai sekitar 1,0. "
            f"{within_week * 100:.0f} persen event menyusul event sebelumnya "
            "dalam tujuh hari."
        ),
        xlabel="Selang waktu sejak event blocking sebelumnya (hari, skala log)",
        ylabel="Jumlah selang",
    )
    viz.add_source_note(figure, SOURCE_NOTE)
    return viz.save(figure, path)


def figure_severity_by_year(registry: pd.DataFrame, path: Path) -> Path:
    """Komposisi severity per tahun untuk event dalam lingkup furnace."""
    import matplotlib.pyplot as plt
    import numpy as np

    settings = get_settings()
    scope = settings.event_taxonomy.get("blocking_event_types") or []
    frame = registry.loc[registry["event_type"].isin(scope)].copy()
    frame["year"] = frame["start_time"].dt.year
    table = (
        frame.pivot_table(
            index="year", columns="severity", values="event_id", aggfunc="count"
        )
        .fillna(0)
        .astype(int)
        .sort_index()
    )

    severity_labels = {2: "2 - Warning", 3: "3 - Blocking", 4: "4 - Critical"}
    columns = [c for c in (2, 3, 4) if c in table.columns]

    figure, axes = plt.subplots(figsize=(9.5, 4.8))
    bottom = np.zeros(len(table.index), dtype=float)
    for index, severity in enumerate(columns):
        values = table[severity].to_numpy(dtype=float)
        axes.bar(
            table.index,
            values,
            bottom=bottom,
            color=viz.CATEGORICAL[index],
            width=0.66,
            label=severity_labels.get(severity, str(severity)),
            edgecolor=viz.SURFACE,
            linewidth=2.0,
            zorder=3,
        )
        bottom += values

    totals = bottom
    for year, total in zip(table.index, totals):
        if total > 0:
            axes.text(
                year,
                total + max(totals) * 0.025,
                f"{int(total)}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=viz.TEXT_SECONDARY,
            )

    viz.style_axes(
        axes,
        title="Event blocking makin sering, tetapi makin jarang berujung outage",
        subtitle=(
            "Komposisi severity event blocking per tahun (README §15), Unit 1. "
            "Sebelum 2020 hampir semua event berseverity 3-4; sejak 2022 mayoritas "
            "berhenti di severity 2, yaitu derating tanpa unit berhenti. "
            "Tahun 2026 baru terisi sampai Mei."
        ),
        xlabel="Tahun",
        ylabel="Jumlah event",
    )
    axes.set_xticks(list(table.index))
    viz.integer_axis(axes, "y")
    axes.legend(loc="upper left", ncols=3)
    viz.add_source_note(figure, SOURCE_NOTE)
    return viz.save(figure, path)


# --------------------------------------------------------------------
# Alur utama
# --------------------------------------------------------------------
def run_analysis(settings: Settings | None = None) -> dict[str, Any]:
    """Jalankan seluruh analisis, tulis grafik dan ringkasan JSON."""
    settings = settings or get_settings()
    settings.paths.ensure()
    viz.apply_theme()

    from backend.app.data.event_etl import load_registry

    registry = load_registry(settings)
    taxonomy = settings.event_taxonomy
    blocking_types = taxonomy.get("blocking_event_types") or []
    context_types = taxonomy.get("context_event_types") or []

    figures_dir = settings.paths.figures
    figures = {
        "pareto_equipment": figure_pareto_equipment(
            registry, figures_dir / "01_pareto_equipment_mwh.png"
        ),
        "pareto_event_type": figure_pareto_event_type(
            registry, figures_dir / "02_pareto_event_type.png"
        ),
        "yearly_trend": figure_yearly_trend(
            registry, figures_dir / "03_tren_tahunan.png"
        ),
        "monthly_profile": figure_monthly_profile(
            registry, figures_dir / "04_profil_bulanan.png"
        ),
        "recurrence": figure_recurrence(
            registry, figures_dir / "05_kekambuhan.png"
        ),
        "severity_by_year": figure_severity_by_year(
            registry, figures_dir / "06_severity_per_tahun.png"
        ),
    }

    summary: dict[str, Any] = {
        "source": SOURCE_NOTE,
        "total_events": int(len(registry)),
        "by_record_kind": registry["record_kind"].value_counts().to_dict(),
        "date_range": {
            "start": str(registry["start_time"].min()),
            "end": str(registry["start_time"].max()),
        },
        "events_by_type": registry["event_type"].value_counts().to_dict(),
        "blocking_event_count": int(
            registry["event_type"].isin(blocking_types).sum()
        ),
        "duration_stats": duration_stats(registry).to_dict(orient="index"),
        "impact_by_event_type": impact_by_event_type(registry).to_dict(orient="index"),
        "yearly_trend": yearly_trend(
            registry, blocking_types + context_types
        ).to_dict(orient="index"),
        "monthly_profile": monthly_profile(
            registry, blocking_types
        ).to_dict(orient="index"),
        "recurrence_all_blocking": recurrence_stats(registry, blocking_types),
        "recurrence_agglomeration": recurrence_stats(
            registry, ["furnace_agglomeration"]
        ),
        "recurrence_feeder": recurrence_stats(registry, ["coal_feeder_blocking"]),
        "top_actions": cause_resolution_pairs(
            registry, blocking_types
        ).to_dict(orient="index"),
        "figures": {name: str(path) for name, path in figures.items()},
    }

    summary_path = settings.paths.reports / "event_analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analisis Event Registry PLTU Jeranjang Unit 1"
    )
    parser.parse_args(argv)

    settings = get_settings()
    summary = run_analysis(settings)

    print()
    print("=" * 72)
    print("ANALISIS EVENT REGISTRY — PLTU JERANJANG UNIT 1")
    print("=" * 72)
    print(f"Total event      : {summary['total_events']}")
    print(f"Rentang          : {summary['date_range']['start'][:10]} s/d "
          f"{summary['date_range']['end'][:10]}")
    print(f"Event blocking   : {summary['blocking_event_count']}")
    print()

    print("Dampak per jenis event (jam hilang dan MWh hilang):")
    impact = pd.DataFrame(summary["impact_by_event_type"]).T
    print(impact.to_string())
    print()

    print("Kekambuhan event blocking:")
    for name, key in [
        ("Semua blocking", "recurrence_all_blocking"),
        ("Aglomerasi", "recurrence_agglomeration"),
        ("Blocking feeder", "recurrence_feeder"),
    ]:
        stats = summary[key]
        if stats.get("insufficient_data"):
            print(f"  {name:<18}: data tidak cukup ({stats['count']} event)")
            continue
        print(
            f"  {name:<18}: n={stats['count']:>3}  "
            f"median {stats['gap_days_median']:>7.1f} hari  "
            f"indeks dispersi {stats['dispersion_index']:.2f}  "
            f"{'MENGGEROMBOL' if stats['clustered'] else 'menyebar'}  "
            f"({stats['share_within_7_days']*100:.0f}% menyusul dalam 7 hari)"
        )
    print()

    print("Statistik durasi (jam):")
    print(pd.DataFrame(summary["duration_stats"]).T.to_string())
    print()

    print("Grafik:")
    for name, path in summary["figures"].items():
        print(f"  {name:<20}: {path}")
    print()
    print(f"Ringkasan JSON: {summary['summary_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
