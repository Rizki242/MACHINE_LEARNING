"""Baseline operasi per zona beban.

README §10 melarang satu baseline untuk seluruh kondisi operasi, dan
alasannya nyata: bed differential pressure pada 8 MW dan pada 24 MW
berbeda jauh, sehingga satu ambang tunggal akan membanjiri operator
dengan alarm palsu di beban rendah sekaligus buta di beban tinggi.

Baseline dihitung per zona beban (startup, low, medium, high, near rated),
dan **di dalam setiap zona masih dikoreksi terhadap beban**.

Koreksi dalam-zona itu bukan hiasan. Zona medium membentang 10 sampai 17
MW, sementara temperatur bed, tekanan, dan aliran udara bergerak terus
mengikuti beban di sepanjang rentang tersebut. Dengan satu nilai tengah
per zona, unit yang berjalan di tepi bawah zona akan terlihat menyimpang
jauh ke bawah pada hampir semua sinyal sekaligus, dan rule engine akan
menyala karena posisi beban, bukan karena ada yang tidak beres. Itu
kegagalan yang menipu: alarmnya menunjuk peralatan yang sehat.

Karena itu setiap kolom pada setiap zona dimodelkan sebagai garis
terhadap beban, lalu simpangan diukur dari garis itu — bukan dari satu
angka. Sebarannya memakai median absolute deviation atas residu, yang
tahan pencilan; penting karena periode pelatihan sendiri mengandung event
gangguan.

Periode di sekitar event dikeluarkan dari perhitungan baseline. Baseline
harus menggambarkan kondisi normal, bukan rata-rata antara normal dan
rusak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

LOGGER = get_logger(__name__)

#: Faktor pengubah MAD menjadi setara simpangan baku pada sebaran normal.
MAD_TO_SIGMA = 1.4826

#: Batas bawah dispersi agar pembagian tidak meledak pada sinyal sangat tenang.
MIN_DISPERSION = 1e-6


@dataclass
class Baseline:
    """Statistik baseline per zona beban untuk sekumpulan kolom."""

    statistics: dict[str, dict[str, dict[str, float]]]
    columns: list[str]
    zones: list[str]
    excluded_rows: int
    source_rows: int

    def deviation(
        self,
        frame: pd.DataFrame,
        zone_column: str = "load_zone",
        load_column: str = "unit_load_mw",
    ) -> pd.DataFrame:
        """Ubah nilai mentah menjadi simpangan baku robust terhadap baseline.

        Nilai keluaran adalah z-score robust: berapa banyak MAD sebuah
        pembacaan menyimpang dari nilai yang **diharapkan pada beban saat
        itu**, di dalam zona beban yang sedang aktif. Kolom hasil diberi
        akhiran ``_dev``.

        Bila ``load_column`` tidak ada, baseline mundur ke nilai tengah
        zona. Hasilnya masih berguna, tetapi akan membawa kembali cacat
        yang dijelaskan di dokumentasi modul ini: sinyal terlihat
        menyimpang hanya karena unit berada di tepi zona.
        """
        zones = frame[zone_column].astype(str).to_numpy()
        has_load = load_column in frame.columns
        load = (
            frame[load_column].to_numpy(dtype=np.float64)
            if has_load
            else np.zeros(len(frame))
        )
        result = pd.DataFrame(index=frame.index)

        for column in self.columns:
            if column not in frame.columns:
                continue
            values = frame[column].to_numpy(dtype=np.float64)
            expected = np.full(values.shape, np.nan)
            spread = np.full(values.shape, np.nan)

            for zone, stats in self.statistics.items():
                entry = stats.get(column)
                if entry is None:
                    continue
                mask = zones == zone
                if not mask.any():
                    continue
                slope = float(entry.get("slope", 0.0)) if has_load else 0.0
                intercept = float(entry.get("intercept", entry["centre"]))
                expected[mask] = intercept + slope * load[mask]
                spread[mask] = entry["dispersion"]

            spread = np.where(
                np.isfinite(spread) & (spread > MIN_DISPERSION), spread, np.nan
            )
            result[f"{column}_dev"] = (values - expected) / spread

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "zones": self.zones,
            "source_rows": self.source_rows,
            "excluded_rows": self.excluded_rows,
            "statistics": self.statistics,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Baseline:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            statistics=payload["statistics"],
            columns=payload["columns"],
            zones=payload["zones"],
            excluded_rows=payload.get("excluded_rows", 0),
            source_rows=payload.get("source_rows", 0),
        )


def assign_load_zone(
    frame: pd.DataFrame,
    settings: Settings | None = None,
    load_column: str = "unit_load_mw",
) -> pd.Series:
    """Beri label zona beban (README §10) pada setiap baris."""
    settings = settings or get_settings()
    zones = settings.load_zones
    load = frame[load_column].to_numpy(dtype=np.float64)

    # Batas: zona pertama tertutup di kedua ujung, sisanya terbuka di bawah.
    edges = [float(zones[0]["min_mw"])] + [float(zone["max_mw"]) for zone in zones]
    names = [str(zone["name"]) for zone in zones]

    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for position, name in enumerate(names):
        lower, upper = edges[position], edges[position + 1]
        if position == 0:
            mask = (load >= lower) & (load <= upper)
        else:
            mask = (load > lower) & (load <= upper)
        result[mask] = name

    # Beban di atas zona terakhir tetap dianggap zona terakhir.
    result[np.isfinite(load) & (load > edges[-1])] = names[-1]
    return result.rename("load_zone")


def _event_exclusion_mask(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
    settings: Settings,
    timestamp_column: str,
) -> np.ndarray:
    """Tandai baris yang berada di sekitar event agar tidak dipakai baseline."""
    config = settings.model["baseline"]
    before = pd.Timedelta(hours=float(config["exclude_hours_before_event"]))
    after = pd.Timedelta(hours=float(config["exclude_hours_after_event"]))

    stamps = pd.to_datetime(frame[timestamp_column]).to_numpy()
    excluded = np.zeros(len(frame), dtype=bool)

    non_target = set(settings.event_taxonomy.get("non_target_event_types") or [])
    relevant = registry.loc[~registry["event_type"].isin(non_target)]

    for row in relevant.itertuples(index=False):
        if pd.isna(row.start_time):
            continue
        end = row.end_time if pd.notna(row.end_time) else row.start_time
        window_start = np.datetime64(row.start_time - before)
        window_end = np.datetime64(end + after)
        excluded |= (stamps >= window_start) & (stamps <= window_end)

    return excluded


#: Variasi beban minimum di dalam zona sebelum tren layak dipasang, MW.
MIN_LOAD_VARIATION_MW = 0.25


def _fit_load_trend(
    loads: np.ndarray, values: np.ndarray, centre: float
) -> tuple[float, float]:
    """Pasang garis nilai terhadap beban di dalam satu zona.

    Kemiringan dipasang dengan kuadrat terkecil biasa, lalu titik potongnya
    digeser ke median residu agar tetap tahan pencilan — rata-rata akan
    tertarik oleh periode gangguan yang lolos penyaringan.

    Bila beban di dalam zona nyaris tidak bervariasi (mis. zona sempit atau
    data uji sintetis berbeban tetap), tren tidak dipasang dan baseline
    kembali menjadi satu nilai tengah.
    """
    if loads.size < 2 or float(np.ptp(loads)) < MIN_LOAD_VARIATION_MW:
        return 0.0, centre

    load_centre = float(np.mean(loads))
    centred = loads - load_centre
    denominator = float(np.dot(centred, centred))
    if denominator <= MIN_DISPERSION:
        return 0.0, centre

    slope = float(np.dot(centred, values - np.mean(values)) / denominator)
    intercept = float(np.median(values - slope * loads))
    return slope, intercept


def fit_baseline(
    frame: pd.DataFrame,
    columns: list[str],
    settings: Settings | None = None,
    registry: pd.DataFrame | None = None,
    timestamp_column: str = "timestamp",
    zone_column: str = "load_zone",
    load_column: str = "unit_load_mw",
) -> Baseline:
    """Hitung baseline per zona beban dari periode operasi bersih.

    Untuk setiap kolom pada setiap zona dipasang garis terhadap beban,
    sehingga simpangan diukur dari nilai yang diharapkan pada beban saat
    itu — bukan dari satu nilai tengah zona.

    ``registry`` opsional; bila diberikan, jendela di sekitar setiap event
    dikeluarkan supaya baseline benar-benar menggambarkan kondisi normal.
    """
    settings = settings or get_settings()
    config = settings.model["baseline"]
    min_samples = int(config["min_samples_per_zone"])

    working = frame.copy()
    if zone_column not in working.columns:
        working[zone_column] = assign_load_zone(working, settings)

    source_rows = len(working)
    clean = working
    excluded_rows = 0

    if "is_running" in clean.columns:
        clean = clean.loc[clean["is_running"] == 1]

    if registry is not None and timestamp_column in clean.columns:
        mask = _event_exclusion_mask(clean, registry, settings, timestamp_column)
        excluded_rows = int(mask.sum())
        clean = clean.loc[~mask]

    LOGGER.info(
        "Baseline dari %d baris bersih (%d baris dikecualikan karena dekat event)",
        len(clean),
        excluded_rows,
    )

    usable = [column for column in columns if column in clean.columns]
    statistics: dict[str, dict[str, dict[str, float]]] = {}

    for zone, group in clean.groupby(zone_column, observed=True):
        if len(group) < min_samples:
            LOGGER.warning(
                "Zona %s hanya punya %d sampel bersih (minimum %d) — dilewati.",
                zone,
                len(group),
                min_samples,
            )
            continue

        load_values = group[load_column].to_numpy(dtype=np.float64)

        zone_stats: dict[str, dict[str, float]] = {}
        for column in usable:
            raw = group[column].to_numpy(dtype=np.float64)
            valid = np.isfinite(raw) & np.isfinite(load_values)
            values = raw[valid]
            loads = load_values[valid]
            if values.size < min_samples // 4:
                continue

            centre = float(np.median(values))
            slope, intercept = _fit_load_trend(loads, values, centre)
            residual = values - (intercept + slope * loads)
            dispersion = float(
                np.median(np.abs(residual - np.median(residual))) * MAD_TO_SIGMA
            )

            zone_stats[column] = {
                "centre": centre,
                "intercept": intercept,
                "slope": slope,
                "dispersion": max(dispersion, MIN_DISPERSION),
                "samples": int(values.size),
            }
        statistics[str(zone)] = zone_stats

    return Baseline(
        statistics=statistics,
        columns=usable,
        zones=sorted(statistics),
        excluded_rows=excluded_rows,
        source_rows=source_rows,
    )
