"""Rekayasa fitur untuk deteksi dini blocking furnace.

Seluruh fitur pada README §9 diimplementasikan di sini, dengan formula
persis seperti tertulis di sana.

ATURAN YANG TIDAK BOLEH DILANGGAR
---------------------------------
Setiap jendela rolling HANYA melihat ke belakang. Fitur pada waktu ``t``
tidak boleh menyentuh satu pun sampel setelah ``t``.

Ini bukan kerapian gaya. Kebocoran masa depan menghasilkan model yang
tampak sangat akurat saat pengujian dan gagal total saat dipasang, karena
saat berjalan sungguhan masa depan itu belum ada. Kegagalannya senyap:
tidak ada pesan galat, hanya metrik bagus yang berbohong.

``backend/tests/test_features.py`` menguji sifat ini secara langsung
dengan sinyal berundak.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

LOGGER = get_logger(__name__)

#: Nilai pembagi minimum agar rasio tidak meledak saat penyebut mendekati nol.
EPSILON = 1e-6

#: Ambang aliran bottom ash yang dianggap "tidak keluar", t/jam.
ASH_FLOW_MINIMUM = 0.30


# --------------------------------------------------------------------
# Primitif rolling — semuanya melihat ke belakang
# --------------------------------------------------------------------
def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Rata-rata bergulir ke belakang."""
    return series.rolling(window, min_periods=max(2, window // 4)).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Simpangan baku bergulir ke belakang."""
    return series.rolling(window, min_periods=max(2, window // 4)).std()


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Kemiringan regresi linear pada jendela ke belakang, satuan per sampel.

    Dihitung tertutup, bukan lewat ``rolling().apply()``. Untuk jendela
    selebar ``W`` dengan jarak sampel seragam, kemiringan kuadrat terkecil
    dapat disusun dari dua jumlah bergulir saja, sehingga biayanya linear
    terhadap panjang deret alih-alih perkalian panjang deret dengan lebar
    jendela.

    Rongga data diisi maju lebih dulu dengan batas selebar jendela; tanpa
    itu, jumlah bergulir melewati nilai hilang dan pasangan indeks-nilai
    bergeser sehingga kemiringannya salah.
    """
    if window < 3:
        raise ValueError("Jendela kemiringan minimal 3 sampel.")

    filled = series.ffill(limit=window)
    values = filled.to_numpy(dtype=np.float64)
    position = np.arange(values.size, dtype=np.float64)

    weighted = pd.Series(position * values, index=series.index)
    plain = pd.Series(values, index=series.index)

    min_periods = max(3, window // 2)
    sum_weighted = weighted.rolling(window, min_periods=min_periods).sum()
    sum_plain = plain.rolling(window, min_periods=min_periods).sum()

    # Indeks awal setiap jendela, dipakai memindahkan asal koordinat.
    window_start = pd.Series(position - window + 1, index=series.index)
    mean_local_t = (window - 1) / 2.0

    numerator = sum_weighted - (window_start + mean_local_t) * sum_plain
    denominator = window * (window * window - 1) / 12.0

    slope = numerator / denominator
    # Baris tempat nilai aslinya hilang tidak boleh membawa kemiringan.
    slope[series.isna()] = np.nan
    return slope


def rate_of_change(series: pd.Series, periods: int) -> pd.Series:
    """Perubahan nilai terhadap ``periods`` sampel sebelumnya."""
    return series.diff(periods)


# --------------------------------------------------------------------
# Fitur turunan spesifik boiler (README §9)
# --------------------------------------------------------------------
def bed_temperature_spread(frame: pd.DataFrame) -> pd.Series:
    """Selisih temperatur bed tertinggi dan terendah antar zona.

    README §9:
        main_aux_temp_spread =
        max(main_bed_temp, front_aux_bed_temp, rear_aux_bed_temp)
        - min(main_bed_temp, front_aux_bed_temp, rear_aux_bed_temp)
    """
    columns = [
        "main_bed_temperature",
        "front_aux_bed_temperature",
        "rear_aux_bed_temperature",
    ]
    available = [column for column in columns if column in frame.columns]
    if len(available) < 2:
        return pd.Series(np.nan, index=frame.index)
    block = frame[available]
    return block.max(axis=1) - block.min(axis=1)


def main_aux_temperature_spread(frame: pd.DataFrame) -> pd.Series:
    """Selisih main bed terhadap rata-rata kedua auxiliary bed.

    Berbeda dari :func:`bed_temperature_spread`, fitur ini bertanda:
    positif berarti main bed lebih panas daripada auxiliary bed, yang
    merupakan pola gangguan sirkulasi material.
    """
    needed = {
        "main_bed_temperature",
        "front_aux_bed_temperature",
        "rear_aux_bed_temperature",
    }
    if not needed.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index)
    aux_mean = (
        frame["front_aux_bed_temperature"] + frame["rear_aux_bed_temperature"]
    ) / 2.0
    return frame["main_bed_temperature"] - aux_mean


def air_resistance(pressure: pd.Series, flow: pd.Series) -> pd.Series:
    """Resistansi udara bed.

    README §9:
        main_bed_air_resistance = main_bed_air_pressure / main_bed_air_flow

    Ini indikator inti aglomerasi: ketika bed mulai menggumpal, tekanan
    yang dibutuhkan untuk mendorong udara yang sama akan naik.
    """
    return pressure / flow.clip(lower=EPSILON)


def feeder_columns(frame: pd.DataFrame, suffix: str) -> list[str]:
    """Kumpulkan kolom coal feeder dengan akhiran tertentu, terurut."""
    return sorted(
        column
        for column in frame.columns
        if column.startswith("coal_feeder_") and column.endswith(suffix)
    )


def feeder_imbalance(frame: pd.DataFrame) -> pd.Series:
    """Ketidakseimbangan aliran antar coal feeder.

    README §9:
        coal_feeder_imbalance =
        (max_feeder_flow - min_feeder_flow) / average_feeder_flow
    """
    columns = feeder_columns(frame, "_flow")
    if len(columns) < 2:
        return pd.Series(np.nan, index=frame.index)
    block = frame[columns]
    average = block.mean(axis=1).clip(lower=EPSILON)
    return (block.max(axis=1) - block.min(axis=1)) / average


def feeder_command_deviation(frame: pd.DataFrame) -> pd.Series:
    """Selisih relatif terbesar antara perintah dan aliran nyata feeder.

    Inilah tanda paling langsung dari blocking feeder: perintah naik,
    batubara tidak ikut naik.
    """
    flows = feeder_columns(frame, "_flow")
    commands = feeder_columns(frame, "_command")
    if not flows or len(flows) != len(commands):
        return pd.Series(np.nan, index=frame.index)

    gaps = []
    for flow_column, command_column in zip(flows, commands):
        command = frame[command_column].clip(lower=EPSILON)
        gaps.append((command - frame[flow_column]) / command)
    return pd.concat(gaps, axis=1).max(axis=1)


def air_to_fuel_ratio(frame: pd.DataFrame) -> pd.Series:
    """Rasio total udara pembakaran terhadap aliran batubara."""
    air_columns = [
        column
        for column in ("main_bed_air_flow", "auxiliary_bed_air_flow", "secondary_air_flow")
        if column in frame.columns
    ]
    if not air_columns or "coal_flow_total" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    total_air = frame[air_columns].sum(axis=1)
    return total_air / frame["coal_flow_total"].clip(lower=EPSILON)


def combustion_instability_index(frame: pd.DataFrame, window: int) -> pd.Series:
    """Indeks ketidakstabilan pembakaran.

    Menggabungkan tiga gejala yang selalu muncul bersama saat pembakaran
    goyah: CO berfluktuasi, O2 berfluktuasi, dan draft furnace bergejolak.
    Masing-masing dinormalkan terhadap tingkatnya sendiri agar tidak ada
    satu suku yang mendominasi hanya karena satuannya besar.
    """
    terms: list[pd.Series] = []

    if "carbon_monoxide_co" in frame.columns:
        series = frame["carbon_monoxide_co"]
        level = rolling_mean(series, window).clip(lower=1.0)
        terms.append(rolling_std(series, window) / level)

    if "oxygen_o2" in frame.columns:
        series = frame["oxygen_o2"]
        level = rolling_mean(series, window).clip(lower=0.5)
        terms.append(rolling_std(series, window) / level)

    if "furnace_pressure" in frame.columns:
        series = frame["furnace_pressure"]
        terms.append(rolling_std(series, window) / 100.0)

    if not terms:
        return pd.Series(np.nan, index=frame.index)
    return pd.concat(terms, axis=1).mean(axis=1)


def slag_discharge_failure(frame: pd.DataFrame) -> pd.Series:
    """Katup slag terbuka tetapi abu tidak keluar.

    README §9:
        slag_discharge_failure = slag_valve_open AND ash_flow_below_minimum
    """
    if "bottom_ash_discharge_status" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    valve_open = frame["bottom_ash_discharge_status"].astype("float") > 0.5

    if "bottom_ash_flow" in frame.columns:
        below_minimum = frame["bottom_ash_flow"] < ASH_FLOW_MINIMUM
    else:
        return pd.Series(np.nan, index=frame.index)

    return (valve_open & below_minimum).astype(np.float32)


def return_system_disturbance(frame: pd.DataFrame, window: int) -> pd.Series:
    """Indeks gangguan sistem sirkulasi material kembali.

    Menggabungkan penurunan tekanan udara U-valve dan penyimpangan
    temperatur return leg terhadap tingkat terkininya sendiri.
    """
    terms: list[pd.Series] = []

    if "u_valve_air_pressure" in frame.columns:
        series = frame["u_valve_air_pressure"]
        level = rolling_mean(series, window).clip(lower=EPSILON)
        terms.append((level - series) / level)

    if "return_leg_temperature" in frame.columns:
        series = frame["return_leg_temperature"]
        level = rolling_mean(series, window)
        spread = rolling_std(series, window).clip(lower=0.5)
        terms.append(((series - level) / spread).abs() / 4.0)

    if "separator_differential_pressure" in frame.columns:
        series = frame["separator_differential_pressure"]
        level = rolling_mean(series, window).clip(lower=EPSILON)
        terms.append(((level - series) / level).abs())

    if not terms:
        return pd.Series(np.nan, index=frame.index)
    return pd.concat(terms, axis=1).mean(axis=1)


def aux_air_flow_imbalance(frame: pd.DataFrame) -> pd.Series:
    """Ketidakseimbangan aliran udara antara front dan rear auxiliary bed."""
    needed = {"front_aux_bed_air_flow", "rear_aux_bed_air_flow"}
    if not needed.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index)
    front = frame["front_aux_bed_air_flow"]
    rear = frame["rear_aux_bed_air_flow"]
    total = (front + rear).clip(lower=EPSILON)
    return (front - rear).abs() / total


def bed_pressure_imbalance(frame: pd.DataFrame) -> pd.Series:
    """Ketidakseimbangan tekanan main bed terhadap auxiliary bed."""
    needed = {"main_bed_pressure", "aux_bed_pressure"}
    if not needed.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index)
    total = (frame["main_bed_pressure"] + frame["aux_bed_pressure"]).clip(lower=EPSILON)
    return (frame["main_bed_pressure"] - frame["aux_bed_pressure"]) / total


# --------------------------------------------------------------------
# Perakit utama
# --------------------------------------------------------------------
def build_features(
    frame: pd.DataFrame,
    settings: Settings | None = None,
    baseline: "object | None" = None,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """Bangun seluruh matriks fitur README §9 dari satu potongan timeseries.

    Masukan harus terurut waktu menaik dan berjarak seragam. Bila
    ``baseline`` diberikan (objek :class:`~backend.app.features.baseline.Baseline`),
    kolom ``*_dev`` ikut dihasilkan sebagai simpangan robust terhadap
    baseline zona beban yang sedang aktif.

    Catatan tentang batas potongan: fitur dihitung per potongan yang
    diberikan. Bila data dipotong per tahun, sekitar 60 menit pertama tiap
    potongan punya riwayat tidak lengkap dan sebagian fitur rolling akan
    kosong di sana. Itu disengaja — mengarang riwayat lebih buruk daripada
    mengakui ketiadaannya.
    """
    settings = settings or get_settings()
    config = settings.model["features"]

    if not config.get("backward_only", True):
        raise ValueError(
            "model_config.yaml: features.backward_only harus true. "
            "Jendela yang melihat ke depan membocorkan masa depan ke fitur."
        )

    result = pd.DataFrame(index=frame.index)
    if "timestamp" in frame.columns:
        result["timestamp"] = frame["timestamp"]

    # ---------------- Fitur seketika ----------------
    result["bed_temp_spread"] = bed_temperature_spread(frame)
    result["main_aux_temp_spread"] = main_aux_temperature_spread(frame)
    result["bed_pressure_imbalance"] = bed_pressure_imbalance(frame)
    result["coal_feeder_imbalance"] = feeder_imbalance(frame)
    result["coal_feeder_command_deviation"] = feeder_command_deviation(frame)
    result["aux_air_flow_imbalance"] = aux_air_flow_imbalance(frame)
    result["air_to_fuel_ratio"] = air_to_fuel_ratio(frame)
    result["slag_discharge_failure"] = slag_discharge_failure(frame)

    if {"primary_air_pressure", "main_bed_air_flow"}.issubset(frame.columns):
        result["main_bed_air_resistance"] = air_resistance(
            frame["primary_air_pressure"], frame["main_bed_air_flow"]
        )
    if {"aux_bed_pressure", "auxiliary_bed_air_flow"}.issubset(frame.columns):
        result["aux_bed_air_resistance"] = air_resistance(
            frame["aux_bed_pressure"], frame["auxiliary_bed_air_flow"]
        )

    if {"main_steam_flow", "unit_load_mw"}.issubset(frame.columns):
        result["steam_flow_per_mw"] = frame["main_steam_flow"] / frame[
            "unit_load_mw"
        ].clip(lower=0.5)
    if {"coal_flow_total", "unit_load_mw"}.issubset(frame.columns):
        result["coal_flow_per_mw"] = frame["coal_flow_total"] / frame[
            "unit_load_mw"
        ].clip(lower=0.5)

    feeder_current_columns = feeder_columns(frame, "_current")
    if feeder_current_columns:
        currents = frame[feeder_current_columns]
        result["coal_feeder_current_max"] = currents.max(axis=1)
        result["coal_feeder_current_spread"] = currents.max(axis=1) - currents.min(axis=1)

    if not include_rolling:
        return result

    # ---------------- Fitur bergulir ----------------
    windows = [int(w) for w in config["rolling_windows_minutes"]]
    std_windows = [int(w) for w in config["rolling_std_windows_minutes"]]
    slope_windows = [int(w) for w in config["slope_windows_minutes"]]
    roc_periods = int(config["rate_of_change_minutes"])
    base_columns = [c for c in config["rolling_base_columns"] if c in frame.columns]

    LOGGER.info(
        "Fitur bergulir: %d kolom dasar, jendela %s menit",
        len(base_columns),
        windows,
    )

    # Operasi rolling pandas menaikkan float32 menjadi float64. Dengan lebih
    # dari dua ratus kolom dan setengah juta baris per tahun, selisihnya
    # ratusan megabyte per potongan — cukup untuk kehabisan memori. Presisi
    # float32 jauh melampaui presisi sensor lapangan, jadi tiap fitur
    # dikembalikan ke float32 begitu selesai dihitung.
    pieces: dict[str, pd.Series] = {}

    def store(name: str, series: pd.Series) -> None:
        pieces[name] = series.astype(np.float32)

    for column in base_columns:
        series = frame[column]
        for window in windows:
            store(f"{column}_rolling_mean_{window}m", rolling_mean(series, window))
        for window in std_windows:
            store(f"{column}_rolling_std_{window}m", rolling_std(series, window))
        for window in slope_windows:
            store(f"{column}_slope_{window}m", rolling_slope(series, window))
        store(f"{column}_roc_{roc_periods}m", rate_of_change(series, roc_periods))

    # Fitur bergulir atas fitur turunan yang paling menentukan.
    derived_targets = [
        "bed_temp_spread",
        "main_aux_temp_spread",
        "main_bed_air_resistance",
        "aux_bed_air_resistance",
        "coal_feeder_imbalance",
        "coal_feeder_command_deviation",
        "air_to_fuel_ratio",
    ]
    for column in derived_targets:
        if column not in result.columns:
            continue
        series = result[column]
        store(f"{column}_rolling_mean_15m", rolling_mean(series, 15))
        store(f"{column}_slope_30m", rolling_slope(series, 30))

    # Kemiringan arus ash cooler diminta eksplisit oleh README §9.
    if "ash_cooler_motor_current" in frame.columns:
        store(
            "ash_cooler_current_slope",
            rolling_slope(frame["ash_cooler_motor_current"], 30),
        )

    window_15 = 15
    store("combustion_instability_index", combustion_instability_index(frame, window_15))
    store("return_system_disturbance", return_system_disturbance(frame, window_15))

    result = pd.concat([result, pd.DataFrame(pieces, index=frame.index)], axis=1)
    del pieces

    # ---------------- Simpangan terhadap baseline ----------------
    if baseline is not None:
        from backend.app.features.baseline import assign_load_zone

        zone = (
            frame["load_zone"]
            if "load_zone" in frame.columns
            else assign_load_zone(frame, settings)
        )
        working = frame.copy()
        working["load_zone"] = zone
        deviations = baseline.deviation(working)

        # Fitur turunan juga dinilai terhadap baseline zonanya sendiri.
        derived_for_baseline = [
            column
            for column in BASELINE_DERIVED_COLUMNS
            if column in result.columns and column in baseline.columns
        ]
        if derived_for_baseline:
            derived_frame = result[derived_for_baseline].copy()
            derived_frame["load_zone"] = zone
            # Kolom beban ikut dibawa: baseline mengoreksi terhadap beban di
            # dalam zona, jadi tanpa kolom ini simpangannya kembali diukur
            # dari satu nilai tengah zona.
            if "unit_load_mw" in frame.columns:
                derived_frame["unit_load_mw"] = frame["unit_load_mw"]
            deviations = pd.concat(
                [deviations, baseline.deviation(derived_frame)], axis=1
            )

        deviations = deviations.loc[:, ~deviations.columns.duplicated()]
        result = pd.concat([result, deviations], axis=1)

        magnitude = deviations.abs()
        result["deviation_from_baseline"] = magnitude.mean(axis=1)
        result["deviation_from_baseline_max"] = magnitude.max(axis=1)
        result["load_zone"] = zone

    for column in result.columns:
        if result[column].dtype == np.float64:
            result[column] = result[column].astype(np.float32)

    return result


#: Fitur turunan yang ikut diberi baseline per zona beban.
#: ``thresholds.yaml`` merujuk versi ``_dev`` dari kolom-kolom ini, jadi
#: bila baseline tidak mencakupnya, aturan yang bersangkutan kehilangan
#: sebagian syaratnya tanpa suara.
BASELINE_DERIVED_COLUMNS = [
    "main_bed_air_resistance",
    "aux_bed_air_resistance",
    "bed_temp_spread",
    "air_to_fuel_ratio",
    "coal_feeder_current_max",
]


def derived_baseline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Hitung fitur turunan yang perlu punya baseline sendiri.

    Dipisahkan dari :func:`build_features` supaya pemasangan baseline tidak
    perlu membangun seluruh matriks fitur — cukup kolom yang benar-benar
    dirujuk oleh aturan.
    """
    result = pd.DataFrame(index=frame.index)

    if {"primary_air_pressure", "main_bed_air_flow"}.issubset(frame.columns):
        result["main_bed_air_resistance"] = air_resistance(
            frame["primary_air_pressure"], frame["main_bed_air_flow"]
        )
    if {"aux_bed_pressure", "auxiliary_bed_air_flow"}.issubset(frame.columns):
        result["aux_bed_air_resistance"] = air_resistance(
            frame["aux_bed_pressure"], frame["auxiliary_bed_air_flow"]
        )

    result["bed_temp_spread"] = bed_temperature_spread(frame)
    result["air_to_fuel_ratio"] = air_to_fuel_ratio(frame)

    currents = feeder_columns(frame, "_current")
    if currents:
        result["coal_feeder_current_max"] = frame[currents].max(axis=1)

    return result.dropna(axis=1, how="all")


def feature_columns(frame: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Daftar kolom numerik yang layak dipakai sebagai masukan model."""
    blocked = set(exclude) | {
        "timestamp",
        "load_zone",
        "synthetic_seed",
        "data_source",
    }
    return [
        column
        for column in frame.columns
        if column not in blocked
        and frame[column].dtype.kind in "fiu"
        and not column.startswith("ramp_")
        and not column.startswith("blocking_next_")
    ]
