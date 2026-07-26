"""Generator timeseries sintetis untuk PLTU Jeranjang Unit 1.

Mengapa modul ini ada
---------------------
Tidak ada satu pun tag timeseries DCS pada data yang tersedia. Seluruh
tag Priority A README §7 masih ``null`` di ``config/tag_mapping.yaml``.
Tanpa timeseries, rekayasa fitur dan model risiko tidak dapat dilatih
maupun diuji sama sekali.

Modul ini membangkitkan timeseries yang meniru bentuk data historian:
nama kolomnya persis ``standard_name`` pada ``tag_mapping.yaml``,
sehingga saat tag DCS asli diisi, pipeline hilir tidak berubah sedikit
pun — hanya sumbernya yang ditukar.

Dasar fisik
-----------
Nilai dasar diturunkan dari spesifikasi desain boiler (README §2-§4):
kapasitas 130 t/jam, steam 9,81 MPa / 540 °C, air pengisi 215 °C, nilai
kalor batubara 3.611 kcal/kg, efisiensi boiler desain 90 % dengan hasil
uji April 2026 sebesar 80,39 %.

Realisme dari data nyata
------------------------
Timeline mengikuti Event Registry hasil ETL: unit benar-benar berhenti
selama event outage, beban benar-benar turun selama event derating, dan
setiap event blocking didahului pola degradasi sesuai tanda tangan
gangguannya menurut README §6.

BATAS PEMAKAIAN
---------------
Data ini SINTETIS. Setiap baris membawa kolom ``data_source='synthetic'``.
Metrik model yang dihitung di atasnya TIDAK BOLEH dikutip sebagai unjuk
kerja lapangan. Tujuannya membuktikan pipeline berjalan benar dari ujung
ke ujung, bukan mengukur kemampuan deteksi sesungguhnya.

Penggunaan
----------
    python -m backend.app.data.synthetic
    python -m backend.app.data.synthetic --years 2022-2026 --freq 1min
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.constants import (
    DATA_SOURCE_COLUMN,
    DATA_SOURCE_SYNTHETIC,
    RECORD_KIND_OUTAGE,
)
from backend.app.core.logging import get_logger

LOGGER = get_logger(__name__)

# --------------------------------------------------------------------
# Tetapan termodinamika untuk neraca energi kasar
# --------------------------------------------------------------------
#: Entalpi steam 9,81 MPa(g) / 540 °C, kJ/kg.
STEAM_ENTHALPY_KJ_PER_KG = 3474.0
#: Entalpi air pengisi 215 °C, kJ/kg.
FEEDWATER_ENTHALPY_KJ_PER_KG = 921.0
#: 1 kcal = 4,1868 kJ.
KJ_PER_KCAL = 4.1868

#: Tanda tangan degradasi per jenis event (README §6).
#: Nilai adalah amplitudo relatif pada puncak ramp, sebelum event mulai.
#: Positif menaikkan, negatif menurunkan.
DEGRADATION_SIGNATURES: dict[str, dict[str, float]] = {
    "furnace_agglomeration": {
        "bed_differential_pressure": 0.35,
        "main_bed_pressure": 0.22,
        "primary_air_pressure": 0.20,
        "main_bed_air_flow": -0.16,
        "bed_temp_spread_boost": 55.0,
        "furnace_pressure_noise": 2.4,
        "carbon_monoxide_co": 0.45,
        "ash_cooler_motor_current": 0.12,
    },
    "coal_feeder_blocking": {
        "coal_feeder_command_gap": 0.30,
        "coal_feeder_current_boost": 0.28,
        "coal_feeder_imbalance_boost": 0.35,
        "oxygen_o2": 0.28,
        "carbon_monoxide_co": 0.55,
        "main_bed_temperature": -0.035,
        "bed_temp_spread_boost": 30.0,
    },
    "bottom_ash_blocking": {
        "ash_cooler_motor_current": 0.40,
        "bottom_ash_flow": -0.65,
        "main_bed_pressure": 0.25,
        "bed_differential_pressure": 0.20,
        "slag_pipe_temperature": -0.10,
        "slag_discharge_stall": 1.0,
    },
    "return_system_disturbance": {
        "return_leg_temperature": -0.09,
        "u_valve_air_pressure": -0.30,
        "return_leg_pressure": 0.22,
        "separator_differential_pressure": -0.25,
        "rear_aux_bed_temperature": -0.05,
    },
    "air_distribution_disturbance": {
        "primary_air_pressure": 0.26,
        "main_bed_air_flow": -0.20,
        "auxiliary_bed_air_flow": -0.18,
        "aux_air_imbalance_boost": 0.30,
        "bed_temp_spread_boost": 45.0,
        "furnace_pressure_noise": 1.8,
    },
    "tube_leak_bed_erosion": {
        "furnace_pressure_noise": 1.5,
        "oxygen_o2": -0.10,
        "main_steam_flow": -0.06,
        "bed_differential_pressure": 0.10,
    },
    "coal_quality_derating": {
        "coal_flow_total": 0.14,
        "main_steam_pressure": -0.05,
        "carbon_monoxide_co": 0.30,
        "main_bed_temperature": -0.02,
    },
}

#: Lama ramp degradasi sebelum event, menit. Diacak dalam rentang ini.
RAMP_MINUTES_RANGE = (30, 240)


@dataclass
class GenerationSpec:
    """Parameter satu kali pembangkitan."""

    start_year: int
    end_year: int
    freq: str
    seed: int
    noise_scale: float = 1.0
    gap_probability: float = 0.0015
    stuck_probability: float = 0.0008


# --------------------------------------------------------------------
# Pembantu deret waktu
# --------------------------------------------------------------------
def _ar1(rng: np.random.Generator, size: int, phi: float, sigma: float) -> np.ndarray:
    """Derau berkorelasi waktu (proses AR(1)).

    Derau putih membuat setiap sinyal proses terlihat palsu dan membuat
    fitur rolling std tidak berarti. Proses AR(1) menghasilkan gerakan
    yang menempel antar-langkah seperti pengukuran sungguhan.
    """
    noise = rng.normal(0.0, sigma, size)
    output = np.empty(size, dtype=np.float64)
    output[0] = noise[0]
    for index in range(1, size):
        output[index] = phi * output[index - 1] + noise[index]
    return output


def _smooth_step(length: int) -> np.ndarray:
    """Ramp 0 -> 1 berbentuk S, tanpa sudut tajam."""
    if length <= 1:
        return np.ones(max(length, 1))
    position = np.linspace(0.0, 1.0, length)
    return position * position * (3.0 - 2.0 * position)


def _daily_load_profile(
    index: pd.DatetimeIndex, rng: np.random.Generator, capacity: float
) -> np.ndarray:
    """Profil beban harian yang masuk akal untuk unit 25 MW.

    Beban dasar tinggi sepanjang hari dengan penurunan dini hari, ditambah
    hanyutan lambat antar-hari agar tidak berulang identik.
    """
    # Diubah ke ndarray sejak awal: aritmetika pada DatetimeIndex
    # menghasilkan Index, yang tidak mendukung penambahan sumbu.
    minutes = (index.hour * 60 + index.minute).to_numpy(dtype=np.float64)
    day_fraction = minutes / 1440.0

    # Puncak sore, lembah dini hari.
    shape = (
        0.86
        + 0.10 * np.sin(2 * np.pi * (day_fraction - 0.22))
        + 0.04 * np.sin(4 * np.pi * (day_fraction - 0.10))
    )

    # Hanyutan lambat berskala hari.
    day_number = index.dayofyear.to_numpy() + index.year.to_numpy() * 366
    unique_days, inverse = np.unique(day_number, return_inverse=True)
    daily_offset = rng.normal(0.0, 0.045, unique_days.size)
    daily_offset = np.convolve(daily_offset, np.ones(5) / 5.0, mode="same")

    # Sebagian hari dijalankan pada beban rendah, entah karena permintaan
    # jaringan atau pembatasan bahan bakar. Tanpa ini, zona beban rendah
    # dan menengah nyaris tidak pernah terisi sehingga baselinenya tidak
    # bisa dihitung sama sekali (README §10 mensyaratkan baseline per zona).
    regime = np.ones(unique_days.size)
    low_days = rng.random(unique_days.size) < 0.10
    regime[low_days] = rng.uniform(0.42, 0.70, int(low_days.sum()))
    medium_days = (~low_days) & (rng.random(unique_days.size) < 0.12)
    regime[medium_days] = rng.uniform(0.62, 0.82, int(medium_days.sum()))
    # Transisi antar-hari dihaluskan agar beban tidak melompat di tengah malam.
    regime = np.convolve(regime, np.ones(3) / 3.0, mode="same")
    regime[0] = regime[min(1, regime.size - 1)]

    profile = (shape + daily_offset[inverse]) * regime[inverse]
    profile = np.clip(profile, 0.30, 1.0)
    return profile * capacity


# --------------------------------------------------------------------
# Timeline dari Event Registry nyata
# --------------------------------------------------------------------
def build_event_timeline(
    registry: pd.DataFrame,
    index: pd.DatetimeIndex,
    settings: Settings,
) -> dict[str, np.ndarray]:
    """Susun penanda waktu dari event nyata ke atas indeks waktu sintetis.

    Menghasilkan:
      ``outage``     — unit berhenti (beban nol).
      ``derating``   — pembatasan beban, besarnya dari kolom derating_mw.
      ``ramp_<tipe>``— intensitas 0..1 selama jendela degradasi sebelum
                       event blocking dengan jenis tersebut.
      ``event_start``— penanda menit dimulainya event blocking.
    """
    size = len(index)
    timeline: dict[str, np.ndarray] = {
        "outage": np.zeros(size, dtype=np.float32),
        "derating": np.zeros(size, dtype=np.float32),
        "event_start": np.zeros(size, dtype=np.float32),
    }
    for event_type in DEGRADATION_SIGNATURES:
        timeline[f"ramp_{event_type}"] = np.zeros(size, dtype=np.float32)

    blocking_types = set(settings.event_taxonomy.get("blocking_event_types") or [])
    context_types = set(settings.event_taxonomy.get("context_event_types") or [])
    relevant = blocking_types | context_types

    rng = np.random.default_rng(settings.synthetic_seed + 991)
    window_start, window_end = index[0], index[-1]
    capacity = float(settings.units["unit"]["capacity_mw"])

    for row in registry.itertuples(index=False):
        start = row.start_time
        end = row.end_time if pd.notna(row.end_time) else start
        if pd.isna(start) or end < window_start or start > window_end:
            continue

        left = int(index.searchsorted(max(start, window_start), side="left"))
        right = int(index.searchsorted(min(end, window_end), side="right"))

        if row.record_kind == RECORD_KIND_OUTAGE:
            timeline["outage"][left:right] = 1.0
        elif right > left:
            derating = row.derating_mw
            fraction = (
                float(derating) / capacity
                if derating is not None and pd.notna(derating)
                else 0.30
            )
            timeline["derating"][left:right] = np.clip(fraction, 0.0, 0.95)

        event_type = str(row.event_type)
        if event_type not in relevant or event_type not in DEGRADATION_SIGNATURES:
            continue
        if left >= size:
            continue

        timeline["event_start"][min(left, size - 1)] = 1.0

        ramp_minutes = int(rng.integers(*RAMP_MINUTES_RANGE))
        ramp_start = max(0, left - ramp_minutes)
        length = left - ramp_start
        if length > 1:
            timeline[f"ramp_{event_type}"][ramp_start:left] = np.maximum(
                timeline[f"ramp_{event_type}"][ramp_start:left],
                _smooth_step(length).astype(np.float32),
            )
        # Degradasi tidak hilang seketika saat event mulai; ia mereda
        # selama event berlangsung sampai tindakan perbaikan selesai.
        if right > left:
            decay = np.linspace(1.0, 0.25, right - left, dtype=np.float32)
            timeline[f"ramp_{event_type}"][left:right] = np.maximum(
                timeline[f"ramp_{event_type}"][left:right], decay
            )

    return timeline


# --------------------------------------------------------------------
# Pembangkitan satu potongan waktu
# --------------------------------------------------------------------
def generate_chunk(
    index: pd.DatetimeIndex,
    registry: pd.DataFrame,
    settings: Settings,
    spec: GenerationSpec,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Bangkitkan satu potongan timeseries beserta seluruh tag Priority A/B."""
    size = len(index)
    units = settings.units
    boiler = units["boiler"]
    fuel = units["design_fuel"]
    capacity = float(units["unit"]["capacity_mw"])
    rated_steam = float(boiler["rated_evaporation_t_per_h"])
    feeder_count = int(units["furnace"]["coal_feeder_count"])

    timeline = build_event_timeline(registry, index, settings)
    scale = spec.noise_scale

    def ramp(event_type: str) -> np.ndarray:
        return timeline.get(f"ramp_{event_type}", np.zeros(size, dtype=np.float32))

    agglomeration = ramp("furnace_agglomeration")
    feeder_block = ramp("coal_feeder_blocking")
    ash_block = ramp("bottom_ash_blocking")
    return_disturb = ramp("return_system_disturbance")
    air_disturb = ramp("air_distribution_disturbance")
    tube_leak = ramp("tube_leak_bed_erosion")
    fuel_quality = ramp("coal_quality_derating")

    signature = DEGRADATION_SIGNATURES

    # ---------------- Beban dan steam ----------------
    load = _daily_load_profile(index, rng, capacity)
    load *= 1.0 - timeline["derating"]
    load *= 1.0 - timeline["outage"]
    load += _ar1(rng, size, 0.985, 0.055 * scale)
    load = np.clip(load, 0.0, capacity * 1.04)

    running = load > 0.4
    load_fraction = load / capacity

    steam_flow = load_fraction * rated_steam
    steam_flow *= 1.0 + signature["tube_leak_bed_erosion"]["main_steam_flow"] * tube_leak
    steam_flow += _ar1(rng, size, 0.97, 0.30 * scale)
    steam_flow = np.clip(steam_flow, 0.0, rated_steam * 1.10)

    steam_pressure = float(boiler["superheated_steam_pressure_mpag"]) * (
        0.55 + 0.45 * load_fraction
    )
    steam_pressure *= (
        1.0 + signature["coal_quality_derating"]["main_steam_pressure"] * fuel_quality
    )
    steam_pressure += _ar1(rng, size, 0.98, 0.030 * scale)

    steam_temperature = float(boiler["superheated_steam_temperature_c"]) - 60.0 * (
        1.0 - load_fraction
    ) ** 2
    steam_temperature += _ar1(rng, size, 0.98, 1.4 * scale)

    # ---------------- Batubara dari neraca energi ----------------
    duty_kj_per_h = (
        steam_flow
        * 1000.0
        * (STEAM_ENTHALPY_KJ_PER_KG - FEEDWATER_ENTHALPY_KJ_PER_KG)
    )
    ncv_kj_per_kg = float(fuel["net_calorific_value_kcal_per_kg"]) * KJ_PER_KCAL
    efficiency = float(boiler["measured_efficiency_pct"]) / 100.0
    coal_flow = duty_kj_per_h / (ncv_kj_per_kg * efficiency) / 1000.0
    # Kualitas batubara buruk menuntut aliran lebih besar untuk beban sama.
    coal_flow *= (
        1.0 + signature["coal_quality_derating"]["coal_flow_total"] * fuel_quality
    )
    coal_flow += _ar1(rng, size, 0.96, 0.07 * scale)
    coal_flow = np.clip(coal_flow, 0.0, 30.0)

    # ---------------- Coal feeder per unit ----------------
    # Unit ini punya empat feeder menurut desain. Jurnal gangguan hanya
    # pernah menyebut 1A dan 1B, sehingga blocking selalu disuntikkan ke
    # feeder pertama agar konsisten dengan catatan nyata.
    base_share = np.full((size, feeder_count), 1.0 / feeder_count)
    # Amplitudo hanyutan dijaga kecil: ketidakseimbangan feeder saat normal
    # harus berada jauh di bawah ketidakseimbangan saat blocking, kalau
    # tidak fitur `coal_feeder_imbalance` kehilangan daya pisahnya.
    drift = np.stack(
        [_ar1(rng, size, 0.995, 0.0012 * scale) for _ in range(feeder_count)], axis=1
    )
    shares = base_share + drift

    imbalance_boost = signature["coal_feeder_blocking"]["coal_feeder_imbalance_boost"]
    shares[:, 0] *= 1.0 - imbalance_boost * feeder_block
    shares[:, 1] *= 1.0 + 0.5 * imbalance_boost * feeder_block
    shares = np.clip(shares, 0.02, None)
    shares /= shares.sum(axis=1, keepdims=True)

    feeder_flow = shares * coal_flow[:, None]
    command_gap = signature["coal_feeder_blocking"]["coal_feeder_command_gap"]
    feeder_command = feeder_flow.copy()
    # Perintah tetap, aliran nyata yang tertinggal — inilah tanda blocking.
    feeder_command[:, 0] = feeder_flow[:, 0] / np.clip(
        1.0 - command_gap * feeder_block, 0.35, 1.0
    )

    current_boost = signature["coal_feeder_blocking"]["coal_feeder_current_boost"]
    feeder_current = 6.0 + 2.6 * feeder_flow
    feeder_current[:, 0] *= 1.0 + current_boost * feeder_block
    feeder_current += rng.normal(0.0, 0.10 * scale, feeder_current.shape)

    # ---------------- Temperatur bed ----------------
    bed_reference = units["nominal_operating_reference"]["bed_temperature_c"]
    bed_base = bed_reference[0] + (bed_reference[1] - bed_reference[0]) * load_fraction
    bed_base = np.where(running, bed_base, 120.0 + 60.0 * load_fraction)

    spread = (
        signature["furnace_agglomeration"]["bed_temp_spread_boost"] * agglomeration
        + signature["coal_feeder_blocking"]["bed_temp_spread_boost"] * feeder_block
        + signature["air_distribution_disturbance"]["bed_temp_spread_boost"]
        * air_disturb
    )

    main_bed_temperature = (
        bed_base
        * (1.0 + signature["coal_feeder_blocking"]["main_bed_temperature"] * feeder_block)
        * (1.0 + signature["coal_quality_derating"]["main_bed_temperature"] * fuel_quality)
        + 0.45 * spread
        + _ar1(rng, size, 0.99, 1.6 * scale)
    )
    # Auxiliary bed lebih dingin: bed-nya lebih rendah 500 mm dan udaranya
    # terpisah, sehingga pembakarannya kurang intens (README §3).
    front_aux_bed_temperature = (
        main_bed_temperature - 18.0 - 0.55 * spread + _ar1(rng, size, 0.99, 1.9 * scale)
    )
    rear_aux_bed_temperature = (
        main_bed_temperature
        - 22.0
        - 0.50 * spread
        + signature["return_system_disturbance"]["rear_aux_bed_temperature"]
        * return_disturb
        * bed_base
        + _ar1(rng, size, 0.99, 2.0 * scale)
    )

    # ---------------- Udara dan tekanan bed ----------------
    main_bed_air_flow = 52000.0 * (0.42 + 0.58 * load_fraction)
    main_bed_air_flow *= (
        1.0
        + signature["furnace_agglomeration"]["main_bed_air_flow"] * agglomeration
        + signature["air_distribution_disturbance"]["main_bed_air_flow"] * air_disturb
    )
    main_bed_air_flow += _ar1(rng, size, 0.97, 260.0 * scale)
    main_bed_air_flow = np.clip(main_bed_air_flow, 0.0, None)

    aux_total = 24000.0 * (0.40 + 0.60 * load_fraction)
    aux_total *= (
        1.0 + signature["air_distribution_disturbance"]["auxiliary_bed_air_flow"] * air_disturb
    )
    aux_imbalance = (
        signature["air_distribution_disturbance"]["aux_air_imbalance_boost"] * air_disturb
    )
    front_aux_air_flow = aux_total * (0.5 - 0.5 * aux_imbalance)
    rear_aux_air_flow = aux_total * (0.5 + 0.5 * aux_imbalance)
    front_aux_air_flow += _ar1(rng, size, 0.97, 130.0 * scale)
    rear_aux_air_flow += _ar1(rng, size, 0.97, 130.0 * scale)
    auxiliary_bed_air_flow = front_aux_air_flow + rear_aux_air_flow

    # Tekanan header naik ketika bed makin sukar difluidisasi.
    primary_air_pressure = 12.0 * (0.55 + 0.45 * load_fraction)
    primary_air_pressure *= (
        1.0
        + signature["furnace_agglomeration"]["primary_air_pressure"] * agglomeration
        + signature["air_distribution_disturbance"]["primary_air_pressure"] * air_disturb
    )
    primary_air_pressure += _ar1(rng, size, 0.98, 0.09 * scale)

    dp_reference = units["nominal_operating_reference"]["bed_differential_pressure_kpa"]
    bed_differential_pressure = dp_reference[0] + (
        dp_reference[1] - dp_reference[0]
    ) * load_fraction
    bed_differential_pressure *= (
        1.0
        + signature["furnace_agglomeration"]["bed_differential_pressure"] * agglomeration
        + signature["bottom_ash_blocking"]["bed_differential_pressure"] * ash_block
        + signature["tube_leak_bed_erosion"]["bed_differential_pressure"] * tube_leak
    )
    bed_differential_pressure += _ar1(rng, size, 0.98, 0.10 * scale)
    bed_differential_pressure = np.clip(bed_differential_pressure, 0.0, None)

    main_bed_pressure = bed_differential_pressure * 1.15
    main_bed_pressure *= (
        1.0
        + signature["furnace_agglomeration"]["main_bed_pressure"] * agglomeration
        + signature["bottom_ash_blocking"]["main_bed_pressure"] * ash_block
    )
    aux_bed_pressure = bed_differential_pressure * 0.72 + _ar1(
        rng, size, 0.97, 0.09 * scale
    )

    secondary_air_flow = 38000.0 * (0.35 + 0.65 * load_fraction) + _ar1(
        rng, size, 0.97, 220.0 * scale
    )

    # ---------------- Draft furnace ----------------
    furnace_noise_gain = (
        1.0
        + signature["furnace_agglomeration"]["furnace_pressure_noise"] * agglomeration
        + signature["air_distribution_disturbance"]["furnace_pressure_noise"] * air_disturb
        + signature["tube_leak_bed_erosion"]["furnace_pressure_noise"] * tube_leak
    )
    furnace_pressure = -120.0 * load_fraction + _ar1(
        rng, size, 0.90, 22.0 * scale
    ) * furnace_noise_gain

    # ---------------- Pembakaran ----------------
    oxygen = 3.2 + 3.4 * (1.0 - load_fraction)
    oxygen *= (
        1.0
        + signature["coal_feeder_blocking"]["oxygen_o2"] * feeder_block
        + signature["tube_leak_bed_erosion"]["oxygen_o2"] * tube_leak
    )
    oxygen += _ar1(rng, size, 0.97, 0.10 * scale)
    oxygen = np.clip(oxygen, 0.2, 21.0)

    carbon_monoxide = 70.0 + 180.0 * np.clip(3.6 - oxygen, 0.0, None)
    carbon_monoxide *= (
        1.0
        + signature["furnace_agglomeration"]["carbon_monoxide_co"] * agglomeration
        + signature["coal_feeder_blocking"]["carbon_monoxide_co"] * feeder_block
        + signature["coal_quality_derating"]["carbon_monoxide_co"] * fuel_quality
    )
    carbon_monoxide *= np.exp(_ar1(rng, size, 0.95, 0.16 * scale))
    carbon_monoxide = np.clip(carbon_monoxide, 0.0, 5000.0)

    id_fan_current = 120.0 * (0.35 + 0.65 * load_fraction) + _ar1(
        rng, size, 0.98, 1.1 * scale
    )

    # ---------------- Sistem abu ----------------
    # Pembuangan bottom ash berjalan berkala, bukan menerus.
    minute_of_day = (index.hour * 60 + index.minute).to_numpy()
    discharge_cycle = ((minute_of_day % 45) < 12) & running
    stall = signature["bottom_ash_blocking"]["slag_discharge_stall"] * ash_block
    discharge_active = discharge_cycle & (rng.random(size) > stall * 0.92)

    bottom_ash_flow = np.where(discharge_active, 1.5 + 1.1 * load_fraction, 0.05)
    bottom_ash_flow *= 1.0 + signature["bottom_ash_blocking"]["bottom_ash_flow"] * ash_block
    bottom_ash_flow = np.clip(bottom_ash_flow + rng.normal(0, 0.04 * scale, size), 0, None)

    ash_cooler_motor_current = 18.0 + 9.0 * load_fraction
    ash_cooler_motor_current *= (
        1.0
        + signature["bottom_ash_blocking"]["ash_cooler_motor_current"] * ash_block
        + signature["furnace_agglomeration"]["ash_cooler_motor_current"] * agglomeration
    )
    ash_cooler_motor_current += _ar1(rng, size, 0.98, 0.22 * scale)

    slag_pipe_base = 190.0 + 120.0 * load_fraction
    slag_pipe_temperature = slag_pipe_base * (
        1.0 + signature["bottom_ash_blocking"]["slag_pipe_temperature"] * ash_block
    )

    # ---------------- Sistem sirkulasi kembali ----------------
    return_leg_temperature = (
        main_bed_temperature - 45.0
    ) * (
        1.0
        + signature["return_system_disturbance"]["return_leg_temperature"] * return_disturb
    ) + _ar1(rng, size, 0.99, 2.2 * scale)

    return_leg_pressure = 4.2 * (0.5 + 0.5 * load_fraction)
    return_leg_pressure *= (
        1.0 + signature["return_system_disturbance"]["return_leg_pressure"] * return_disturb
    )
    return_leg_pressure += _ar1(rng, size, 0.98, 0.06 * scale)

    u_valve_air_pressure = 9.5 * (0.6 + 0.4 * load_fraction)
    u_valve_air_pressure *= (
        1.0
        + signature["return_system_disturbance"]["u_valve_air_pressure"] * return_disturb
    )
    u_valve_air_pressure += _ar1(rng, size, 0.98, 0.08 * scale)

    separator_differential_pressure = 1.8 * (0.4 + 0.6 * load_fraction)
    separator_differential_pressure *= (
        1.0
        + signature["return_system_disturbance"]["separator_differential_pressure"]
        * return_disturb
    )
    separator_differential_pressure += _ar1(rng, size, 0.98, 0.04 * scale)

    furnace_outlet_temperature = main_bed_temperature - 70.0 + _ar1(
        rng, size, 0.98, 2.5 * scale
    )

    # ---------------- Rakit dataframe ----------------
    frame = pd.DataFrame({"timestamp": index})
    frame["unit_load_mw"] = load
    frame["main_steam_flow"] = steam_flow
    frame["main_steam_pressure"] = steam_pressure
    frame["main_steam_temperature"] = steam_temperature
    frame["coal_flow_total"] = coal_flow

    for position in range(feeder_count):
        frame[f"coal_feeder_{position + 1}_flow"] = feeder_flow[:, position]
        frame[f"coal_feeder_{position + 1}_command"] = feeder_command[:, position]
        frame[f"coal_feeder_{position + 1}_current"] = feeder_current[:, position]

    frame["main_bed_temperature"] = main_bed_temperature
    frame["front_aux_bed_temperature"] = front_aux_bed_temperature
    frame["rear_aux_bed_temperature"] = rear_aux_bed_temperature
    frame["main_bed_pressure"] = main_bed_pressure
    frame["bed_differential_pressure"] = bed_differential_pressure
    frame["aux_bed_pressure"] = aux_bed_pressure
    frame["furnace_pressure"] = furnace_pressure
    frame["main_bed_air_flow"] = main_bed_air_flow
    frame["auxiliary_bed_air_flow"] = auxiliary_bed_air_flow
    frame["front_aux_bed_air_flow"] = front_aux_air_flow
    frame["rear_aux_bed_air_flow"] = rear_aux_air_flow
    frame["primary_air_pressure"] = primary_air_pressure
    frame["secondary_air_flow"] = secondary_air_flow
    frame["oxygen_o2"] = oxygen
    frame["carbon_monoxide_co"] = carbon_monoxide
    frame["id_fan_current"] = id_fan_current
    frame["bottom_ash_discharge_status"] = discharge_active.astype(np.int8)
    frame["bottom_ash_flow"] = bottom_ash_flow
    frame["ash_cooler_motor_current"] = ash_cooler_motor_current
    frame["ash_cooler_inlet_temperature"] = slag_pipe_temperature + 25.0
    frame["ash_cooler_outlet_temperature"] = 95.0 + 40.0 * load_fraction
    frame["furnace_outlet_temperature"] = furnace_outlet_temperature

    for pipe in range(1, int(units["furnace"]["slag_pipe_count"]) + 1):
        frame[f"slag_pipe_{pipe}_status"] = discharge_active.astype(np.int8)
        frame[f"slag_pipe_{pipe}_temperature"] = slag_pipe_temperature + rng.normal(
            0.0, 4.0 * scale, size
        )

    frame["separator_differential_pressure"] = separator_differential_pressure
    frame["return_leg_temperature"] = return_leg_temperature
    frame["return_leg_pressure"] = return_leg_pressure
    frame["u_valve_air_pressure"] = u_valve_air_pressure

    # ---------------- Label dan penanda ----------------
    frame["is_running"] = running.astype(np.int8)
    frame["outage_active"] = timeline["outage"].astype(np.int8)
    frame["event_start"] = timeline["event_start"].astype(np.int8)
    for event_type in DEGRADATION_SIGNATURES:
        frame[f"ramp_{event_type}"] = timeline[f"ramp_{event_type}"]

    # ---------------- Cacat data yang disengaja ----------------
    _inject_data_defects(frame, rng, spec)

    # float32 memadai untuk pengukuran proses dan memangkas separuh ukuran
    # berkas. Presisi sensor lapangan jauh di bawah presisi float32.
    for column in frame.columns:
        if frame[column].dtype == np.float64:
            frame[column] = frame[column].astype(np.float32)

    frame[DATA_SOURCE_COLUMN] = DATA_SOURCE_SYNTHETIC
    frame["synthetic_seed"] = np.int32(spec.seed)
    return frame


def _inject_data_defects(
    frame: pd.DataFrame, rng: np.random.Generator, spec: GenerationSpec
) -> None:
    """Sisipkan data hilang dan pembacaan macet.

    Historian sungguhan punya keduanya. Tanpa cacat ini, modul kualitas
    data tidak pernah teruji dan skornya selalu sempurna secara palsu.
    """
    analog = [
        column
        for column in frame.columns
        if frame[column].dtype.kind == "f"
        and not column.startswith("ramp_")
        and column not in {"synthetic_seed"}
    ]
    size = len(frame)
    days = max(size / 1440.0, 1.0)

    for column in analog:
        values = frame[column].to_numpy(copy=True)

        # Data hilang berupa rumpun, bukan titik tunggal.
        for _ in range(int(rng.poisson(spec.gap_probability * days * 1440 / 60))):
            start = int(rng.integers(0, max(size - 90, 1)))
            length = int(rng.integers(5, 90))
            values[start : start + length] = np.nan

        # Sensor macet: nilai membeku beberapa saat. Ini lebih berbahaya
        # daripada data hilang — nilainya tampak sah tetapi sudah basi.
        for _ in range(int(rng.poisson(spec.stuck_probability * days * 1440 / 60))):
            start = int(rng.integers(0, max(size - 180, 1)))
            length = int(rng.integers(30, 180))
            frozen = values[start]
            if np.isfinite(frozen):
                values[start : start + length] = frozen

        frame[column] = values


# --------------------------------------------------------------------
# Alur utama
# --------------------------------------------------------------------
def generate(
    settings: Settings | None = None, spec: GenerationSpec | None = None
) -> dict[str, Any]:
    """Bangkitkan seluruh rentang tahun, ditulis per tahun ke Parquet.

    Pembangkitan dipotong per tahun supaya penggunaan memori puncak tetap
    sekitar satu tahun data, bukan sepuluh tahun sekaligus.
    """
    settings = settings or get_settings()
    settings.paths.ensure()

    spec = spec or GenerationSpec(
        start_year=2020,
        end_year=2026,
        freq="1min",
        seed=settings.synthetic_seed,
    )

    from backend.app.data.event_etl import load_registry

    registry = load_registry(settings)

    output_dir = settings.paths.synthetic
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "data_source": DATA_SOURCE_SYNTHETIC,
        "seed": spec.seed,
        "freq": spec.freq,
        "years": [],
        "warning": (
            "Data sintetis. Metrik model yang dihitung di atasnya tidak boleh "
            "dikutip sebagai unjuk kerja lapangan."
        ),
    }

    registry_end = registry["start_time"].max()

    for year in range(spec.start_year, spec.end_year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59)
        # Tahun terakhir dipotong pada event terakhir yang tercatat —
        # membangkitkan data setelahnya berarti mengarang periode yang
        # tidak punya rujukan sama sekali.
        if year == spec.end_year:
            end = min(end, registry_end.normalize() + pd.Timedelta(days=1))

        index = pd.date_range(start, end, freq=spec.freq, name="timestamp")
        rng = np.random.default_rng(spec.seed + year)
        chunk = generate_chunk(index, registry, settings, spec, rng)

        path = output_dir / f"timeseries_unit1_{year}.parquet"
        chunk.to_parquet(path, index=False, compression="snappy")

        manifest["years"].append(
            {
                "year": year,
                "rows": int(len(chunk)),
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1_048_576, 1),
                "running_share": round(float(chunk["is_running"].mean()), 3),
                "event_starts": int(chunk["event_start"].sum()),
            }
        )
        LOGGER.info(
            "%d: %d baris, %.1f MB, %d event mulai, unit jalan %.1f%%",
            year,
            len(chunk),
            path.stat().st_size / 1_048_576,
            int(chunk["event_start"].sum()),
            100 * float(chunk["is_running"].mean()),
        )
        del chunk

    manifest["total_rows"] = sum(entry["rows"] for entry in manifest["years"])
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_synthetic(
    settings: Settings | None = None,
    years: list[int] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Muat timeseries sintetis dari disk, opsional dibatasi tahun tertentu."""
    settings = settings or get_settings()
    directory = settings.paths.synthetic
    paths = sorted(directory.glob("timeseries_unit1_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            "Data sintetis belum dibangkitkan. Jalankan: "
            "python -m backend.app.data.synthetic"
        )
    if years:
        wanted = {str(year) for year in years}
        paths = [p for p in paths if p.stem.rsplit("_", 1)[-1] in wanted]

    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _parse_years(text: str) -> tuple[int, int]:
    if "-" in text:
        start, end = text.split("-", 1)
        return int(start), int(end)
    year = int(text)
    return year, year


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generator timeseries sintetis PLTU Jeranjang Unit 1"
    )
    parser.add_argument(
        "--years",
        default="2020-2026",
        help="Rentang tahun, contoh 2022-2026 (bawaan 2020-2026)",
    )
    parser.add_argument("--freq", default="1min", help="Jarak sampel (bawaan 1min)")
    parser.add_argument("--seed", type=int, default=None, help="Seed acak")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Pengali amplitudo derau (bawaan 1.0)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    start_year, end_year = _parse_years(args.years)
    spec = GenerationSpec(
        start_year=start_year,
        end_year=end_year,
        freq=args.freq,
        seed=args.seed if args.seed is not None else settings.synthetic_seed,
        noise_scale=args.noise_scale,
    )

    manifest = generate(settings, spec)

    print()
    print("=" * 72)
    print("GENERATOR TIMESERIES SINTETIS — PLTU JERANJANG UNIT 1")
    print("=" * 72)
    print("PERINGATAN: data ini SINTETIS, bukan data operasi pembangkit.")
    print("Metrik model di atasnya tidak boleh dikutip sebagai unjuk kerja lapangan.")
    print()
    print(f"Seed            : {manifest['seed']}")
    print(f"Jarak sampel    : {manifest['freq']}")
    print(f"Total baris     : {manifest['total_rows']:,}")
    print()
    print(f"{'Tahun':>6} {'Baris':>10} {'MB':>7} {'Event':>6} {'Jalan':>7}")
    for entry in manifest["years"]:
        print(
            f"{entry['year']:>6} {entry['rows']:>10,} {entry['size_mb']:>7.1f} "
            f"{entry['event_starts']:>6} {entry['running_share']*100:>6.1f}%"
        )
    print()
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
