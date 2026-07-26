"""Penjelasan prediksi model dengan SHAP.

README §14 menuntut keluaran menyebutkan indikator dominan dan suspected
area, bukan hanya angka risiko. Skor tanpa alasan tidak dapat ditindak
lanjuti: operator tidak bisa memeriksa "81 persen", ia memeriksa slag
pipe 2 karena aliran abunya berhenti.

Modul ini menyediakan dua lapis penjelasan yang saling melengkapi:

* SHAP — kontribusi tiap fitur terhadap satu prediksi, diambil dari model
  itu sendiri. Menjawab: apa yang dilihat model.
* Rule engine (:mod:`backend.app.rules.risk_rules`) — kondisi teknik yang
  menyala. Menjawab: gejala apa yang sedang terjadi menurut pemahaman
  boiler.

Keduanya digabung dalam :func:`explain` supaya penjelasan yang sampai ke
operator memakai bahasa peralatan, bukan nama kolom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

LOGGER = get_logger(__name__)

#: Terjemahan nama kolom fitur menjadi kalimat yang dimengerti operator.
FEATURE_LABELS: dict[str, str] = {
    "bed_differential_pressure": "Bed differential pressure",
    "main_bed_pressure": "Tekanan main bed",
    "aux_bed_pressure": "Tekanan auxiliary bed",
    "main_bed_temperature": "Temperatur main bed",
    "front_aux_bed_temperature": "Temperatur front auxiliary bed",
    "rear_aux_bed_temperature": "Temperatur rear auxiliary bed",
    "bed_temp_spread": "Sebaran temperatur bed",
    "main_aux_temp_spread": "Selisih temperatur main dan auxiliary bed",
    "main_bed_air_resistance": "Resistansi udara main bed",
    "aux_bed_air_resistance": "Resistansi udara auxiliary bed",
    "main_bed_air_flow": "Aliran udara main bed",
    "auxiliary_bed_air_flow": "Aliran udara auxiliary bed",
    "aux_air_flow_imbalance": "Ketidakseimbangan udara auxiliary bed",
    "primary_air_pressure": "Tekanan primary air",
    "secondary_air_flow": "Aliran secondary air",
    "furnace_pressure": "Tekanan furnace",
    "oxygen_o2": "Kandungan O2",
    "carbon_monoxide_co": "Kandungan CO",
    "coal_flow_total": "Total aliran batubara",
    "coal_feeder_imbalance": "Ketidakseimbangan coal feeder",
    "coal_feeder_command_deviation": "Selisih perintah dan aliran coal feeder",
    "coal_feeder_current_max": "Arus motor coal feeder tertinggi",
    "coal_feeder_current_spread": "Sebaran arus motor coal feeder",
    "air_to_fuel_ratio": "Rasio udara terhadap bahan bakar",
    "combustion_instability_index": "Indeks ketidakstabilan pembakaran",
    "ash_cooler_motor_current": "Arus motor ash cooler",
    "ash_cooler_current_slope": "Laju kenaikan arus ash cooler",
    "slag_discharge_failure": "Kegagalan pembuangan slag",
    "bottom_ash_flow": "Aliran bottom ash",
    "return_system_disturbance": "Gangguan sistem sirkulasi kembali",
    "return_leg_temperature": "Temperatur return leg",
    "return_leg_pressure": "Tekanan return leg",
    "u_valve_air_pressure": "Tekanan udara U-valve",
    "separator_differential_pressure": "Differential pressure separator",
    "deviation_from_baseline": "Simpangan rata-rata dari baseline zona beban",
    "deviation_from_baseline_max": "Simpangan terbesar dari baseline zona beban",
    "unit_load_mw": "Beban unit",
    "main_steam_flow": "Aliran main steam",
    "steam_flow_per_mw": "Aliran steam per MW",
    "coal_flow_per_mw": "Aliran batubara per MW",
    "id_fan_current": "Arus ID fan",
}

#: Akhiran nama fitur turunan, diterjemahkan menjadi keterangan.
SUFFIX_LABELS: list[tuple[str, str]] = [
    ("_rolling_mean_5m", "rata-rata 5 menit"),
    ("_rolling_mean_15m", "rata-rata 15 menit"),
    ("_rolling_mean_30m", "rata-rata 30 menit"),
    ("_rolling_mean_60m", "rata-rata 60 menit"),
    ("_rolling_std_15m", "fluktuasi 15 menit"),
    ("_slope_15m", "kecenderungan 15 menit"),
    ("_slope_30m", "kecenderungan 30 menit"),
    ("_roc_5m", "perubahan 5 menit"),
    ("_dev", "simpangan dari baseline"),
]


def humanise(feature: str) -> str:
    """Ubah nama kolom fitur menjadi kalimat berbahasa Indonesia."""
    for suffix, description in SUFFIX_LABELS:
        if feature.endswith(suffix):
            base = feature[: -len(suffix)]
            label = FEATURE_LABELS.get(base, base.replace("_", " "))
            return f"{label} ({description})"
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


@dataclass
class Explanation:
    """Penjelasan satu prediksi, siap ditampilkan (README §14)."""

    row_index: int
    probability: float
    top_features: list[tuple[str, float]] = field(default_factory=list)
    rule_indicators: list[str] = field(default_factory=list)
    suspected_area: str | None = None

    def dominant_indicators(self, limit: int = 5) -> list[str]:
        """Gabungkan indikator aturan dan SHAP, aturan didahulukan.

        Aturan didahulukan karena kalimatnya sudah berbentuk gejala
        teknik yang bisa langsung diperiksa di lapangan. SHAP dipakai
        melengkapi ketika aturan yang menyala belum cukup banyak.
        """
        combined = list(self.rule_indicators)
        for feature, contribution in self.top_features:
            if len(combined) >= limit:
                break
            arah = "meningkat" if contribution > 0 else "menurun"
            sentence = f"{humanise(feature)} {arah}"
            if sentence not in combined:
                combined.append(sentence)
        return combined[:limit]


class ShapExplainer:
    """Pembungkus SHAP untuk model berbasis pohon maupun linear."""

    def __init__(self, model: Any, columns: list[str], background: np.ndarray | None = None):
        self.model = model
        self.columns = columns
        self._explainer = None
        self._background = background

    @staticmethod
    def _unwrap(model: Any) -> Any:
        """Ambil model dasar dari pembungkus kalibrator.

        ``CalibratedClassifierCV`` menyimpan model aslinya di dalam;
        SHAP butuh model itu, bukan pembungkusnya.
        """
        inner = getattr(model, "estimator", None)
        if inner is not None:
            deeper = getattr(inner, "estimator", None)
            return deeper if deeper is not None else inner

        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            first = calibrated[0]
            return getattr(first, "estimator", first)
        return model

    def fit(self, background: pd.DataFrame | np.ndarray | None = None) -> ShapExplainer:
        """Siapkan explainer, memilih algoritma sesuai jenis model."""
        import shap

        base = self._unwrap(self.model)
        if background is not None:
            self._background = (
                background.to_numpy(dtype=np.float32)
                if isinstance(background, pd.DataFrame)
                else background
            )

        try:
            self._explainer = shap.TreeExplainer(base)
            LOGGER.info("SHAP: TreeExplainer dipakai untuk %s", type(base).__name__)
        except Exception:  # pragma: no cover - bergantung jenis model
            if self._background is None:
                raise ValueError(
                    "Model bukan berbasis pohon; SHAP membutuhkan sampel latar. "
                    "Berikan argumen background."
                )
            self._explainer = shap.LinearExplainer(base, self._background)
            LOGGER.info("SHAP: LinearExplainer dipakai untuk %s", type(base).__name__)
        return self

    def contributions(self, features: pd.DataFrame) -> np.ndarray:
        """Hitung nilai SHAP untuk setiap baris dan setiap fitur."""
        if self._explainer is None:
            self.fit()

        matrix = features.reindex(columns=self.columns).to_numpy(dtype=np.float32)
        np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        values = self._explainer.shap_values(matrix)
        if isinstance(values, list):
            # Keluaran dua kelas: ambil kelas positif.
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]
        return values

    def global_importance(self, features: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Peringkat fitur menurut rata-rata besar kontribusinya."""
        values = np.abs(self.contributions(features)).mean(axis=0)
        frame = pd.DataFrame(
            {
                "feature": self.columns,
                "label": [humanise(column) for column in self.columns],
                "mean_abs_shap": values,
            }
        )
        return frame.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(
            drop=True
        )


def explain(
    features: pd.DataFrame,
    probabilities: np.ndarray,
    positions: list[int],
    explainer: ShapExplainer | None = None,
    settings: Settings | None = None,
    top_n: int = 5,
) -> list[Explanation]:
    """Susun penjelasan untuk baris-baris tertentu.

    Bila ``explainer`` tidak diberikan, penjelasan disusun dari rule engine
    saja. Itu tetap berguna — aturan justru lebih mudah dipertanggung-
    jawabkan di depan engineer boiler daripada nilai SHAP.
    """
    from backend.app.rules.risk_rules import rule_triggers_at, suspected_area

    settings = settings or get_settings()
    explanations: list[Explanation] = []

    shap_values: np.ndarray | None = None
    if explainer is not None and positions:
        subset = features.iloc[positions]
        shap_values = explainer.contributions(subset)

    for order, position in enumerate(positions):
        triggers = rule_triggers_at(features, position, settings)

        top_features: list[tuple[str, float]] = []
        if shap_values is not None:
            row = shap_values[order]
            ranking = np.argsort(np.abs(row))[::-1][:top_n]
            top_features = [
                (explainer.columns[index], float(row[index])) for index in ranking
            ]

        explanations.append(
            Explanation(
                row_index=position,
                probability=float(probabilities[position]),
                top_features=top_features,
                rule_indicators=[trigger.describe() for trigger in triggers[:top_n]],
                suspected_area=suspected_area(triggers, settings),
            )
        )

    return explanations


def format_operator_output(
    timestamp: pd.Timestamp,
    load_mw: float,
    steam_flow: float,
    risk_score: float,
    status: str,
    confidence: float,
    data_quality: float,
    suspected_area: str | None,
    indicators: list[str],
    horizon_minutes: tuple[int, int],
    recommendations: list[str],
) -> str:
    """Susun keluaran teks sesuai contoh README §14."""
    from backend.app.core.constants import DISCLAIMER

    area = (suspected_area or "belum dapat ditentukan").replace("_", " ").title()
    lines = [
        "PLTU JERANJANG UNIT 1",
        "FURNACE BLOCKING EARLY WARNING",
        "",
        f"Waktu               : {timestamp:%Y-%m-%d %H:%M} WITA",
        f"Unit Load           : {load_mw:.1f} MW".replace(".", ","),
        f"Boiler Steam Flow   : {steam_flow:.1f} t/h".replace(".", ","),
        f"Risk Score          : {risk_score:.0f}%",
        f"Status              : {status.upper()}",
        f"Prediction Horizon  : {horizon_minutes[0]}-{horizon_minutes[1]} menit",
        f"Confidence Score    : {confidence:.0f}%",
        f"Data Quality Score  : {data_quality:.0f}%",
        "",
        "Suspected Area:",
        area,
        "",
        "Dominant Indicators:",
    ]
    lines.extend(
        f"{number}. {indicator}" for number, indicator in enumerate(indicators, 1)
    )
    lines.extend(["", "Recommended Verification:"])
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(["", "-" * 60, DISCLAIMER])
    return "\n".join(lines)


#: Rekomendasi pemeriksaan per lokasi terduga (README §14).
VERIFICATION_ACTIONS: dict[str, list[str]] = {
    "main_bed": [
        "Periksa distribusi udara main bed dan kondisi air nozzle",
        "Periksa temperatur bed di seluruh titik pengukuran",
        "Verifikasi aliran bottom ash dari keempat slag pipe",
        "Periksa keseimbangan coal feeder",
        "Konfirmasi kondisi lokal sesuai SOP",
    ],
    "front_auxiliary_bed": [
        "Periksa aliran udara fluidisasi front auxiliary bed",
        "Bandingkan temperatur front auxiliary bed dengan main bed",
        "Periksa kemungkinan penyumbatan wind cap sisi depan",
        "Konfirmasi kondisi lokal sesuai SOP",
    ],
    "rear_auxiliary_bed": [
        "Periksa aliran udara fluidisasi rear auxiliary bed",
        "Periksa return leg dan U-type return valve",
        "Bandingkan temperatur rear auxiliary bed dengan main bed",
        "Konfirmasi kondisi lokal sesuai SOP",
    ],
    "slag_pipe_unknown": [
        "Verifikasi aliran bottom ash dari keempat slag pipe",
        "Periksa ash cooler dan valve terkait",
        "Periksa arus motor ash cooler terhadap riwayatnya",
        "Konfirmasi kondisi lokal sesuai SOP",
    ],
    "coal_feeder_unknown": [
        "Periksa aliran nyata setiap coal feeder terhadap perintahnya",
        "Periksa arus motor dan chain cleaning coal feeder",
        "Periksa kondisi kelembaban dan ukuran batubara di chute",
        "Periksa temperatur bed pada sisi feeder yang dicurigai",
        "Konfirmasi kondisi lokal sesuai SOP",
    ],
    "return_leg": [
        "Periksa tekanan udara U-type return valve",
        "Periksa temperatur return leg",
        "Periksa differential pressure cyclone separator",
        "Konfirmasi kondisi lokal sesuai SOP",
    ],
}

DEFAULT_ACTIONS = [
    "Periksa tren parameter bed selama satu jam terakhir",
    "Verifikasi pembacaan sensor terhadap indikasi lokal",
    "Konfirmasi kondisi lokal sesuai SOP",
]


def recommendations_for(area: str | None) -> list[str]:
    """Ambil daftar pemeriksaan yang disarankan untuk satu lokasi terduga."""
    if area is None:
        return DEFAULT_ACTIONS
    return VERIFICATION_ACTIONS.get(area, DEFAULT_ACTIONS)
