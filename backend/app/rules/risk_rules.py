"""Rule engine risiko dan pemetaan ke Risk Score 0-100.

README §11 menempatkan rule engine berdampingan dengan model machine
learning dan detektor anomali, lalu menggabungkan ketiganya di Hybrid
Risk Engine. Modul ini mengerjakan bagian rule engine dan penggabungannya.

Mengapa rule engine tetap ada meski model sudah dilatih: aturan dapat
dibaca, diperdebatkan, dan diubah oleh engineer boiler tanpa melatih
ulang apa pun. Saat model memberi angka tinggi tanpa alasan yang bisa
dijelaskan, aturanlah yang memberi tahu operator gejala mana yang
sedang menyala.

Seluruh ambang berasal dari ``config/thresholds.yaml``. Tidak ada satu pun
angka ambang yang ditulis di dalam kode ini.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

LOGGER = get_logger(__name__)

#: Operator perbandingan yang dikenali pada ``thresholds.yaml``.
OPERATORS = {
    ">": lambda values, threshold: values > threshold,
    ">=": lambda values, threshold: values >= threshold,
    "<": lambda values, threshold: values < threshold,
    "<=": lambda values, threshold: values <= threshold,
    "abs>": lambda values, threshold: np.abs(values) > threshold,
    "abs<": lambda values, threshold: np.abs(values) < threshold,
}


@dataclass
class RuleTrigger:
    """Satu kondisi aturan yang menyala pada satu titik waktu."""

    rule: str
    event_type: str
    feature: str
    reason: str
    points: float
    value: float
    threshold: float

    def describe(self) -> str:
        return f"{self.reason} ({self.feature} = {self.value:.3g})"


@dataclass
class RiskAssessment:
    """Hasil penilaian risiko satu titik waktu, siap disajikan (README §14)."""

    risk_score: float
    status: str
    status_meaning: str
    confidence_score: float
    data_quality_score: float
    suspected_area: str | None
    dominant_indicators: list[str]
    rule_score: float
    model_score: float
    anomaly_score: float
    triggers: list[RuleTrigger] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 1),
            "status": self.status,
            "status_meaning": self.status_meaning,
            "confidence_score": round(self.confidence_score, 1),
            "data_quality_score": round(self.data_quality_score, 1),
            "suspected_area": self.suspected_area,
            "dominant_indicators": self.dominant_indicators,
            "components": {
                "rule_engine": round(self.rule_score, 1),
                "ml_model": round(self.model_score, 1),
                "anomaly": round(self.anomaly_score, 1),
            },
        }


# --------------------------------------------------------------------
# Rule engine
# --------------------------------------------------------------------
def evaluate_rules(
    features: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Jalankan seluruh aturan README §6 pada matriks fitur.

    Mengembalikan pasangan:
      * dataframe skor per aturan, 0-100, satu kolom per aturan;
      * seri skor rule engine gabungan, yaitu nilai aturan tertinggi.

    Nilai tertinggi dipakai, bukan penjumlahan: satu gangguan yang jelas
    sudah cukup untuk menaikkan risiko. Menjumlahkan beberapa aturan akan
    membuat gangguan tunggal yang parah kalah oleh beberapa gangguan
    ringan yang tidak berhubungan.
    """
    settings = settings or get_settings()
    rules = settings.thresholds["rules"]

    scores: dict[str, pd.Series] = {}
    missing_features: set[str] = set()

    for rule_name, spec in rules.items():
        conditions = spec.get("conditions") or []
        total_points = sum(float(c["points"]) for c in conditions)
        if total_points <= 0:
            continue

        earned = pd.Series(0.0, index=features.index)
        usable_points = 0.0

        for condition in conditions:
            feature = str(condition["feature"])
            if feature not in features.columns:
                missing_features.add(feature)
                continue

            operator = OPERATORS.get(str(condition["operator"]))
            if operator is None:
                raise ValueError(
                    f"thresholds.yaml: operator tidak dikenal "
                    f"{condition['operator']!r} pada aturan {rule_name}."
                )

            values = features[feature].to_numpy(dtype=np.float64)
            threshold = float(condition["value"])
            fired = operator(values, threshold) & np.isfinite(values)

            points = float(condition["points"])
            usable_points += points
            earned += pd.Series(fired.astype(float) * points, index=features.index)

        if usable_points <= 0:
            continue
        scores[rule_name] = (100.0 * earned / usable_points).clip(0.0, 100.0)

    if missing_features:
        LOGGER.warning(
            "Fitur berikut dirujuk thresholds.yaml tetapi tidak ada di data: %s",
            ", ".join(sorted(missing_features)),
        )

    if not scores:
        raise ValueError(
            "Tidak ada aturan yang dapat dievaluasi. Periksa apakah nama fitur "
            "di thresholds.yaml cocok dengan keluaran features.engineering."
        )

    rule_frame = pd.DataFrame(scores, index=features.index)
    return rule_frame, rule_frame.max(axis=1).rename("rule_score")


def rule_triggers_at(
    features: pd.DataFrame, position: int, settings: Settings | None = None
) -> list[RuleTrigger]:
    """Kumpulkan kondisi aturan yang menyala pada satu baris tertentu.

    Dipakai untuk menyusun daftar indikator dominan pada keluaran operator
    (README §14) — bagian yang menjelaskan mengapa skornya tinggi.
    """
    settings = settings or get_settings()
    rules = settings.thresholds["rules"]
    row = features.iloc[position]
    triggers: list[RuleTrigger] = []

    for rule_name, spec in rules.items():
        for condition in spec.get("conditions") or []:
            feature = str(condition["feature"])
            if feature not in features.columns:
                continue
            value = row[feature]
            if not np.isfinite(value):
                continue
            operator = OPERATORS[str(condition["operator"])]
            threshold = float(condition["value"])
            if bool(operator(np.array([value]), threshold)[0]):
                triggers.append(
                    RuleTrigger(
                        rule=rule_name,
                        event_type=str(spec.get("event_type", rule_name)),
                        feature=feature,
                        reason=str(condition["reason"]),
                        points=float(condition["points"]),
                        value=float(value),
                        threshold=threshold,
                    )
                )

    return sorted(triggers, key=lambda trigger: trigger.points, reverse=True)


def suspected_area(
    triggers: list[RuleTrigger], settings: Settings | None = None
) -> str | None:
    """Tentukan lokasi terduga dari aturan yang paling banyak menyala."""
    if not triggers:
        return None
    settings = settings or get_settings()
    rules = settings.thresholds["rules"]

    weight: dict[str, float] = {}
    for trigger in triggers:
        weight[trigger.rule] = weight.get(trigger.rule, 0.0) + trigger.points

    best_rule = max(weight, key=weight.get)
    return rules.get(best_rule, {}).get("default_location")


# --------------------------------------------------------------------
# Hybrid Risk Engine (README §11, §13)
# --------------------------------------------------------------------
def probability_to_risk(
    probability: pd.Series | np.ndarray,
    threshold: float,
    spread: float = 1.6,
) -> pd.Series:
    """Ubah probabilitas model menjadi skor risiko 0-100 berporos ambang.

    Probabilitas yang terkalibrasi baik pada peristiwa yang hanya terjadi
    satu dari seribu menit akan hampir selalu bernilai kecil. Memakainya
    langsung sebagai skor 0-100 membuat Risk Score tidak pernah beranjak
    dari angka rendah, betapapun jelas gejalanya — model yang benar justru
    terlihat seperti tidak menemukan apa-apa.

    Pemetaan ini memindahkan porosnya ke ambang keputusan: probabilitas
    tepat di ambang menjadi **50**, yaitu batas bawah status Warning pada
    README §13 dan sekaligus titik ketika sistem memang mulai membunyikan
    alarm. Di bawah ambang skornya turun, di atasnya naik, dan urutannya
    tetap sama persis dengan urutan probabilitas.

    Perbandingan dilakukan dalam log-odds, bukan selisih biasa, supaya
    perbedaan antara 0,001 dan 0,01 terbaca sebesar perbedaan antara 0,1
    dan 0,5 — pada peristiwa jarang, keduanya memang sama pentingnya.
    """
    values = np.asarray(
        probability.to_numpy() if isinstance(probability, pd.Series) else probability,
        dtype=np.float64,
    )
    epsilon = 1e-9
    values = np.clip(values, epsilon, 1.0 - epsilon)
    anchor = float(np.clip(threshold, epsilon, 1.0 - epsilon))

    logit = np.log(values / (1.0 - values))
    anchor_logit = np.log(anchor / (1.0 - anchor))
    score = 100.0 / (1.0 + np.exp(-(logit - anchor_logit) / spread))

    index = probability.index if isinstance(probability, pd.Series) else None
    return pd.Series(score, index=index, name="model_risk_score")


def combine_scores(
    rule_score: pd.Series,
    model_score: pd.Series | None = None,
    anomaly_score: pd.Series | None = None,
    settings: Settings | None = None,
) -> pd.Series:
    """Gabungkan ketiga sumber menjadi satu Risk Score 0-100.

    Bobot dari ``thresholds.yaml``. Bila salah satu sumber tidak tersedia,
    bobotnya dibagikan ke sumber yang ada agar skala tetap 0-100.
    """
    settings = settings or get_settings()
    weights = dict(settings.thresholds["hybrid_weights"])

    parts: dict[str, tuple[pd.Series, float]] = {
        "rule_engine": (rule_score, float(weights["rule_engine"]))
    }
    if model_score is not None:
        parts["ml_classifier"] = (model_score, float(weights["ml_classifier"]))
    if anomaly_score is not None:
        parts["anomaly_detector"] = (anomaly_score, float(weights["anomaly_detector"]))

    total_weight = sum(weight for _, weight in parts.values())
    combined = pd.Series(0.0, index=rule_score.index)
    for series, weight in parts.values():
        combined += series.fillna(0.0) * (weight / total_weight)

    return combined.clip(0.0, 100.0).rename("risk_score")


def apply_persistence(
    risk_score: pd.Series, settings: Settings | None = None
) -> pd.Series:
    """Haluskan skor dengan syarat ketahanan minimum (README, alarm).

    Lonjakan satu menit akibat derau sensor tidak boleh memicu alarm.
    Skor yang dilaporkan adalah nilai minimum selama jendela ketahanan,
    sehingga hanya kondisi yang benar-benar bertahan yang naik tinggi.
    """
    settings = settings or get_settings()
    minutes = int(settings.thresholds["alarm"]["persistence_minutes"])
    if minutes <= 1:
        return risk_score
    return (
        risk_score.rolling(minutes, min_periods=1)
        .min()
        .rename("risk_score_persistent")
    )


def confidence_from_quality(
    model_probability: pd.Series,
    data_quality_score: pd.Series,
    settings: Settings | None = None,
) -> pd.Series:
    """Hitung Confidence Score terpisah dari Risk Score (README §13).

    Keyakinan tinggi berarti model berada jauh dari titik ragu 0,5 DAN
    data masukannya layak. Prediksi tegas di atas data buruk tidak berhak
    disebut meyakinkan.
    """
    settings = settings or get_settings()
    penalty = settings.thresholds["confidence_penalty"]
    threshold = float(penalty["data_quality_below"])
    factor = float(penalty["penalty_factor"])

    decisiveness = (2.0 * (model_probability - 0.5).abs()).clip(0.0, 1.0)
    confidence = 100.0 * decisiveness
    poor_data = data_quality_score < threshold
    confidence = confidence.where(~poor_data, confidence * factor)
    return confidence.clip(0.0, 100.0).rename("confidence_score")


def status_for(score: float, settings: Settings | None = None) -> tuple[str, str]:
    """Petakan skor ke label status dan maknanya (README §13)."""
    settings = settings or get_settings()
    band = settings.risk_band_for(score)
    return str(band["label"]), str(band["meaning"])


def assess_row(
    features: pd.DataFrame,
    position: int,
    risk_score: float,
    model_probability: float,
    data_quality: float,
    rule_score: float,
    anomaly_score: float,
    settings: Settings | None = None,
    top_n: int = 5,
) -> RiskAssessment:
    """Susun satu penilaian lengkap siap tampil untuk operator."""
    settings = settings or get_settings()
    triggers = rule_triggers_at(features, position, settings)
    label, meaning = status_for(risk_score, settings)

    decisiveness = min(1.0, 2.0 * abs(model_probability - 0.5))
    confidence = 100.0 * decisiveness
    penalty = settings.thresholds["confidence_penalty"]
    if data_quality < float(penalty["data_quality_below"]):
        confidence *= float(penalty["penalty_factor"])

    return RiskAssessment(
        risk_score=risk_score,
        status=label,
        status_meaning=meaning,
        confidence_score=confidence,
        data_quality_score=data_quality,
        suspected_area=suspected_area(triggers, settings),
        dominant_indicators=[trigger.describe() for trigger in triggers[:top_n]],
        rule_score=rule_score,
        model_score=100.0 * model_probability,
        anomaly_score=anomaly_score,
        triggers=triggers,
    )
