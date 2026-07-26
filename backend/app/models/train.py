"""Pelatihan dan evaluasi model risiko blocking furnace.

Alur
----
1. Baseline per zona beban dihitung HANYA dari tahun latih (README §10).
   Menghitungnya dari seluruh data akan membocorkan statistik masa depan
   ke fitur masa lalu.
2. Fitur dibangun per tahun agar penggunaan memori tetap terkendali.
3. Label ``blocking_next_30m/60m/180m`` dibentuk dari waktu mulai event
   nyata yang disuntikkan ke timeline sintetis (README §5).
4. Data dibagi BERDASARKAN WAKTU dengan jeda embargo, bukan diacak
   (README §12).
5. Model dilatih, dikalibrasi, lalu dievaluasi dengan metrik lengkap
   README §12. Ambang keputusan dipilih di data validasi berdasarkan
   anggaran alarm palsu harian.

Dua keputusan yang menentukan kebenaran hasil
---------------------------------------------
**Hanya himpunan latih yang disubsampel.** Validasi dan uji dibiarkan
utuh. Menyubsampel keduanya akan menaikkan prevalensi kelas positif jauh
di atas kenyataan, sehingga ambang yang dipilih di sana akan runtuh
begitu dipakai pada aliran data sungguhan. Metrik alarm palsu per hari
juga hanya bermakna di atas deret waktu yang tidak berlubang.

**Fitur himpunan besar dialirkan dari disk.** Himpunan validasi dan uji
yang utuh berukuran ratusan megabyte. Keduanya ditulis ke Parquet lalu
dibaca per potongan saat prediksi, sehingga puncak pemakaian memori
ditentukan oleh besar potongan, bukan besar himpunan.

PERINGATAN
----------
Seluruh angka unjuk kerja yang dihasilkan modul ini berasal dari DATA
SINTETIS. Angka tersebut membuktikan pipeline berjalan benar, bukan
mengukur kemampuan deteksi di PLTU Jeranjang Unit 1.

Penggunaan
----------
    python -m backend.app.models.train
    python -m backend.app.models.train --horizon blocking_next_60m
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.constants import (
    DATA_SOURCE_SYNTHETIC,
    SYNTHETIC_METRIC_WARNING,
)
from backend.app.core.logging import get_logger
from backend.app.models.evaluate import (
    EvaluationResult,
    choose_threshold,
    comparison_table,
    evaluate,
)
from backend.app.models.registry import ModelEntry, ModelRegistry

LOGGER = get_logger(__name__)

#: Kolom penyerta yang ikut dibawa melewati pipeline tetapi bukan fitur.
CARRY_COLUMNS = ["timestamp", "event_start", "is_running", "load_zone", "unit_load_mw"]

#: Jumlah baris per potongan saat membaca fitur dari disk.
PREDICT_BATCH_ROWS = 60_000

#: Batas jumlah baris untuk pemasangan kalibrator dan Isolation Forest.
CALIBRATION_SAMPLE_LIMIT = 60_000
ANOMALY_SAMPLE_LIMIT = 120_000


@dataclass
class Dataset:
    """Satu himpunan data siap latih.

    Fitur berada di memori (``features``) untuk himpunan kecil, atau di
    berkas Parquet (``path``) untuk himpunan besar yang dialirkan.
    """

    name: str
    labels: dict[str, np.ndarray]
    carry: pd.DataFrame
    features: pd.DataFrame | None = None
    path: Path | None = None
    row_count: int = 0
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.row_count

    def batches(self, batch_rows: int = PREDICT_BATCH_ROWS) -> Iterator[pd.DataFrame]:
        """Hasilkan fitur per potongan, dari memori maupun dari disk."""
        if self.features is not None:
            for start in range(0, len(self.features), batch_rows):
                yield self.features.iloc[start : start + batch_rows]
            return

        if self.path is None:
            raise ValueError(f"Himpunan {self.name} tidak punya fitur maupun berkas.")

        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(self.path)
        for batch in parquet.iter_batches(batch_size=batch_rows):
            yield batch.to_pandas()

    def sample(self, indices: np.ndarray) -> pd.DataFrame:
        """Ambil sebagian baris fitur berdasarkan posisi, terurut menaik."""
        if self.features is not None:
            return self.features.iloc[indices]

        wanted = np.sort(indices)
        collected: list[pd.DataFrame] = []
        offset = 0
        for chunk in self.batches():
            end = offset + len(chunk)
            inside = wanted[(wanted >= offset) & (wanted < end)]
            if inside.size:
                collected.append(chunk.iloc[inside - offset])
            offset = end
            if offset > wanted[-1]:
                break
        return pd.concat(collected, ignore_index=True)


# --------------------------------------------------------------------
# Pelabelan
# --------------------------------------------------------------------
def build_labels(
    event_start: pd.Series,
    horizons: list[dict[str, Any]],
    samples_per_minute: float = 1.0,
) -> dict[str, np.ndarray]:
    """Bentuk label ``blocking_next_*`` dari penanda mulai event.

    Label bernilai 1 pada waktu ``t`` bila ada event yang dimulai dalam
    rentang ``(t, t + horizon]``. Perhatikan arah jendelanya: label
    memang melihat ke depan — itulah yang membuatnya menjadi target
    prediksi. Fitur tidak boleh melakukan hal yang sama.
    """
    marks = event_start.to_numpy(dtype=np.float32)
    labels: dict[str, np.ndarray] = {}

    for horizon in horizons:
        window = int(round(int(horizon["minutes"]) * samples_per_minute))
        # Jumlah event pada jendela ke depan: jumlah bergulir mundur atas
        # deret yang dibalik, lalu dibalik lagi.
        forward = (
            pd.Series(marks[::-1]).rolling(window, min_periods=1).sum().to_numpy()[::-1]
        )
        # Digeser satu langkah agar event yang terjadi TEPAT pada waktu t
        # tidak ikut dihitung — pada saat itu kejadiannya sudah terjadi,
        # bukan lagi diprediksi.
        shifted = np.empty_like(forward)
        shifted[:-1] = forward[1:]
        shifted[-1] = 0.0
        labels[str(horizon["name"])] = (shifted > 0).astype(np.int8)

    return labels


# --------------------------------------------------------------------
# Pembagian waktu
# --------------------------------------------------------------------
@dataclass
class SplitPlan:
    """Batas waktu tiap himpunan, sudah termasuk jeda embargo."""

    train_years: list[int]
    validation_years: list[int]
    test_years: list[int]
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp

    def bounds(self, name: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if name == "train":
            return None, self.train_end
        if name == "validation":
            return self.validation_start, self.validation_end
        return self.test_start, None

    def years(self, name: str) -> list[int]:
        return {
            "train": self.train_years,
            "validation": self.validation_years,
            "test": self.test_years,
        }[name]


def build_split_plan(settings: Settings) -> SplitPlan:
    """Tentukan tahun dan batas waktu tiap himpunan beserta embargonya.

    Embargo diterapkan sebagai potongan waktu nyata di kedua sisi setiap
    perbatasan. Tanpa itu, fitur bergulir di awal himpunan uji masih
    memuat jejak sampel terakhir himpunan validasi, dan hasil ujinya
    menjadi terlalu bagus tanpa alasan yang sah.
    """
    config = settings.model["split"]
    embargo = pd.Timedelta(hours=float(config["embargo_hours"]))
    train_end = pd.Timestamp(config["train_end"])
    validation_end = pd.Timestamp(config["validation_end"])

    manifest_path = settings.paths.synthetic / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Data sintetis belum dibangkitkan. Jalankan: "
            "python -m backend.app.data.synthetic"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    years = sorted(entry["year"] for entry in manifest["years"])

    plan = SplitPlan(
        train_years=[y for y in years if y <= train_end.year],
        validation_years=[
            y for y in years if train_end.year < y <= validation_end.year
        ],
        test_years=[y for y in years if y > validation_end.year],
        train_end=train_end - embargo,
        validation_start=train_end + embargo,
        validation_end=validation_end - embargo,
        test_start=validation_end + embargo,
    )

    if not plan.validation_years or not plan.test_years:
        raise ValueError(
            "Pembagian waktu menghasilkan himpunan validasi atau uji kosong. "
            "Periksa model_config.yaml bagian split terhadap tahun yang tersedia."
        )
    return plan


# --------------------------------------------------------------------
# Penyusunan dataset
# --------------------------------------------------------------------
def fit_training_baseline(settings: Settings, train_years: list[int]):
    """Hitung baseline per zona beban dari tahun latih saja."""
    from backend.app.data.event_etl import load_registry
    from backend.app.data.synthetic import load_synthetic
    from backend.app.features.baseline import assign_load_zone, fit_baseline
    from backend.app.features.engineering import derived_baseline_frame

    base_columns = list(settings.model["features"]["rolling_base_columns"])
    # Kolom tambahan yang hanya dipakai menghitung fitur turunan berbaseline.
    extra = [
        "aux_bed_pressure",
        "coal_feeder_1_current",
        "coal_feeder_2_current",
        "coal_feeder_3_current",
        "coal_feeder_4_current",
    ]
    needed = sorted(
        set(base_columns + extra + ["timestamp", "unit_load_mw", "is_running"])
    )

    frame = load_synthetic(settings, years=train_years, columns=needed)
    frame["load_zone"] = assign_load_zone(frame, settings)
    registry = load_registry(settings)

    # Fitur turunan ikut diberi baseline: thresholds.yaml merujuk versi
    # `_dev` dari resistansi udara dan arus feeder, dan tanpa baseline
    # kolom-kolom itu tidak pernah terbentuk.
    derived = derived_baseline_frame(frame)
    combined = pd.concat([frame, derived], axis=1)
    columns = base_columns + list(derived.columns)

    baseline = fit_baseline(combined, columns, settings, registry=registry)
    del frame, derived, combined
    gc.collect()
    return baseline


def build_year_dataset(
    year: int,
    settings: Settings,
    baseline,
    subsample: bool,
    bounds: tuple[pd.Timestamp | None, pd.Timestamp | None],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Bangun fitur, label, dan kolom penyerta untuk satu tahun."""
    from backend.app.data.synthetic import load_synthetic
    from backend.app.features.baseline import assign_load_zone
    from backend.app.features.engineering import build_features

    raw = load_synthetic(settings, years=[year])
    raw["load_zone"] = assign_load_zone(raw, settings)

    features = build_features(raw, settings, baseline=baseline)
    labels = build_labels(raw["event_start"], settings.model["horizons"])
    carry = raw[[c for c in CARRY_COLUMNS if c in raw.columns]].copy()

    # Baris saat unit berhenti tidak mengandung informasi tentang risiko
    # blocking — tidak ada pembakaran yang sedang berlangsung.
    keep = (raw["is_running"] == 1).to_numpy(copy=True)

    stamps = pd.to_datetime(raw["timestamp"])
    lower, upper = bounds
    if lower is not None:
        keep &= (stamps >= lower).to_numpy()
    if upper is not None:
        keep &= (stamps <= upper).to_numpy()

    if subsample:
        config = settings.model["sampling"]
        rate = float(config["negative_subsample_rate"])
        buffer_minutes = int(config["exclusion_buffer_minutes"])

        any_positive = np.zeros(len(raw), dtype=bool)
        for values in labels.values():
            any_positive |= values.astype(bool)

        # Jendela penyangga di tepi label: menit yang tepat berada di
        # perbatasan positif-negatif punya label ambigu dan lebih baik
        # dibuang daripada mengajari model batas yang salah.
        margin = (
            pd.Series(any_positive.astype(float))
            .rolling(buffer_minutes * 2 + 1, min_periods=1, center=True)
            .max()
            .to_numpy()
            .astype(bool)
        )
        ambiguous = margin & ~any_positive
        negative_keep = rng.random(len(raw)) < rate
        keep &= any_positive | (negative_keep & ~ambiguous)

    features = features.loc[keep].reset_index(drop=True)
    carry = carry.loc[keep].reset_index(drop=True)
    labels = {name: values[keep] for name, values in labels.items()}

    del raw, stamps
    gc.collect()
    return features, labels, carry


def assemble(
    settings: Settings,
    baseline,
    plan: SplitPlan,
    name: str,
    subsample: bool,
    stream_to_disk: bool,
) -> Dataset:
    """Susun satu himpunan data dari beberapa tahun.

    Bila ``stream_to_disk`` benar, fitur ditulis ke Parquet dan tidak
    ditahan di memori. Label dan kolom penyerta tetap di memori karena
    ukurannya kecil dan dibutuhkan berulang kali.
    """
    rng = np.random.default_rng(int(settings.model["sampling"]["random_state"]))
    bounds = plan.bounds(name)
    years = plan.years(name)

    writer = None
    path = settings.paths.processed / f"dataset_{name}.parquet"
    feature_parts: list[pd.DataFrame] = []
    carry_parts: list[pd.DataFrame] = []
    label_parts: dict[str, list[np.ndarray]] = {}
    feature_names: list[str] = []
    total_rows = 0

    try:
        for year in years:
            features, labels, carry = build_year_dataset(
                year, settings, baseline, subsample, bounds, rng
            )
            if not len(features):
                LOGGER.info("%s %d: tidak ada baris setelah penyaringan.", name, year)
                continue

            if not feature_names:
                feature_names = list(features.columns)

            if stream_to_disk:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pandas(features, preserve_index=False)
                if writer is None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(path, table.schema, compression="snappy")
                writer.write_table(table)
                del table
            else:
                feature_parts.append(features)

            carry_parts.append(carry)
            for label_name, values in labels.items():
                label_parts.setdefault(label_name, []).append(values)
            total_rows += len(features)

            LOGGER.info(
                "%s %d: %d baris, positif %s",
                name,
                year,
                len(features),
                {k: int(v.sum()) for k, v in labels.items()},
            )
            del features, labels, carry
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if not carry_parts:
        raise ValueError(f"Himpunan {name} kosong setelah penyaringan waktu.")

    dataset = Dataset(
        name=name,
        labels={
            label_name: np.concatenate(parts)
            for label_name, parts in label_parts.items()
        },
        carry=pd.concat(carry_parts, ignore_index=True),
        features=(
            pd.concat(feature_parts, ignore_index=True) if feature_parts else None
        ),
        path=path if stream_to_disk else None,
        row_count=total_rows,
        feature_names=feature_names,
    )
    LOGGER.info(
        "%s siap: %d baris%s",
        name,
        total_rows,
        f", dialirkan dari {path.name}" if stream_to_disk else " (di memori)",
    )
    return dataset


# --------------------------------------------------------------------
# Pelatihan
# --------------------------------------------------------------------
def _build_estimator(name: str, params: dict[str, Any]):
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params)
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(**params))]
        )
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(**params)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params)
    raise ValueError(f"Model tidak dikenal: {name}")


def _freeze(estimator):
    """Bungkus model terlatih agar kalibrator tidak melatihnya ulang.

    scikit-learn 1.6 menghapus ``cv="prefit"`` dan menggantinya dengan
    ``FrozenEstimator``. Keduanya ditangani supaya modul ini berjalan pada
    kedua rentang versi.
    """
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # pragma: no cover - scikit-learn < 1.6
        return estimator
    return FrozenEstimator(estimator)


def _prepare_matrix(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Susun matriks masukan model, dengan nilai hilang diisi nol.

    Nol berarti "tidak menyimpang" pada kolom simpangan, dan berada di
    tengah rentang untuk kolom lain setelah penskalaan. Model berbasis
    pohon menanganinya tanpa masalah; regresi logistik memakai penskala
    di dalam pipeline-nya sendiri.
    """
    # ``copy=True`` wajib: potongan yang dibaca dari Parquet didukung buffer
    # Arrow yang hanya-baca, sehingga pembersihan nilai hilang di tempat akan
    # ditolak.
    matrix = features.reindex(columns=columns).to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def predict_dataset(model, dataset: Dataset, columns: list[str]) -> np.ndarray:
    """Hitung probabilitas untuk seluruh himpunan, potongan demi potongan."""
    outputs: list[np.ndarray] = []
    for chunk in dataset.batches():
        outputs.append(model.predict_proba(_prepare_matrix(chunk, columns))[:, 1])
    if not outputs:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(outputs)


def stratified_positions(
    labels: np.ndarray, limit: int, rng: np.random.Generator
) -> np.ndarray:
    """Pilih posisi baris untuk pemasangan kalibrator.

    Seluruh baris positif dipertahankan — jumlahnya sedikit dan setiap
    satunya berharga. Baris negatif diambil acak sampai batas jumlah.
    """
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    budget = max(limit - positive.size, limit // 2)
    if negative.size > budget:
        negative = rng.choice(negative, budget, replace=False)
    return np.sort(np.concatenate([positive, negative]))


def train_anomaly_detector(train: Dataset, columns: list[str], settings: Settings):
    """Latih Isolation Forest pada periode normal saja (README §12).

    Detektor anomali harus belajar seperti apa kondisi normal. Melatihnya
    pada data yang berisi event justru mengajarkan bahwa gangguan itu
    normal.
    """
    config = settings.model["models"]["isolation_forest"]
    if not config.get("enabled", True):
        return None

    from sklearn.ensemble import IsolationForest

    horizon = str(settings.model["primary_horizon"])
    normal = np.flatnonzero(train.labels[horizon] == 0)
    rng = np.random.default_rng(42)
    if normal.size > ANOMALY_SAMPLE_LIMIT:
        normal = np.sort(rng.choice(normal, ANOMALY_SAMPLE_LIMIT, replace=False))

    matrix = _prepare_matrix(train.sample(normal), columns)
    LOGGER.info("Isolation Forest dilatih pada %d baris normal.", len(matrix))

    detector = IsolationForest(**config["params"])
    detector.fit(matrix)
    return detector


def anomaly_score(detector, features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Ubah skor Isolation Forest menjadi skala 0-100, makin tinggi makin aneh."""
    if detector is None:
        return np.zeros(len(features), dtype=np.float32)
    raw = detector.score_samples(_prepare_matrix(features, columns))
    # score_samples: makin negatif makin anomali. Dibalik dan dinormalkan.
    inverted = -raw
    low, high = np.percentile(inverted, [1, 99])
    span = max(high - low, 1e-9)
    return np.clip(100.0 * (inverted - low) / span, 0.0, 100.0).astype(np.float32)


def train_horizon(
    horizon: str,
    train: Dataset,
    validation: Dataset,
    test: Dataset,
    columns: list[str],
    settings: Settings,
) -> tuple[list[EvaluationResult], dict[str, Any]]:
    """Latih dan evaluasi seluruh model untuk satu horizon prediksi."""
    from sklearn.calibration import CalibratedClassifierCV

    calibration = settings.model["calibration"]
    results: list[EvaluationResult] = []
    artifacts: dict[str, Any] = {}

    train_labels = train.labels[horizon]
    validation_labels = validation.labels[horizon]
    test_labels = test.labels[horizon]

    LOGGER.info(
        "%s | latih %d (positif %d, prevalensi %.3f%%) | "
        "validasi %d (positif %d, prevalensi %.3f%%) | "
        "uji %d (positif %d, prevalensi %.3f%%)",
        horizon,
        len(train),
        int(train_labels.sum()),
        100 * train_labels.mean(),
        len(validation),
        int(validation_labels.sum()),
        100 * validation_labels.mean(),
        len(test),
        int(test_labels.sum()),
        100 * test_labels.mean(),
    )

    if train_labels.sum() < 10:
        LOGGER.warning("%s: sampel positif latih terlalu sedikit; dilewati.", horizon)
        return results, artifacts

    train_matrix = _prepare_matrix(train.features, columns)

    rng = np.random.default_rng(7)
    calibration_positions = stratified_positions(
        validation_labels, CALIBRATION_SAMPLE_LIMIT, rng
    )
    calibration_matrix = _prepare_matrix(
        validation.sample(calibration_positions), columns
    )
    calibration_labels = validation_labels[calibration_positions]

    for name, config in settings.model["models"].items():
        if name == "isolation_forest" or not config.get("enabled", True):
            continue

        params = dict(config["params"])
        if name == "xgboost":
            negative = int((train_labels == 0).sum())
            positive = max(int(train_labels.sum()), 1)
            params["scale_pos_weight"] = negative / positive

        estimator = _build_estimator(name, params)
        estimator.fit(train_matrix, train_labels)

        model = estimator
        if calibration.get("enabled", True) and calibration_labels.sum() >= 20:
            # Kalibrasi dipasang pada data validasi, bukan data latih —
            # probabilitas dari model yang mengenal datanya sendiri selalu
            # terlalu percaya diri.
            model = CalibratedClassifierCV(
                _freeze(estimator), method=str(calibration["method"])
            )
            model.fit(calibration_matrix, calibration_labels)

        # Ambang dipilih di validasi yang UTUH: prevalensi dan kerapatan
        # waktunya harus sama dengan yang akan dihadapi di lapangan.
        validation_probability = predict_dataset(model, validation, columns)
        threshold, threshold_info = choose_threshold(
            validation.carry["timestamp"],
            validation_probability,
            validation.carry["event_start"],
            settings,
        )

        results.append(
            evaluate(
                model_name=name,
                horizon=horizon,
                dataset="validation",
                timestamps=validation.carry["timestamp"],
                probabilities=validation_probability,
                labels=validation_labels,
                event_starts=validation.carry["event_start"],
                threshold=threshold,
                settings=settings,
            )
        )
        del validation_probability

        test_probability = predict_dataset(model, test, columns)
        results.append(
            evaluate(
                model_name=name,
                horizon=horizon,
                dataset="test",
                timestamps=test.carry["timestamp"],
                probabilities=test_probability,
                labels=test_labels,
                event_starts=test.carry["event_start"],
                threshold=threshold,
                settings=settings,
            )
        )
        del test_probability
        gc.collect()

        artifacts[name] = {
            "model": model,
            "threshold": threshold,
            "threshold_note": threshold_info["note"],
            "params": params,
        }
        LOGGER.info("  %-20s ambang %.5f — %s", name, threshold, threshold_info["note"])

    return results, artifacts


# --------------------------------------------------------------------
# Alur utama
# --------------------------------------------------------------------
def run(
    settings: Settings | None = None, horizons: list[str] | None = None
) -> dict[str, Any]:
    """Jalankan pelatihan penuh dan tulis seluruh artefak."""
    import joblib

    from backend.app.features.engineering import feature_columns

    settings = settings or get_settings()
    settings.paths.ensure()

    plan = build_split_plan(settings)
    LOGGER.info(
        "Pembagian waktu — latih %s | validasi %s | uji %s (embargo %s jam)",
        plan.train_years,
        plan.validation_years,
        plan.test_years,
        settings.model["split"]["embargo_hours"],
    )

    baseline = fit_training_baseline(settings, plan.train_years)
    baseline_path = baseline.save(settings.paths.models / "baseline_load_zone.json")
    LOGGER.info("Baseline disimpan: %s", baseline_path)

    train = assemble(
        settings, baseline, plan, "train", subsample=True, stream_to_disk=False
    )
    validation = assemble(
        settings, baseline, plan, "validation", subsample=False, stream_to_disk=True
    )
    test = assemble(
        settings, baseline, plan, "test", subsample=False, stream_to_disk=True
    )

    columns = feature_columns(train.features)
    LOGGER.info("Kolom fitur: %d", len(columns))

    detector = train_anomaly_detector(train, columns, settings)
    if detector is not None:
        joblib.dump(detector, settings.paths.models / "isolation_forest.joblib")

    wanted = horizons or [h["name"] for h in settings.model["horizons"]]
    all_results: list[EvaluationResult] = []
    registry = ModelRegistry.load(settings)
    registry.metadata = {
        "data_source": DATA_SOURCE_SYNTHETIC,
        "warning": SYNTHETIC_METRIC_WARNING,
        "train_years": plan.train_years,
        "validation_years": plan.validation_years,
        "test_years": plan.test_years,
        "feature_count": len(columns),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    for horizon in wanted:
        results, artifacts = train_horizon(
            horizon, train, validation, test, columns, settings
        )
        all_results.extend(results)

        for name, artifact in artifacts.items():
            path = settings.paths.models / f"{name}_{horizon}.joblib"
            joblib.dump(
                {
                    "model": artifact["model"],
                    "columns": columns,
                    "threshold": artifact["threshold"],
                },
                path,
            )
            test_result = next(
                (
                    result
                    for result in results
                    if result.model_name == name and result.dataset == "test"
                ),
                None,
            )
            registry.add(
                ModelEntry(
                    name=name,
                    horizon=horizon,
                    version="0.1.0",
                    data_source=DATA_SOURCE_SYNTHETIC,
                    artifact_path=str(path),
                    feature_count=len(columns),
                    train_rows=len(train),
                    trained_at=datetime.now(timezone.utc).isoformat(),
                    threshold=artifact["threshold"],
                    metrics=(
                        {**test_result.sample_metrics, **test_result.event_metrics}
                        if test_result
                        else {}
                    ),
                    parameters=artifact["params"],
                    notes=[artifact["threshold_note"]],
                )
            )
        gc.collect()

    registry.save()

    table = comparison_table(all_results)
    table_path = settings.paths.reports / "model_evaluation.csv"
    table.to_csv(table_path, index=False)

    columns_path = settings.paths.models / "feature_columns.json"
    columns_path.write_text(json.dumps(columns, indent=2), encoding="utf-8")

    return {
        "results": all_results,
        "table": table,
        "table_path": table_path,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "feature_count": len(columns),
        "train_years": plan.train_years,
        "validation_years": plan.validation_years,
        "test_years": plan.test_years,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pelatihan model risiko blocking furnace Unit 1"
    )
    parser.add_argument(
        "--horizon",
        action="append",
        help="Latih hanya horizon tertentu; boleh diulang",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    outcome = run(settings, args.horizon)

    print()
    print("=" * 78)
    print("EVALUASI MODEL — PLTU JERANJANG UNIT 1")
    print("=" * 78)
    print(SYNTHETIC_METRIC_WARNING)
    print()
    print(f"Tahun latih    : {outcome['train_years']}")
    print(f"Tahun validasi : {outcome['validation_years']}")
    print(f"Tahun uji      : {outcome['test_years']}")
    print(
        f"Baris          : latih {outcome['train_rows']:,} | "
        f"validasi {outcome['validation_rows']:,} | uji {outcome['test_rows']:,}"
    )
    print(f"Jumlah fitur   : {outcome['feature_count']}")
    print()

    table = outcome["table"]
    show = table[table["dataset"] == "test"] if "dataset" in table.columns else table
    display_columns = [
        c
        for c in (
            "model",
            "horizon",
            "pr_auc",
            "roc_auc",
            "precision",
            "recall",
            "event_detection_rate",
            "false_alarms_per_day",
            "median_warning_horizon_minutes",
            "brier_score",
            "calibration_error",
        )
        if c in show.columns
    ]
    print("Himpunan UJI:")
    print(show[display_columns].to_string(index=False))
    print()
    print(f"Tabel lengkap: {outcome['table_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
