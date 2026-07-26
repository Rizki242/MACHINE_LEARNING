"""Klasifikasi otomatis event gangguan berbahasa Indonesia.

Mengubah teks bebas kolom GANGGUAN / PENYEBAB / EQUIPMENT menjadi
``event_type``, ``event_location``, dan ``severity`` terstruktur sesuai
README §6 dan §15.

Dua lapis
---------
Lapis 1 — aturan kata kunci dari ``config/event_taxonomy.yaml``.
    Deterministik, dapat ditelusuri, dan dapat diubah engineer tanpa
    menyentuh Python. Inilah label yang dipakai hilir.

Lapis 2 — TF-IDF karakter n-gram + Logistic Regression.
    Dilatih pada label lapis 1 sebagai *weak supervision*. Analyzer
    karakter dipilih karena data penuh salah ketik: ``funace``,
    ``pembebeban``, ``batuabra``, ``indikiasi``, ``slaging``,
    ``Cute Coal Feeder``. Tugasnya bukan menggantikan lapis 1, melainkan
    menemukan baris yang lolos dari kata kunci.

Setiap ketidaksepakatan antar lapis ditandai ``needs_review`` dan
diekspor ke ``backend/reports/events_needing_review.csv`` untuk
diverifikasi engineer. Tidak ada baris yang dibuang diam-diam.

Penggunaan
----------
    python -m backend.app.rules.event_classifier
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.constants import (
    EVENT_TYPE_UNCLASSIFIED,
    EVENT_TYPE_UNRELATED,
    RECORD_KIND_OUTAGE,
)
from backend.app.core.logging import get_logger

LOGGER = get_logger(__name__)

METHOD_KEYWORD = "keyword"
METHOD_EQUIPMENT = "equipment_fallback"
METHOD_TEXT_MODEL = "text_model"
METHOD_NONE = "unmatched"


@dataclass
class KeywordVerdict:
    """Hasil pencocokan kata kunci untuk satu baris."""

    event_type: str
    score: float
    method: str
    matched_terms: list[str]


# --------------------------------------------------------------------
# Lapis 1 — aturan kata kunci
# --------------------------------------------------------------------
def _searchable(*parts: Any) -> str:
    """Gabungkan beberapa kolom teks menjadi satu haystack huruf kecil."""
    chunks = [str(part) for part in parts if part is not None and pd.notna(part)]
    return " ".join(chunks).casefold()


def match_keywords(
    gangguan: str | None,
    penyebab: str | None,
    equipment_raw: str | None,
    equipment_canonical: str | None,
    taxonomy: dict[str, Any],
) -> KeywordVerdict:
    """Tentukan jenis event dari kata kunci.

    Kata kunci pada ``keywords`` dicocokkan ke teks gangguan dan
    equipment; ``cause_keywords`` dicocokkan ke teks penyebab. Setiap
    kecocokan menambah skor sebesar bobot jenis event, dikali panjang
    frasa supaya frasa spesifik mengalahkan frasa umum.

    Equipment yang tidak pernah relevan untuk blocking furnace (CWP,
    condenser, turbin, generator, dan sejenisnya) langsung diberi label
    ``not_furnace_related``, kecuali teksnya menyebut furnace, bed, atau
    feeder secara eksplisit.
    """
    event_types = taxonomy.get("event_types") or {}
    non_furnace = set(taxonomy.get("non_furnace_equipment") or [])

    symptom_text = _searchable(gangguan, equipment_raw)
    cause_text = _searchable(penyebab)
    full_text = f"{symptom_text} {cause_text}"

    best: KeywordVerdict | None = None
    for name, spec in event_types.items():
        weight = float(spec.get("weight", 50))
        matched: list[str] = []
        score = 0.0

        for keyword in spec.get("keywords", []) or []:
            term = str(keyword).casefold()
            if term in symptom_text:
                score += weight * len(term)
                matched.append(str(keyword))

        for keyword in spec.get("cause_keywords", []) or []:
            term = str(keyword).casefold()
            if term in cause_text:
                # Penyebab bersifat menguatkan, bobotnya separuh gejala.
                score += 0.5 * weight * len(term)
                matched.append(str(keyword))

        if score > 0 and (best is None or score > best.score):
            best = KeywordVerdict(name, score, METHOD_KEYWORD, matched)

    # Equipment yang jelas di luar lingkup furnace mengalahkan kecocokan
    # kata kunci yang lemah — mis. "cleaning condensor" tidak boleh
    # terbaca sebagai gangguan bed hanya karena kata "cleaning".
    if equipment_canonical in non_furnace:
        explicit_furnace = any(
            term in full_text
            for term in ("furnace", "funace", "bed", "coal feeder", "coalfeeder",
                         "slag", "aglomerasi", "agglomerasi", "slagging")
        )
        if not explicit_furnace:
            return KeywordVerdict(
                EVENT_TYPE_UNRELATED, 1.0, METHOD_EQUIPMENT, []
            )

    if best is not None:
        return best

    if equipment_canonical in non_furnace:
        return KeywordVerdict(EVENT_TYPE_UNRELATED, 1.0, METHOD_EQUIPMENT, [])

    return KeywordVerdict(EVENT_TYPE_UNCLASSIFIED, 0.0, METHOD_NONE, [])


def match_location(
    gangguan: str | None,
    penyebab: str | None,
    equipment_raw: str | None,
    event_type: str,
    taxonomy: dict[str, Any],
) -> str | None:
    """Tentukan lokasi terduga dari teks; jatuh ke lokasi bawaan jenis event."""
    haystack = _searchable(gangguan, penyebab, equipment_raw)
    for location, terms in (taxonomy.get("location_keywords") or {}).items():
        for term in terms or []:
            if str(term).casefold() in haystack:
                return str(location)

    spec = (taxonomy.get("event_types") or {}).get(event_type) or {}
    return spec.get("default_location")


def derive_severity(
    status_raw: str | None,
    duration_hours: float | None,
    derating_mw: float | None,
    record_kind: str,
    event_type: str,
    taxonomy: dict[str, Any],
) -> int:
    """Turunkan severity 0-4 (README §15) dari status, durasi, dan derating.

    Aturan dievaluasi berurutan dari ``event_taxonomy.yaml``; yang
    pertama cocok menang. Jenis event yang tidak menandakan gangguan
    peralatan — di luar lingkup furnace, pembatasan pasokan batubara,
    penormalan pasca sinkron — diberi 0. Bukan karena ringan, melainkan
    karena tidak relevan untuk risiko blocking yang dimodelkan sistem ini.
    """
    zero_types = set(taxonomy.get("zero_severity_event_types") or [EVENT_TYPE_UNRELATED])
    if event_type in zero_types:
        return 0

    status = (status_raw or "").upper().strip()
    duration = float(duration_hours) if duration_hours is not None and pd.notna(duration_hours) else 0.0
    derating = float(derating_mw) if derating_mw is not None and pd.notna(derating_mw) else 0.0

    for rule in taxonomy.get("severity_rules") or []:
        wanted_kind = rule.get("match_record_kind")
        if wanted_kind and record_kind != wanted_kind:
            continue
        prefixes = tuple(rule.get("match_status_prefix") or ())
        if prefixes and not status.startswith(prefixes):
            continue
        if duration < float(rule.get("min_duration_hours", 0.0)):
            continue
        if "min_derating_mw" in rule and derating < float(rule["min_derating_mw"]):
            continue
        return int(rule["severity"])

    # Jaring pengaman bila YAML tidak punya aturan default.
    return 3 if record_kind == RECORD_KIND_OUTAGE else 2


def apply_keyword_layer(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Jalankan lapis 1 pada seluruh baris registry."""
    taxonomy = settings.event_taxonomy
    result = frame.copy()

    event_types: list[str] = []
    locations: list[str | None] = []
    severities: list[int] = []
    methods: list[str] = []
    matched_terms: list[str] = []

    for row in result.itertuples(index=False):
        verdict = match_keywords(
            row.initial_symptom,
            row.notes,
            row.equipment_raw,
            row.equipment_canonical,
            taxonomy,
        )
        location = match_location(
            row.initial_symptom,
            row.notes,
            row.equipment_raw,
            verdict.event_type,
            taxonomy,
        )
        severity = derive_severity(
            row.status_raw,
            row.duration_hours,
            row.derating_mw,
            row.record_kind,
            verdict.event_type,
            taxonomy,
        )

        event_types.append(verdict.event_type)
        locations.append(location)
        severities.append(severity)
        methods.append(verdict.method)
        matched_terms.append("; ".join(verdict.matched_terms))

    result["event_type"] = event_types
    result["event_location"] = locations
    result["severity"] = severities
    result["classification_method"] = methods
    result["matched_terms"] = matched_terms
    return result


# --------------------------------------------------------------------
# Lapis 2 — model teks
# --------------------------------------------------------------------
def _build_text_pipeline(config: dict[str, Any]):
    """Rakit pipeline TF-IDF karakter n-gram + Logistic Regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    vectoriser = TfidfVectorizer(
        analyzer=str(config.get("analyzer", "char_wb")),
        ngram_range=tuple(config.get("ngram_range", (3, 5))),
        min_df=int(config.get("min_df", 2)),
        max_features=int(config.get("max_features", 20000)),
        sublinear_tf=bool(config.get("sublinear_tf", True)),
        lowercase=True,
    )
    classifier = LogisticRegression(
        C=float(config.get("C", 4.0)),
        class_weight=config.get("class_weight", "balanced"),
        max_iter=int(config.get("max_iter", 2000)),
        random_state=int(config.get("random_state", 42)),
    )
    return Pipeline([("tfidf", vectoriser), ("clf", classifier)])


def _corpus(frame: pd.DataFrame) -> pd.Series:
    """Susun teks masukan model dari gejala, penyebab, dan equipment."""
    parts = [
        frame["initial_symptom"].fillna(""),
        frame["notes"].fillna(""),
        frame["equipment_raw"].fillna(""),
    ]
    return (parts[0] + " || " + parts[1] + " || " + parts[2]).astype(str)


def apply_text_layer(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Latih lapis 2 pada label lapis 1 lalu tandai ketidaksepakatan.

    Model ini sengaja dilatih dan diprediksi pada data yang sama. Ia
    bukan alat pengukur akurasi — ia alat pencari baris yang polanya
    mirip kelas tertentu tetapi tidak tertangkap kata kunci. Label lapis
    1 tetap yang dipakai hilir; lapis 2 hanya menaikkan bendera.
    """
    config = settings.event_taxonomy.get("text_model") or {}
    threshold = float(config.get("min_confidence_to_challenge", 0.60))
    result = frame.copy()

    trainable = result["event_type"] != EVENT_TYPE_UNCLASSIFIED
    label_counts = result.loc[trainable, "event_type"].value_counts()
    usable_labels = label_counts[label_counts >= 3].index

    if len(usable_labels) < 2:
        LOGGER.warning(
            "Lapis 2 dilewati: hanya %d kelas dengan minimal 3 contoh.",
            len(usable_labels),
        )
        result["text_model_prediction"] = pd.NA
        result["classification_confidence"] = np.where(
            result["event_type"] == EVENT_TYPE_UNCLASSIFIED, 0.0, 1.0
        )
        result["needs_review"] = result["event_type"] == EVENT_TYPE_UNCLASSIFIED
        return result

    train_mask = trainable & result["event_type"].isin(usable_labels)
    pipeline = _build_text_pipeline(config)
    pipeline.fit(_corpus(result.loc[train_mask]), result.loc[train_mask, "event_type"])

    probabilities = pipeline.predict_proba(_corpus(result))
    classes = list(pipeline.named_steps["clf"].classes_)
    best_index = probabilities.argmax(axis=1)
    predictions = [classes[i] for i in best_index]
    confidences = probabilities[np.arange(len(result)), best_index]

    result["text_model_prediction"] = predictions
    result["text_model_confidence"] = confidences

    keyword_labels = result["event_type"]
    disagreement = (
        (keyword_labels != pd.Series(predictions, index=result.index))
        & (confidences >= threshold)
    )
    unmatched = keyword_labels == EVENT_TYPE_UNCLASSIFIED

    # Baris tanpa kecocokan kata kunci mengambil usulan lapis 2 yang
    # cukup yakin, tetapi selalu ditandai untuk verifikasi.
    adopt = unmatched & (confidences >= threshold)
    result.loc[adopt, "event_type"] = pd.Series(predictions, index=result.index)[adopt]
    result.loc[adopt, "classification_method"] = METHOD_TEXT_MODEL

    result["needs_review"] = (disagreement | unmatched).to_numpy()
    result["classification_confidence"] = np.where(
        result["classification_method"] == METHOD_KEYWORD,
        1.0,
        confidences,
    )

    LOGGER.info(
        "Lapis 2: %d kelas terlatih, %d baris diadopsi, %d baris perlu tinjauan",
        len(usable_labels),
        int(adopt.sum()),
        int(result["needs_review"].sum()),
    )
    return result


# --------------------------------------------------------------------
# Alur utama
# --------------------------------------------------------------------
def classify_registry(
    frame: pd.DataFrame, settings: Settings | None = None
) -> pd.DataFrame:
    """Jalankan kedua lapis klasifikasi pada registry hasil ETL."""
    settings = settings or get_settings()
    result = apply_keyword_layer(frame, settings)
    result = apply_text_layer(result, settings)

    # Lokasi dan severity dihitung ulang untuk baris yang jenisnya
    # berubah akibat adopsi lapis 2.
    adopted = result["classification_method"] == METHOD_TEXT_MODEL
    if adopted.any():
        taxonomy = settings.event_taxonomy
        for index in result.index[adopted]:
            row = result.loc[index]
            result.at[index, "event_location"] = match_location(
                row["initial_symptom"],
                row["notes"],
                row["equipment_raw"],
                row["event_type"],
                taxonomy,
            )
            result.at[index, "severity"] = derive_severity(
                row["status_raw"],
                row["duration_hours"],
                row["derating_mw"],
                row["record_kind"],
                row["event_type"],
                taxonomy,
            )

    return result


def blocking_events(
    registry: pd.DataFrame, settings: Settings | None = None
) -> pd.DataFrame:
    """Ambil hanya event yang relevan untuk risiko blocking furnace."""
    settings = settings or get_settings()
    types = settings.event_taxonomy.get("blocking_event_types") or []
    return registry.loc[registry["event_type"].isin(types)].copy()


def coverage_summary(registry: pd.DataFrame) -> pd.DataFrame:
    """Ringkas cakupan klasifikasi per jenis catatan sumber."""
    return (
        registry.pivot_table(
            index="event_type",
            columns="record_kind",
            values="event_id",
            aggfunc="count",
            fill_value=0,
        )
        .assign(total=lambda f: f.sum(axis=1))
        .sort_values("total", ascending=False)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Klasifikasi event gangguan PLTU Jeranjang Unit 1"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Jalankan ETL dari XLSX terlebih dahulu, bukan memuat hasil olahan",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.paths.ensure()

    if args.rebuild:
        from backend.app.data.event_etl import build_event_registry, write_registry

        registry, _ = build_event_registry(settings)
        write_registry(registry, settings)
    else:
        from backend.app.data.event_etl import load_registry

        registry = load_registry(settings)

    summary = coverage_summary(registry)
    blocking = blocking_events(registry, settings)

    print()
    print("=" * 72)
    print("KLASIFIKASI EVENT — PLTU JERANJANG UNIT 1")
    print("=" * 72)
    print(summary.to_string())
    print()
    print(f"Total event                    : {len(registry)}")
    print(f"Event relevan blocking furnace : {len(blocking)}")
    print(f"Event di luar lingkup furnace  : {int((registry['event_type'] == EVENT_TYPE_UNRELATED).sum())}")
    print(f"Belum terklasifikasi           : {int((registry['event_type'] == EVENT_TYPE_UNCLASSIFIED).sum())}")
    print()
    print("Metode klasifikasi:")
    for method, count in registry["classification_method"].value_counts().items():
        print(f"  {method:<20}: {count}")
    print()
    print("Severity (README §15):")
    for severity, count in registry["severity"].value_counts().sort_index().items():
        print(f"  {severity}: {count}")

    review = registry.loc[registry["needs_review"].fillna(False).astype(bool)]
    review_path = settings.paths.reports / "events_needing_review.csv"
    columns = [
        "event_id",
        "start_time",
        "record_kind",
        "equipment_raw",
        "initial_symptom",
        "notes",
        "event_type",
        "event_location",
        "severity",
        "classification_method",
        "classification_confidence",
    ]
    review[[c for c in columns if c in review.columns]].to_csv(
        review_path, index=False, encoding="utf-8"
    )
    print()
    print(f"Perlu tinjauan engineer: {len(review)} event -> {review_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
