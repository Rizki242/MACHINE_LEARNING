"""Skema dan validasi struktur data FurnaceGuard AI.

Dipakai bersama oleh ETL event registry, generator sintetis, dan modul
kualitas data supaya bentuk data terjaga di seluruh pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.app.core.constants import EVENT_REGISTRY_COLUMNS

# Tipe kolom event registry. Kolom yang tidak disebut dibiarkan apa adanya.
EVENT_REGISTRY_DTYPES: dict[str, str] = {
    "event_id": "string",
    "unit_id": "string",
    "event_type": "string",
    "event_location": "string",
    "severity": "Int64",
    "initial_symptom": "string",
    "operator_action": "string",
    "maintenance_action": "string",
    "derating_mw": "Float64",
    "trip_status": "string",
    "clinker_found": "string",
    "coal_source": "string",
    "coal_blending": "string",
    "notes": "string",
    "source_file": "string",
    "source_sheet": "string",
    "source_row": "Int64",
    "equipment_raw": "string",
    "equipment_canonical": "string",
    "status_raw": "string",
    "record_kind": "string",
    "duration_hours": "Float64",
    "duration_hours_source": "Float64",
    "duration_mismatch_hours": "Float64",
    "mwh_lost": "Float64",
    "classification_method": "string",
    "classification_confidence": "Float64",
    "needs_review": "boolean",
}

DATETIME_COLUMNS = ("start_time", "end_time")


@dataclass
class ValidationIssue:
    """Satu temuan validasi.

    ``severity`` di sini adalah tingkat keseriusan temuan data, bukan
    severity event boiler.
    """

    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    count: int = 0
    sample: list[Any] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - untuk pembacaan log
        prefix = self.severity.upper()
        suffix = f" ({self.count} baris)" if self.count else ""
        return f"[{prefix}] {self.code}: {self.message}{suffix}"


@dataclass
class ValidationReport:
    """Kumpulan temuan validasi untuk satu dataset."""

    dataset: str
    row_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        count: int = 0,
        sample: list[Any] | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                count=count,
                sample=list(sample or []),
            )
        )

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_clean(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "row_count": self.row_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                    "count": issue.count,
                    "sample": issue.sample,
                }
                for issue in self.issues
            ],
        }

    def render(self) -> str:
        """Ringkasan teks untuk log dan laporan."""
        lines = [
            f"Validasi: {self.dataset}",
            f"  baris     : {self.row_count}",
            f"  error     : {len(self.errors)}",
            f"  peringatan: {len(self.warnings)}",
        ]
        lines.extend(f"  {issue}" for issue in self.issues)
        return "\n".join(lines)


def coerce_event_registry(frame: pd.DataFrame) -> pd.DataFrame:
    """Paksa dataframe ke skema event registry: kolom lengkap, urut, bertipe.

    Kolom yang hilang ditambahkan berisi ``NA`` — hilang secara eksplisit
    lebih baik daripada hilang secara diam-diam. Kolom asing dibuang.
    """
    result = frame.copy()

    for column in EVENT_REGISTRY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[EVENT_REGISTRY_COLUMNS]

    for column in DATETIME_COLUMNS:
        result[column] = pd.to_datetime(result[column], errors="coerce")

    for column, dtype in EVENT_REGISTRY_DTYPES.items():
        try:
            result[column] = result[column].astype(dtype)
        except (TypeError, ValueError):
            # Biarkan apa adanya; validator akan menandai masalahnya.
            pass

    return result


def validate_event_registry(frame: pd.DataFrame) -> ValidationReport:
    """Periksa kewarasan event registry hasil ETL."""
    report = ValidationReport(dataset="event_registry", row_count=len(frame))

    missing = [c for c in EVENT_REGISTRY_COLUMNS if c not in frame.columns]
    if missing:
        report.add("error", "missing_columns", f"Kolom hilang: {missing}")
        return report

    duplicated = frame["event_id"].duplicated()
    if duplicated.any():
        report.add(
            "error",
            "duplicate_event_id",
            "event_id tidak unik",
            count=int(duplicated.sum()),
            sample=frame.loc[duplicated, "event_id"].head(5).tolist(),
        )

    no_start = frame["start_time"].isna()
    if no_start.any():
        report.add(
            "error",
            "missing_start_time",
            "Event tanpa start_time",
            count=int(no_start.sum()),
        )

    both = frame["start_time"].notna() & frame["end_time"].notna()
    reversed_time = both & (frame["end_time"] < frame["start_time"])
    if reversed_time.any():
        report.add(
            "error",
            "end_before_start",
            "end_time mendahului start_time",
            count=int(reversed_time.sum()),
            sample=frame.loc[reversed_time, "event_id"].head(5).tolist(),
        )

    no_end = frame["end_time"].isna()
    if no_end.any():
        report.add(
            "warning",
            "missing_end_time",
            "Event tanpa end_time — durasi tidak dapat dihitung",
            count=int(no_end.sum()),
            sample=frame.loc[no_end, "event_id"].head(5).tolist(),
        )

    mismatch = frame["duration_mismatch_hours"].abs() > 1.0
    if mismatch.fillna(False).any():
        report.add(
            "warning",
            "duration_mismatch",
            "Durasi hasil hitung berbeda >1 jam dari kolom durasi sumber",
            count=int(mismatch.fillna(False).sum()),
            sample=frame.loc[mismatch.fillna(False), "event_id"].head(5).tolist(),
        )

    bad_severity = frame["severity"].notna() & ~frame["severity"].isin([0, 1, 2, 3, 4])
    if bad_severity.any():
        report.add(
            "error",
            "invalid_severity",
            "severity di luar rentang 0-4 (README §15)",
            count=int(bad_severity.sum()),
        )

    unclassified = frame["event_type"] == "unclassified"
    if unclassified.any():
        report.add(
            "warning",
            "unclassified_events",
            "Event tanpa jenis yang cocok — perlu tinjauan manual",
            count=int(unclassified.sum()),
            sample=frame.loc[unclassified, "event_id"].head(5).tolist(),
        )

    if "needs_review" in frame.columns:
        review = frame["needs_review"].fillna(False).astype(bool)
        if review.any():
            report.add(
                "info",
                "needs_review",
                "Event ditandai untuk verifikasi engineer",
                count=int(review.sum()),
            )

    return report
