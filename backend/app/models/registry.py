"""Registri artefak model.

README §21 mewajibkan versi model yang aktif dicatat, dan README §12
mewajibkan metriknya dilaporkan. Modul ini menyimpan keduanya bersama
artefaknya sehingga setiap angka unjuk kerja selalu dapat ditelusuri ke
model yang menghasilkannya.

Setiap entri WAJIB membawa label sumber data. Model yang dilatih di atas
data sintetis tidak boleh dapat disalahartikan sebagai model yang sudah
tervalidasi lapangan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings, get_settings
from backend.app.core.constants import SYNTHETIC_METRIC_WARNING
from backend.app.core.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ModelEntry:
    """Satu artefak model beserta asal-usul dan metriknya."""

    name: str
    horizon: str
    version: str
    data_source: str
    artifact_path: str
    feature_count: int
    train_rows: int
    trained_at: str
    threshold: float
    metrics: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "version": self.version,
            "data_source": self.data_source,
            "artifact_path": self.artifact_path,
            "feature_count": self.feature_count,
            "train_rows": self.train_rows,
            "trained_at": self.trained_at,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "notes": self.notes,
        }


class ModelRegistry:
    """Berkas JSON berisi seluruh model yang pernah dilatih."""

    def __init__(self, path: Path, require_data_source_label: bool = True) -> None:
        self.path = path
        self.require_data_source_label = require_data_source_label
        self.entries: list[ModelEntry] = []
        self.metadata: dict[str, Any] = {}

    @classmethod
    def load(cls, settings: Settings | None = None) -> ModelRegistry:
        settings = settings or get_settings()
        config = settings.model["registry"]
        path = settings.paths.root / config["metadata_file"]
        registry = cls(path, bool(config.get("require_data_source_label", True)))

        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            registry.metadata = payload.get("metadata", {})
            registry.entries = [
                ModelEntry(**entry) for entry in payload.get("models", [])
            ]
        return registry

    def add(self, entry: ModelEntry) -> None:
        if self.require_data_source_label and not entry.data_source:
            raise ValueError(
                "Entri model wajib membawa data_source. Model yang dilatih di "
                "atas data sintetis harus dapat dibedakan dari model yang "
                "dilatih di atas data operasi."
            )
        if entry.data_source == "synthetic" and SYNTHETIC_METRIC_WARNING not in entry.notes:
            entry.notes.append(SYNTHETIC_METRIC_WARNING)

        # Satu nama dan horizon hanya menyimpan versi terbaru.
        self.entries = [
            existing
            for existing in self.entries
            if not (existing.name == entry.name and existing.horizon == entry.horizon)
        ]
        self.entries.append(entry)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "models": [entry.to_dict() for entry in self.entries],
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        LOGGER.info("Registri model ditulis: %s (%d entri)", self.path, len(self.entries))
        return self.path

    def best(self, horizon: str, metric: str = "pr_auc") -> ModelEntry | None:
        """Ambil model terbaik untuk satu horizon menurut metrik tertentu."""
        candidates = [
            entry
            for entry in self.entries
            if entry.horizon == horizon and metric in entry.metrics
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda entry: entry.metrics[metric])
