"""Pemuat konfigurasi terpusat.

Seluruh berkas YAML di ``config/`` dimuat sekali lewat :func:`get_settings`
yang di-cache. Tidak ada modul lain yang boleh membaca YAML sendiri —
dengan begitu ada satu tempat untuk memvalidasi dan menelusuri asal nilai.

Jalur direktori dapat ditimpa lewat variabel lingkungan (lihat ``.env.example``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Akar proyek: backend/app/core/config.py -> naik empat tingkat
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: str) -> Path:
    """Ambil jalur dari environment, relatif terhadap akar proyek bila perlu."""
    raw = os.environ.get(name, default)
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Berkas konfigurasi tidak ditemukan: {path}. "
            "Pastikan direktori config/ lengkap."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Isi {path} bukan pemetaan YAML yang valid.")
    return data


@dataclass(frozen=True)
class Paths:
    """Kumpulan jalur direktori proyek."""

    root: Path
    raw_data: Path
    config: Path
    datasets: Path
    models: Path
    reports: Path

    @property
    def processed(self) -> Path:
        return self.datasets / "processed"

    @property
    def synthetic(self) -> Path:
        return self.datasets / "synthetic"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def ensure(self) -> None:
        """Buat direktori keluaran bila belum ada.

        Direktori data mentah tidak pernah dibuat — bila hilang, itu
        masalah yang harus dilaporkan, bukan ditutupi.
        """
        for directory in (
            self.processed,
            self.synthetic,
            self.models,
            self.reports,
            self.figures,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Seluruh konfigurasi FurnaceGuard AI dalam satu objek."""

    paths: Paths
    units: dict[str, Any]
    thresholds: dict[str, Any]
    tag_mapping: dict[str, Any]
    event_taxonomy: dict[str, Any]
    model: dict[str, Any]
    synthetic_seed: int = 20260726
    timezone: str = "Asia/Makassar"
    deployment_mode: str = "offline"
    log_level: str = "INFO"

    # ---------------- pembantu yang sering dipakai ----------------

    @property
    def load_zones(self) -> list[dict[str, Any]]:
        return self.units["load_zones"]

    @property
    def plausible_ranges(self) -> dict[str, list[float]]:
        return self.units["plausible_ranges"]

    @property
    def risk_bands(self) -> list[dict[str, Any]]:
        return self.thresholds["risk_bands"]

    def load_zone_for(self, load_mw: float) -> str:
        """Kembalikan nama zona beban untuk suatu nilai beban.

        Batas bawah eksklusif kecuali zona pertama, sesuai README §10
        (``0-5``, ``>5-10``, ``>10-17``, ``>17-22``, ``>22-25``).
        Nilai di atas zona terakhir tetap dipetakan ke zona terakhir.
        """
        zones = self.load_zones
        for index, zone in enumerate(zones):
            lower = float(zone["min_mw"])
            upper = float(zone["max_mw"])
            if index == 0:
                if lower <= load_mw <= upper:
                    return str(zone["name"])
            elif lower < load_mw <= upper:
                return str(zone["name"])
        return str(zones[-1]["name"]) if load_mw > 0 else str(zones[0]["name"])

    def risk_band_for(self, score: float) -> dict[str, Any]:
        """Kembalikan definisi band risiko untuk suatu skor 0-100."""
        clamped = max(0.0, min(100.0, float(score)))
        for band in self.risk_bands:
            if float(band["min"]) <= clamped <= float(band["max"]):
                return band
        return self.risk_bands[-1]

    def standard_names(self, *priorities: str) -> list[str]:
        """Daftar ``standard_name`` dari tag_mapping untuk prioritas tertentu.

        Contoh: ``settings.standard_names("priority_a", "priority_b")``.
        """
        names: list[str] = []
        for priority in priorities:
            for entry in self.tag_mapping.get(priority, []) or []:
                names.append(str(entry["standard_name"]))
        return names

    def dcs_tag_lookup(self) -> dict[str, str]:
        """Pemetaan ``dcs_tag -> standard_name`` untuk tag yang sudah diisi.

        Selama seluruh tag masih ``null`` (Fase 0), hasilnya kosong dan
        satu-satunya sumber timeseries adalah generator sintetis.
        """
        lookup: dict[str, str] = {}
        for priority in ("priority_a", "priority_b", "priority_c"):
            for entry in self.tag_mapping.get(priority, []) or []:
                tag = entry.get("dcs_tag")
                if tag:
                    lookup[str(tag)] = str(entry["standard_name"])
        return lookup


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Muat dan cache seluruh konfigurasi proyek."""
    paths = Paths(
        root=PROJECT_ROOT,
        raw_data=_env_path("FG_RAW_DATA_DIR", "./data"),
        config=_env_path("FG_CONFIG_DIR", "./config"),
        datasets=_env_path("FG_DATASETS_DIR", "./backend/datasets"),
        models=_env_path("FG_MODELS_DIR", "./backend/models"),
        reports=_env_path("FG_REPORTS_DIR", "./backend/reports"),
    )

    settings = Settings(
        paths=paths,
        units=_load_yaml(paths.config / "units.yaml"),
        thresholds=_load_yaml(paths.config / "thresholds.yaml"),
        tag_mapping=_load_yaml(paths.config / "tag_mapping.yaml"),
        event_taxonomy=_load_yaml(paths.config / "event_taxonomy.yaml"),
        model=_load_yaml(paths.config / "model_config.yaml"),
        synthetic_seed=int(os.environ.get("FG_SYNTHETIC_SEED", "20260726")),
        timezone=os.environ.get("FG_TIMEZONE", "Asia/Makassar"),
        deployment_mode=os.environ.get("FG_DEPLOYMENT_MODE", "offline"),
        log_level=os.environ.get("FG_LOG_LEVEL", "INFO"),
    )

    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    """Pemeriksaan kewarasan konfigurasi saat pemuatan."""
    zones = settings.units.get("load_zones") or []
    if not zones:
        raise ValueError("units.yaml: load_zones kosong.")
    for previous, current in zip(zones, zones[1:]):
        if float(previous["max_mw"]) != float(current["min_mw"]):
            raise ValueError(
                "units.yaml: zona beban tidak bersambung — "
                f"{previous['name']}.max_mw={previous['max_mw']} tetapi "
                f"{current['name']}.min_mw={current['min_mw']}."
            )

    bands = settings.thresholds.get("risk_bands") or []
    if not bands:
        raise ValueError("thresholds.yaml: risk_bands kosong.")
    if float(bands[0]["min"]) != 0 or float(bands[-1]["max"]) != 100:
        raise ValueError("thresholds.yaml: risk_bands harus menutup rentang 0-100.")

    weights = settings.thresholds.get("hybrid_weights") or {}
    total = sum(float(value) for value in weights.values())
    if weights and abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"thresholds.yaml: hybrid_weights berjumlah {total}, seharusnya 1.0."
        )

    if settings.deployment_mode not in {"offline", "shadow", "advisory", "integrated"}:
        raise ValueError(
            f"FG_DEPLOYMENT_MODE tidak dikenal: {settings.deployment_mode!r}."
        )
