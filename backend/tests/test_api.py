"""Pengujian lapisan API FastAPI.

Untuk ``/events`` dan ``/models``, dependensi pemuat data ditimpa lewat
``app.dependency_overrides`` dengan data kecil buatan sendiri — sama seperti
``test_pipeline.py`` memakai dataframe kecil buatan sendiri, bukan berkas
hasil pipeline sungguhan — sehingga pengujian ini cepat dan tidak bergantung
pada berkas hasil generate.

Untuk ``/risk/assessment`` dan ``/risk/demo``, langkah pemuatan artefak
model (``.joblib`` terlatih) tidak praktis dipalsukan tanpa estimator
sungguhan, jadi pengujian dilewati (``pytest.skip``) bila artefak belum
ada — pola yang sama dipakai ``test_pipeline.py::test_split_is_temporal_and_embargoed``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api import deps
from backend.app.core.config import get_settings
from backend.app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


# --------------------------------------------------------------------
# Health
# --------------------------------------------------------------------
def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["deployment_mode"] == "offline"
    assert "disclaimer" in body


# --------------------------------------------------------------------
# Events — dependensi ditimpa, tidak menyentuh berkas sungguhan
# --------------------------------------------------------------------
def _fake_event_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["JRG-U1-BLK-001", "JRG-U1-BLK-002", "JRG-U1-ASH-003"],
            "unit_id": ["UNIT_1"] * 3,
            "start_time": pd.to_datetime(
                ["2024-01-01 08:00", "2024-02-01 08:00", "2024-03-01 08:00"]
            ),
            "end_time": pd.to_datetime(
                ["2024-01-01 10:00", "2024-02-01 09:00", "2024-03-01 09:00"]
            ),
            "event_type": ["main_bed_agglomeration", "coal_feeder_blocking", "bottom_ash_blocking"],
            "event_location": ["main_bed", "coal_feeder_2", "slag_pipe_2"],
            "severity": [3, 2, 4],
            "needs_review": [False, True, False],
        }
    )


def test_list_events_and_filter():
    app.dependency_overrides[deps.get_event_registry] = _fake_event_registry
    response = client.get("/api/v1/events", params={"needs_review": True})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["event_id"] == "JRG-U1-BLK-002"
    assert "disclaimer" in body


def test_list_events_min_severity():
    app.dependency_overrides[deps.get_event_registry] = _fake_event_registry
    response = client.get("/api/v1/events", params={"min_severity": 3})
    assert response.status_code == 200
    assert response.json()["total"] == 2


def test_get_event_by_id():
    app.dependency_overrides[deps.get_event_registry] = _fake_event_registry
    response = client.get("/api/v1/events/JRG-U1-BLK-001")
    assert response.status_code == 200
    assert response.json()["event_type"] == "main_bed_agglomeration"


def test_get_event_missing_returns_404():
    app.dependency_overrides[deps.get_event_registry] = _fake_event_registry
    response = client.get("/api/v1/events/DOES-NOT-EXIST")
    assert response.status_code == 404


# --------------------------------------------------------------------
# Models — dependensi ditimpa dengan registri kecil buatan sendiri
# --------------------------------------------------------------------
def _fake_model_registry():
    from backend.app.models.registry import ModelEntry, ModelRegistry

    registry = ModelRegistry(path=Path("unused.json"), require_data_source_label=False)
    registry.metadata = {"data_source": "synthetic"}
    registry.entries = [
        ModelEntry(
            name="xgboost",
            horizon="blocking_next_60m",
            version="0.1.0",
            data_source="synthetic",
            artifact_path="backend/models/xgboost_blocking_next_60m.joblib",
            feature_count=231,
            train_rows=100_000,
            trained_at="2026-01-01T00:00:00+00:00",
            threshold=0.02,
            metrics={"pr_auc": 0.5},
        ),
        ModelEntry(
            name="logistic_regression",
            horizon="blocking_next_30m",
            version="0.1.0",
            data_source="synthetic",
            artifact_path="backend/models/logistic_regression_blocking_next_30m.joblib",
            feature_count=231,
            train_rows=100_000,
            trained_at="2026-01-01T00:00:00+00:00",
            threshold=0.03,
            metrics={"pr_auc": 0.4},
        ),
    ]
    return registry


def test_list_models():
    app.dependency_overrides[deps.get_model_registry] = _fake_model_registry
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert response.json()["total"] == 2


def test_list_models_filter_by_horizon():
    app.dependency_overrides[deps.get_model_registry] = _fake_model_registry
    response = client.get("/api/v1/models", params={"horizon": "blocking_next_60m"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["name"] == "xgboost"


def test_best_model():
    app.dependency_overrides[deps.get_model_registry] = _fake_model_registry
    response = client.get(
        "/api/v1/models/best", params={"horizon": "blocking_next_60m", "metric": "pr_auc"}
    )
    assert response.status_code == 200
    assert response.json()["entry"]["name"] == "xgboost"


def test_best_model_no_match_returns_404():
    app.dependency_overrides[deps.get_model_registry] = _fake_model_registry
    response = client.get("/api/v1/models/best", params={"horizon": "blocking_next_180m"})
    assert response.status_code == 404


# --------------------------------------------------------------------
# Risk — validasi murni, tanpa I/O
# --------------------------------------------------------------------
def test_risk_assessment_rejects_unknown_model():
    response = client.get(
        "/api/v1/risk/assessment",
        params={"start": "2024-01-01T00:00:00", "end": "2024-01-01T01:00:00", "model": "not_a_model"},
    )
    assert response.status_code == 400


def test_risk_assessment_rejects_unknown_horizon():
    response = client.get(
        "/api/v1/risk/assessment",
        params={
            "start": "2024-01-01T00:00:00",
            "end": "2024-01-01T01:00:00",
            "horizon": "blocking_next_999m",
        },
    )
    assert response.status_code == 400


def test_risk_assessment_rejects_end_before_start():
    response = client.get(
        "/api/v1/risk/assessment",
        params={"start": "2024-01-01T02:00:00", "end": "2024-01-01T00:00:00"},
    )
    assert response.status_code == 400


def test_risk_assessment_rejects_span_too_wide():
    response = client.get(
        "/api/v1/risk/assessment",
        params={"start": "2024-01-01T00:00:00", "end": "2024-02-01T00:00:00"},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------
# Risk — jalur sungguhan, dilewati bila artefak belum ada
# --------------------------------------------------------------------
def _require_trained_model() -> None:
    settings = get_settings()
    path = settings.paths.models / "xgboost_blocking_next_60m.joblib"
    if not path.exists():
        pytest.skip(
            "Model belum dilatih. Jalankan: python -m backend.app.models.train"
        )
    if not (settings.paths.synthetic / "manifest.json").exists():
        pytest.skip(
            "Data sintetis belum dibangkitkan. Jalankan: "
            "python -m backend.app.data.synthetic"
        )


def test_risk_assessment_end_to_end():
    _require_trained_model()
    response = client.get(
        "/api/v1/risk/assessment",
        params={
            "start": "2024-03-01T00:00:00",
            "end": "2024-03-01T02:00:00",
            "model": "xgboost",
            "horizon": "blocking_next_60m",
            "top": 2,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "xgboost"
    assert len(body["series"]) > 0
    assert len(body["snapshots"]) <= 2
    for snapshot in body["snapshots"]:
        assert 0.0 <= snapshot["risk_score"] <= 100.0
        assert snapshot["status"] in {
            "Normal",
            "Early Warning",
            "Warning",
            "High Risk",
            "Critical",
        }
    assert "SINTETIS" in body["warning"]


def test_risk_demo_end_to_end():
    _require_trained_model()
    response = client.get("/api/v1/risk/demo", params={"model": "xgboost"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "event_time" in body
    assert len(body["series"]) > 0
