"""Pengujian endpoint health API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_endpoint_returns_read_only_status():
    client = TestClient(create_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["deployment_mode"] == "offline"
    assert payload["timezone"] == "Asia/Makassar"
    assert "pendukung keputusan" in payload["disclaimer"]
