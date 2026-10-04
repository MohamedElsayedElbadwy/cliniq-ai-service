"""Integration tests for MRI within the central FastAPI application."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app
from mri.router import get_mri_service
from mri.service import MRIPredictor


class MockKerasModel:
    def predict(self, batch, verbose: int = 0):
        return [[0.05, 0.05, 0.85, 0.05]]


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_mri_service] = lambda: MRIPredictor(model_instance=MockKerasModel())
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_mri_routes_mounted_in_main_app(client: TestClient) -> None:
    health_response = client.get("/api/ai/mri/health")
    assert health_response.status_code == 200
    assert set(health_response.json()) == {"status", "modelAvailable", "message"}

    stream = io.BytesIO()
    Image.new("RGB", (100, 100)).save(stream, format="PNG")
    prediction_response = client.post(
        "/api/ai/mri/predict",
        files={"file": ("test_scan.png", stream.getvalue(), "image/png")},
    )
    assert prediction_response.status_code == 200
    payload = prediction_response.json()
    assert payload["prediction"] == "notumor"
    assert payload["classIndex"] == 2
    assert payload["confidence"] == pytest.approx(0.85)
    assert "diagnosis" not in payload


def test_central_app_keeps_voice_and_mri_routes(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/api/ai/voice/turn" in schema["paths"]
    assert "/api/ai/mri/health" in schema["paths"]
    assert "/api/ai/mri/predict" in schema["paths"]
