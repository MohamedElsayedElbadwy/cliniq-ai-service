"""MRI route contract tests against the central FastAPI application."""

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
        assert batch.shape == (1, 299, 299, 3)
        return [[0.05, 0.10, 0.80, 0.05]]


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_mri_service] = lambda: MRIPredictor(model_instance=MockKerasModel())
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _image_bytes(mode: str = "RGB") -> bytes:
    stream = io.BytesIO()
    Image.new(mode, (80, 50)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_predict_accepts_supported_image_modes(client: TestClient, mode: str) -> None:
    response = client.post(
        "/api/ai/mri/predict",
        files={"file": ("scan.png", _image_bytes(mode), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "notumor"
    assert body["confidence"] == pytest.approx(0.80)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert "diagnosis" not in body


def test_predict_rejects_unsupported_or_corrupt_uploads(client: TestClient) -> None:
    unsupported = client.post("/api/ai/mri/predict", files={"file": ("scan.gif", b"gif", "image/gif")})
    assert unsupported.status_code == 415

    empty = client.post("/api/ai/mri/predict", files={"file": ("scan.png", b"", "image/png")})
    assert empty.status_code == 422

    corrupt = client.post("/api/ai/mri/predict", files={"file": ("scan.png", b"not an image", "image/png")})
    assert corrupt.status_code == 422


def test_predict_rejects_uploads_over_the_size_limit(client: TestClient) -> None:
    response = client.post(
        "/api/ai/mri/predict",
        files={"file": ("large_scan.png", b"0" * (10 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413


def test_health_endpoint_is_exposed(client: TestClient) -> None:
    response = client.get("/api/ai/mri/health")
    assert response.status_code == 200
    assert set(response.json()) == {"status", "model_available", "message"}
