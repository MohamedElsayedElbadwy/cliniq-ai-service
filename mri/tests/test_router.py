"""MRI route contract tests against the central FastAPI application."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app
import mri.router as mri_router
from mri.router import get_mri_service
from mri.service import MRIPredictor


class MockKerasModel:
    output_shape = (None, 4)

    def predict(self, batch, verbose: int = 0):
        assert batch.shape == (1, 299, 299, 3)
        return [[0.05, 0.10, 0.80, 0.05]]


@pytest.fixture
def client(class_names_path) -> TestClient:
    app.dependency_overrides[get_mri_service] = lambda: MRIPredictor(
        model_instance=MockKerasModel(), class_names_path=class_names_path
    )
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
    assert body["classIndex"] == 2
    assert "class_index" not in body
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
    assert "Image processing failed" in corrupt.json()["detail"]


def test_predict_rejects_uploads_over_the_size_limit(client: TestClient) -> None:
    response = client.post(
        "/api/ai/mri/predict",
        files={"file": ("large_scan.png", b"0" * (10 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413


def test_predict_runs_in_threadpool(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def mock_run_in_threadpool(function, *args, **kwargs):
        nonlocal called
        called = True
        return function(*args, **kwargs)

    monkeypatch.setattr(mri_router, "run_in_threadpool", mock_run_in_threadpool)
    response = client.post(
        "/api/ai/mri/predict",
        files={"file": ("scan.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert called is True


def test_predict_returns_503_when_class_mapping_is_missing(tmp_path) -> None:
    app.dependency_overrides[get_mri_service] = lambda: MRIPredictor(
        model_instance=MockKerasModel(), class_names_path=tmp_path / "class_names.json"
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/mri/predict",
            files={"file": ("scan.png", _image_bytes(), "image/png")},
        )
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Class mapping metadata 'class_names.json' is missing." == response.json()["detail"]


def test_predict_returns_generic_500_for_model_failure(class_names_path) -> None:
    class FailingModel(MockKerasModel):
        def predict(self, batch, verbose: int = 0):
            raise RuntimeError("internal model detail")

    app.dependency_overrides[get_mri_service] = lambda: MRIPredictor(
        model_instance=FailingModel(), class_names_path=class_names_path
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ai/mri/predict",
            files={"file": ("scan.png", _image_bytes(), "image/png")},
        )
    app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json()["detail"] == "MRI model inference failed."


def test_health_endpoint_is_exposed(client: TestClient) -> None:
    response = client.get("/api/ai/mri/health")
    assert response.status_code == 200
    assert response.json()["modelAvailable"] is True
    assert "model_available" not in response.json()


def test_health_surfaces_predictor_load_error(tmp_path) -> None:
    app.dependency_overrides[get_mri_service] = lambda: MRIPredictor(
        model_instance=MockKerasModel(), class_names_path=tmp_path / "class_names.json"
    )
    with TestClient(app) as client:
        response = client.get("/api/ai/mri/health")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["modelAvailable"] is False
    assert response.json()["message"] == "Class mapping metadata 'class_names.json' is missing."
