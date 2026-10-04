"""
Integration and contract tests for the top-level FastAPI application in main.py.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure Brain MRI project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def test_root_health_check(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "ClinIQ Brain MRI Classification API" in response.json()["service"]


def test_web_static_index_serving(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "ClinIQ" in response.text
    assert "Brain MRI" in response.text


def test_web_static_assets_serving(client: TestClient) -> None:
    css_resp = client.get("/styles.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.headers.get("content-type", "")

    js_resp = client.get("/app.js")
    assert js_resp.status_code == 200


def test_mri_routes_mounted_in_main_app(client: TestClient) -> None:
    # 1. Health check
    health_resp = client.get("/mri/health")
    assert health_resp.status_code == 200

    # 2. Predict endpoint
    stream = io.BytesIO()
    Image.new("RGB", (100, 100)).save(stream, format="PNG")
    predict_resp = client.post(
        "/mri/predict",
        files={"file": ("test_scan.png", stream.getvalue(), "image/png")},
    )
    assert predict_resp.status_code == 200
    data = predict_resp.json()
    assert data["prediction"] == "notumor"
    assert data["confidence"] == pytest.approx(0.85)


def test_openapi_docs_accessible(client: TestClient) -> None:
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    assert openapi_resp.json()["info"]["title"] == "ClinIQ — Brain MRI Classification API"
