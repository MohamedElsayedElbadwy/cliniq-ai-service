"""Application-level tests for the centralized FastAPI entry point."""

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_application_metadata_and_voice_openapi_contract():
    schema = client.get("/openapi.json").json()
    assert schema["info"] == {"title": "ClinIQ AI Service", "version": "1.0.0"}
    assert "/api/ai/voice/turn" in schema["paths"]
    assert client.get("/docs").status_code == 200


def test_root_serves_voice_call_interface_and_static_assets():
    page = client.get("/")
    assert page.status_code == 200
    assert "ClinIQ" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
