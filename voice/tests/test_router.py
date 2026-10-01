"""API contract tests for the centralized Voice Call router."""

import base64

from fastapi.testclient import TestClient

from main import app
from voice.service import VoiceTurnError

client = TestClient(app)


class FakeVoiceService:
    """Injectable voice service used to keep route tests offline."""

    result = {"user_text": "hello", "assistant_text": "hi", "audio": b"wav"}
    error = None

    def process_turn(self, audio_bytes, filename, history):
        if self.error:
            raise self.error
        return self.result


def install_fake_service(monkeypatch, service):
    """Replace the route's service constructor for one request."""
    monkeypatch.setattr("voice.router.VoiceCallService", lambda: service)


def test_voice_turn_returns_existing_response_shape(monkeypatch):
    install_fake_service(monkeypatch, FakeVoiceService())
    response = client.post(
        "/api/ai/voice/turn",
        files={"audio": ("turn.webm", b"audio", "audio/webm")},
        data={"history_json": "[]"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "userText": "hello",
        "assistantText": "hi",
        "audioBase64": base64.b64encode(b"wav").decode("ascii"),
    }


def test_voice_turn_validates_upload_and_history(monkeypatch):
    install_fake_service(monkeypatch, FakeVoiceService())
    assert client.post("/api/ai/voice/turn").status_code == 422
    assert client.post(
        "/api/ai/voice/turn", files={"audio": ("note.txt", b"x", "text/plain")}
    ).status_code == 400
    assert client.post(
        "/api/ai/voice/turn",
        files={"audio": ("turn.webm", b"audio", "audio/webm")},
        data={"history_json": "{}"},
    ).status_code == 400


def test_voice_turn_maps_pipeline_errors_to_502(monkeypatch):
    service = FakeVoiceService()
    service.error = VoiceTurnError("TTS", "Groq speech synthesis failed: unavailable")
    install_fake_service(monkeypatch, service)
    response = client.post(
        "/api/ai/voice/turn",
        files={"audio": ("turn.webm", b"audio", "audio/webm")},
    )
    assert response.status_code == 502
    assert response.json()["detail"].startswith("TTS failed:")
