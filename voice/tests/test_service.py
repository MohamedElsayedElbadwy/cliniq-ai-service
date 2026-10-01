"""Unit tests for the Groq Voice Call service without external API requests."""

from io import BytesIO
from types import SimpleNamespace
import wave

import pytest

from voice.service import (
    MAX_TTS_INPUT_CHARS,
    VoiceCallService,
    VoiceTurnError,
    merge_wav_audio,
    normalize_audio_for_stt,
    read_tts_audio,
    split_tts_text,
)


def make_wav(frames: bytes = b"\x00\x00" * 40, rate: int = 16000) -> bytes:
    """Build a minimal valid mono PCM WAV payload for service tests."""
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(frames)
    return output.getvalue()


class FakeBinaryResponse:
    """Minimal stand-in for Groq's BinaryAPIResponse."""

    def __init__(self, audio: bytes) -> None:
        self.audio = audio

    def write_to_file(self, path) -> None:
        with open(path, "wb") as audio_file:
            audio_file.write(self.audio)


class FakeGroqClient:
    """Configurable fake covering STT, LLM, and TTS calls."""

    def __init__(self, transcription="hello", assistant="Hi there", audio=None) -> None:
        self.transcription = transcription
        self.assistant = assistant
        self.audio = make_wav() if audio is None else audio
        self.stt_error = None
        self.llm_error = None
        self.tts_error = None
        self.transcription_calls = []
        self.chat_calls = []
        self.tts_calls = []
        self.audio_api = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self.create_transcription),
            speech=SimpleNamespace(create=self.create_speech),
        )
        self.audio = self.audio_api
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_completion))
        self.speech_audio = make_wav() if audio is None else audio

    def create_transcription(self, **kwargs):
        self.transcription_calls.append(kwargs)
        if self.stt_error:
            raise self.stt_error
        return self.transcription

    def create_completion(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.llm_error:
            raise self.llm_error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.assistant))]
        )

    def create_speech(self, **kwargs):
        self.tts_calls.append(kwargs)
        if self.tts_error:
            raise self.tts_error
        return FakeBinaryResponse(self.speech_audio)


@pytest.fixture
def service(monkeypatch):
    """Create a service with valid config and replace its network client."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    instance = VoiceCallService()
    instance.client = FakeGroqClient()
    return instance


def test_service_requires_groq_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        VoiceCallService()


def test_read_tts_audio_reads_playable_wav():
    assert read_tts_audio(FakeBinaryResponse(make_wav()))[:4] == b"RIFF"


@pytest.mark.parametrize("response", [object(), FakeBinaryResponse(b""), FakeBinaryResponse(b"bad")])
def test_read_tts_audio_rejects_invalid_response(response):
    with pytest.raises(RuntimeError):
        read_tts_audio(response)


def test_split_tts_text_handles_short_and_long_responses():
    assert split_tts_text("short answer") == ["short answer"]
    text = "word " * (MAX_TTS_INPUT_CHARS // 2)
    chunks = split_tts_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= MAX_TTS_INPUT_CHARS for chunk in chunks)


def test_merge_wav_audio_combines_valid_chunks():
    merged = merge_wav_audio([make_wav(b"\x01\x00"), make_wav(b"\x02\x00")])
    with wave.open(BytesIO(merged), "rb") as wav_file:
        assert wav_file.getnframes() == 2


def test_merge_wav_audio_rejects_incompatible_chunks():
    with pytest.raises(RuntimeError, match="incompatible"):
        merge_wav_audio([make_wav(rate=16000), make_wav(rate=8000)])


def test_normalize_audio_for_stt_returns_16khz_mono_wav():
    normalized = normalize_audio_for_stt(make_wav(rate=8000))
    with wave.open(BytesIO(normalized), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2


def test_normalize_audio_for_stt_rejects_invalid_recording():
    with pytest.raises(RuntimeError, match="could not be decoded"):
        normalize_audio_for_stt(b"not audio")


def test_process_turn_runs_stt_llm_tts_and_keeps_history(service):
    fake = FakeGroqClient(transcription="  ازيك  ", assistant="أهلا بيك")
    service.client = fake
    result = service.process_turn(make_wav(), "recording.webm", [{"role": "assistant", "content": "old"}])

    assert result["user_text"] == "ازيك"
    assert result["assistant_text"] == "أهلا بيك"
    assert result["audio"][:4] == b"RIFF"
    uploaded_file = fake.transcription_calls[0]["file"]
    assert uploaded_file[0] == "recording.wav"
    assert uploaded_file[2] == "audio/wav"
    assert uploaded_file[1].tell() == 0
    assert uploaded_file[1].read(4) == b"RIFF"
    assert fake.chat_calls[0]["messages"][-2:] == [
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "ازيك"},
    ]
    assert fake.tts_calls[0]["response_format"] == "wav"


@pytest.mark.parametrize(
    ("stage", "configure", "message"),
    [
        ("STT", lambda fake: setattr(fake, "stt_error", ConnectionError("offline")), "transcription"),
        ("LLM", lambda fake: setattr(fake, "llm_error", RuntimeError("chat down")), "chat"),
        ("TTS", lambda fake: setattr(fake, "tts_error", RuntimeError("tts down")), "synthesis"),
    ],
)
def test_process_turn_reports_each_groq_stage(service, stage, configure, message):
    fake = FakeGroqClient()
    configure(fake)
    service.client = fake
    with pytest.raises(VoiceTurnError, match=message) as error:
        service.process_turn(make_wav(), "recording.webm", [])
    assert error.value.stage == stage


def test_process_turn_rejects_empty_stt_response(service):
    service.client = FakeGroqClient(transcription="  ")
    with pytest.raises(VoiceTurnError, match="detect speech") as error:
        service.process_turn(make_wav(), "recording.webm", [])
    assert error.value.stage == "STT"


def test_process_turn_rejects_empty_llm_response(service):
    service.client = FakeGroqClient(assistant="")
    with pytest.raises(VoiceTurnError, match="did not generate") as error:
        service.process_turn(make_wav(), "recording.webm", [])
    assert error.value.stage == "LLM"
