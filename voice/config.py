"""Configuration for the Groq-backed Voice Call feature."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class VoiceSettings:
    """Runtime settings required by the Voice Call service."""

    groq_api_key: str
    stt_model: str = "whisper-large-v3"
    llm_model: str = "openai/gpt-oss-120b"
    tts_model: str = "canopylabs/orpheus-arabic-saudi"
    tts_voice: str = "fahad"
    request_timeout_seconds: float = 90.0


def load_settings() -> VoiceSettings:
    """Load the server-side Groq configuration from the environment."""
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        raise RuntimeError("Missing required environment variable: GROQ_API_KEY")
    return VoiceSettings(groq_api_key=groq_api_key)
