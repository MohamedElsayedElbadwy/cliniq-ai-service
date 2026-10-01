"""HTTP response models for Voice Call endpoints."""

from pydantic import BaseModel


class VoiceTurnResponse(BaseModel):
    """The transcript and WAV payload returned after one Voice Call turn."""

    userText: str
    assistantText: str
    audioBase64: str
