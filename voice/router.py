"""Voice Call HTTP endpoints exposed by the central AI service."""

import asyncio
import base64
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .schemas import VoiceTurnResponse
from .service import VoiceCallService, VoiceTurnError

router = APIRouter(tags=["voice"])


@router.post("/turn", response_model=VoiceTurnResponse)
async def process_voice_turn(
    audio: UploadFile = File(...), history_json: str = Form("[]")
) -> VoiceTurnResponse:
    """Process a microphone recording through the existing Voice Call pipeline."""
    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="The uploaded file must be audio.")

    try:
        history = json.loads(history_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="history_json must be valid JSON.") from exc

    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history_json must be a JSON list.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The uploaded audio is empty.")

    try:
        result = await asyncio.to_thread(
            VoiceCallService().process_turn,
            audio_bytes,
            audio.filename or "voice-turn.webm",
            history,
        )
    except VoiceTurnError as exc:
        raise HTTPException(status_code=502, detail=f"{exc.stage} failed: {exc}") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Voice Call processing failed.") from exc

    return VoiceTurnResponse(
        userText=result["user_text"],
        assistantText=result["assistant_text"],
        audioBase64=base64.b64encode(result["audio"]).decode("ascii"),
    )
