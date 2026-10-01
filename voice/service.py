"""Groq-only service layer for the turn-based ClinIQ Voice Call."""

from io import BytesIO
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
import wave
from typing import Any

import av
from groq import Groq

from .config import load_settings

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are the ClinIQ clinic voice assistant.
Provide general informational assistance and clinic-related guidance. Do not diagnose
medical conditions or prescribe medication. For urgent or emergency symptoms, advise
the user to seek appropriate professional medical care. Support Egyptian Arabic,
Modern Standard Arabic, and English. Respond in the same language and dialect as the
user. Keep spoken responses to one to three short sentences, natural, and free of markdown formatting."""

MAX_TTS_INPUT_CHARS = 200
STT_SAMPLE_RATE = 16000


class VoiceTurnError(RuntimeError):
    """Identifies the Groq stage that failed during a Voice Call turn."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def read_tts_audio(response: Any) -> bytes:
    """Save Groq's BinaryAPIResponse as WAV, then return its audio bytes."""
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_audio_path = Path(temp_file.name)

    if not hasattr(response, "write_to_file"):
        temp_audio_path.unlink(missing_ok=True)
        raise RuntimeError("Groq TTS returned an unsupported audio response.")

    try:
        response.write_to_file(temp_audio_path)
        audio_bytes = temp_audio_path.read_bytes()
    finally:
        temp_audio_path.unlink(missing_ok=True)

    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise RuntimeError("Groq TTS returned empty or invalid audio data.")

    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            wav_file.getparams()
    except (EOFError, wave.Error) as exc:
        raise RuntimeError("Groq TTS did not return a playable WAV response.") from exc

    return audio_bytes


def normalize_audio_for_stt(audio_bytes: bytes) -> bytes:
    """Decode browser audio and return a 16 kHz mono PCM WAV for Whisper."""
    if not audio_bytes:
        raise RuntimeError("The uploaded audio is empty.")

    try:
        with av.open(BytesIO(audio_bytes), mode="r") as input_container:
            audio_stream = next(iter(input_container.streams.audio), None)
            if audio_stream is None:
                raise RuntimeError("The uploaded file does not contain an audio stream.")

            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=STT_SAMPLE_RATE
            )
            pcm_chunks: list[bytes] = []
            for frame in input_container.decode(audio_stream):
                for resampled_frame in resampler.resample(frame):
                    pcm_chunks.append(bytes(resampled_frame.planes[0])[: resampled_frame.samples * 2])
            for resampled_frame in resampler.resample(None):
                pcm_chunks.append(bytes(resampled_frame.planes[0])[: resampled_frame.samples * 2])
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("The uploaded audio could not be decoded as a valid recording.") from exc

    if not pcm_chunks:
        raise RuntimeError("The uploaded audio did not contain any audio samples.")

    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(STT_SAMPLE_RATE)
        wav_file.writeframes(b"".join(pcm_chunks))
    return output.getvalue()


def split_tts_text(text: str) -> list[str]:
    """Keep each Orpheus request inside Groq's 200-character limit."""
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > MAX_TTS_INPUT_CHARS:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def merge_wav_audio(audio_chunks: list[bytes]) -> bytes:
    """Join Groq WAV chunks into one browser-playable response."""
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    output = BytesIO()
    params = None
    frames: list[bytes] = []
    for chunk in audio_chunks:
        with wave.open(BytesIO(chunk), "rb") as wav_file:
            if params is None:
                params = wav_file.getparams()
            elif wav_file.getparams()[:4] != params[:4]:
                raise RuntimeError("TTS returned incompatible WAV chunks.")
            frames.append(wav_file.readframes(wav_file.getnframes()))
    with wave.open(output, "wb") as merged_wav:
        merged_wav.setparams(params)
        for frame_chunk in frames:
            merged_wav.writeframes(frame_chunk)
    return output.getvalue()


class VoiceCallService:
    """Runs one microphone turn through Groq STT, LLM, and TTS."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.client = Groq(
            api_key=self.settings.groq_api_key,
            timeout=self.settings.request_timeout_seconds,
            max_retries=2,
        )

    def process_turn(
        self, audio_bytes: bytes, filename: str, history: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Run microphone bytes through STT, LLM, TTS, and WAV assembly."""
        try:
            wav_audio = normalize_audio_for_stt(audio_bytes)
            wav_stream = BytesIO(wav_audio)
            wav_stream.seek(0)
            transcription = self.client.audio.transcriptions.create(
                file=(f"{Path(filename).stem or 'voice-turn'}.wav", wav_stream, "audio/wav"),
                model=self.settings.stt_model,
                response_format="text",
            )
            user_text = (
                transcription.strip()
                if isinstance(transcription, str)
                else getattr(transcription, "text", str(transcription)).strip()
            )
        except Exception as exc:
            logger.exception("Voice Call STT failed")
            raise VoiceTurnError("STT", f"Groq Whisper transcription failed: {exc}") from exc

        if not user_text:
            raise VoiceTurnError("STT", "Groq could not detect speech in this recording.")

        messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
        for message in history:
            if message.get("role") in {"user", "assistant"} and message.get("content"):
                messages.append({"role": message["role"], "content": message["content"]})
        messages.append({"role": "user", "content": user_text})

        try:
            completion = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=0.4,
                max_tokens=300,
            )
            assistant_text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.exception("Voice Call LLM failed")
            raise VoiceTurnError("LLM", f"Groq chat response failed: {exc}") from exc

        if not assistant_text:
            raise VoiceTurnError("LLM", "Groq did not generate a response.")

        audio_chunks: list[bytes] = []
        for text_chunk in split_tts_text(assistant_text):
            try:
                speech = self.client.audio.speech.create(
                    model=self.settings.tts_model,
                    voice=self.settings.tts_voice,
                    input=text_chunk,
                    response_format="wav",
                )
                audio_chunks.append(read_tts_audio(speech))
            except Exception as exc:
                logger.exception("Voice Call TTS failed")
                raise VoiceTurnError("TTS", f"Groq speech synthesis failed: {exc}") from exc

        try:
            audio = merge_wav_audio(audio_chunks)
        except Exception as exc:
            logger.exception("Voice Call WAV assembly failed")
            raise VoiceTurnError("TTS", f"Groq TTS WAV assembly failed: {exc}") from exc

        return {"user_text": user_text, "assistant_text": assistant_text, "audio": audio}
