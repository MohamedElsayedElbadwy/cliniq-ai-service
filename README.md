# ClinIQ AI Service

Central FastAPI service for ClinIQ AI capabilities. The current implementation exposes
the existing turn-based Voice Call feature and its local HTML/JavaScript test UI.

## Run locally

```powershell
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env
python -m uvicorn main:app --reload --port 8080
```

Open `http://127.0.0.1:8080/` for the Voice Call UI and
`http://127.0.0.1:8080/docs` for Swagger/OpenAPI documentation.

## Voice API

`POST /api/ai/voice/turn` accepts multipart form fields:

- `audio`: the microphone audio file.
- `history_json`: a JSON list of prior `{ "role", "content" }` messages.

It returns `userText`, `assistantText`, and `audioBase64`. The pipeline remains Groq
Whisper STT -> Groq LLM -> Groq WAV TTS. Only `GROQ_API_KEY` is required.

## Architecture

```text
Flutter / .NET services -> ClinIQ AI Service (FastAPI) -> voice / future MRI / future RAG
```

Future AI modules will be mounted from `main.py` without changing the Voice Call API.

## Tests

```powershell
python -m pytest -v
```

Tests mock Groq and do not send requests to external services.
