# ClinIQ AI Service

Central FastAPI service for ClinIQ AI capabilities. The current implementation exposes
the existing turn-based Voice Call feature, its local HTML/JavaScript test UI, and a
Brain MRI image-classification module.

## Run locally

```powershell
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env. Configure CLINIQ_MRI_MODEL_PATH only when needed.
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

## Brain MRI API

The MRI module is available at:

- `GET /api/ai/mri/health` returns `status`, `modelAvailable`, and `message`.
- `POST /api/ai/mri/predict` accepts a `file` upload in JPG, JPEG, or PNG format and
  returns `prediction`, `classIndex`, `confidence`, and `probabilities`.

Place `mri_xception_selective_finetuning_best.keras` in `models/`, or set
`CLINIQ_MRI_MODEL_PATH` in `.env` to the full path of that file. The model weights are
intentionally ignored by Git. Also place the matching `class_names.json` exported from
the same training run in `models/`, or set `CLINIQ_MRI_CLASS_NAMES_PATH` to its path.
This metadata is required and must use the format `{ "0": "class-name", ... }`.
If either artifact is absent or invalid, health reports the module as unavailable and
predict returns HTTP 503; the Voice feature remains available.

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/ai/mri/predict `
  -F "file=@C:\path\to\scan.png;type=image/png"
```

MRI output is a model classification only. It is not a medical diagnosis and must not
be used as a substitute for qualified clinical judgment.

## Verifying the class mapping

After exporting `class_names.json` from the training run, verify it against a labeled
testing directory with one subfolder per true class:

```powershell
python scripts/verify_class_mapping.py `
  --model models/mri_xception_selective_finetuning_best.keras `
  --data C:\path\to\testing `
  --per-class 50
```

The script reuses production preprocessing and prints the inferred index-to-folder
mapping, the mapping in `class_names.json`, whether they match, and sampled accuracy.
Set `CLINIQ_MRI_CLASS_NAMES_PATH` when the metadata is not at `models/class_names.json`.

## Architecture

```text
Flutter / .NET services -> ClinIQ AI Service (FastAPI)
                                      -> voice
                                      -> mri
                                      -> future RAG
```

`main.py` owns each module prefix and initializes the cached MRI predictor during app
startup. Future AI modules can be mounted there without changing the Voice Call API.

## Tests

```powershell
python -m pytest -v
```

Tests mock Groq and the MRI model; they do not require model weights or send requests to
external services.
