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
The predictor is created at startup and cached. After adding or changing the `.keras`
file or `class_names.json`, restart the service.

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
mapping, the mapping in `class_names.json`, whether they match, sampled and per-class
accuracy, and a confusion table whose rows are true folders and columns are predicted
indexes. Set `CLINIQ_MRI_CLASS_NAMES_PATH` when the metadata is not at
`models/class_names.json`. Use `--min-accuracy` to set the minimum overall accuracy
(default `0.80`). The script exits 0 when the mapping matches and accuracy meets the
threshold, 1 when the predictor is unavailable, 2 for invalid input, and 3 when the
mapping differs or accuracy is below the threshold. Low accuracy spread across all
classes usually points to a preprocessing mismatch, not a class-order problem.

## Skin API

The optional Skin module uses the same central application:

- `GET /api/ai/skin/health` reports `status`, `modelAvailable`, `message`, `modelVersion`, and `classCount`.
- `POST /api/ai/skin/predict` accepts multipart `file` in JPEG/PNG with matching extension/content, up to 10 MiB and 120 million decoded pixels by default. Portrait, landscape and panoramic images are resized internally to 224 x 224 using the training preprocessing.
- Returns `predictedClass` (null when abstained), calibrated `confidence`, `abstained`, `abstainReason`, `topPredictions`, all-class `probabilities`, `modelVersion`, `inferenceMs`, and `disclaimer`.

Swagger documents upload errors, including HTTP 413. Oversized responses state the configured file or pixel limit in `detail`; reduce the image size before retrying. `SKIN_MAX_UPLOAD_BYTES` and `SKIN_MAX_PIXELS` override these defaults after restart. Existing installations with `SKIN_MAX_UPLOAD_BYTES=5242880` or `SKIN_MAX_PIXELS=20000000` should update them to `10485760` and `120000000` respectively to use the new defaults. Multipart overhead is capped separately; uploads remain in memory. The pixel count is checked before full decoding, and each upload is decoded once inside the Skin inference queue. Increasing accepted resolution does not improve model accuracy.

The active project trains **Acne, Eczema, Psoriasis, Rosacea, Ringworm and Normal** from the pinned [Skin Assessment dataset](https://huggingface.co/datasets/Cuebot/skin-assessment-13-class-filtered). It starts a new ImageNet transfer-learning run and does not reuse old Skin data, weights, splits or metrics. See [six-class training instructions](docs/skin/SIX_CLASS_TRAINING.md) and [Colab/Kaggle GPU notebook](skin/training/skin_six_colab.ipynb).

The completed experiment measured **82.96% accuracy and 0.8323 macro-F1 on all 804 held-out images**. Every class exceeded 0.70 recall, but the 85% accuracy/0.85 macro-F1 engineering gate failed. After reviewing these results, the requester explicitly approved local default activation **for educational use**. The measured gate remains false; this is a research prototype. Weights remain outside Git. See the [model card and per-class results](docs/skin/SIX_CLASS_MODEL_CARD.md).

An independent, frozen SCIN evaluation subsequently measured **47.83% agreement (66/138 images), 45.44% balanced accuracy and 0.4818 macro-F1 across five diseases**. No exact/near overlap was detected against the original 5,372 images. Normal was not evaluated. The model needs better training data and a fresh external evaluation; the same-source score does not establish generalization. See the [external evaluation, protocol and limitations](docs/skin/EXTERNAL_SCIN_EVALUATION.md). The original weights and measured gate remain unchanged.

The notebook downloads a pinned public source commit automatically; no local upload or GitHub credentials are needed. `python scripts/package_skin_six.py` also creates a source-only ZIP for manual transfer. Dataset preparation excludes supplied `aug_*` images before grouped duplicate-safe splitting. Training uses class weights, augmentation on training images only and two-stage fine-tuning; model selection uses validation macro-F1. Finalization locks the chosen model, evaluates the untouched test split and exports the actual accuracy, per-class metrics, confusion matrix and artifact metadata. The default 85% accuracy/0.85 macro-F1 target is not a measured result or a promise.

Place the newly exported `skin_six.keras`, matching `model_manifest.json` and numeric `class_names.json` in `skin/artifacts/six-class/`, or set `SKIN_MODEL_PATH` and `SKIN_MANIFEST_PATH`. Legacy weights in the parent artifacts directory are not loaded by default. The manifest validates the model SHA256, preprocessing and exact six-class names/index order. Model weights are ignored by Git; deliver them on a local/shared volume and restart. Missing/bad artifacts report not-ready and prediction HTTP 503 without blocking Voice or MRI. Six-class model readiness and measured accuracy require an actual completed GPU training/export run.

```powershell
curl.exe -X POST http://127.0.0.1:8080/api/ai/skin/predict `
  -F "file=@C:\path\to\skin.jpg;type=image/jpeg"
```

This is a research prototype, not a medical diagnosis or validated medical device. Confidence is a model score, not diagnostic correctness. `Normal` is a dataset label, not proof of healthy skin. External clinical generalization, patient independence and demographic fairness are not validated. Source labels and upstream image rights are not independently verified. Skin uploads remain in memory and are not logged or persisted; the multipart body is bounded before parsing. New GPU training dependencies are isolated in `skin/training/requirements-cloud.txt`; historical documents/reports refer to the earlier experiment and do not describe six-class performance.

## Architecture (voice / mri / skin)

```text
Flutter / .NET services -> ClinIQ AI Service (FastAPI)
                                      -> voice
                                      -> mri
                                      -> skin
                                      -> future RAG
```

`main.py` owns each module prefix and initializes the optional cached MRI/Skin predictors during app
startup. Future AI modules can be mounted there without changing the Voice Call API.

## Tests

```powershell
python -m pytest -v
```

Tests mock Groq and MRI/Skin inference; they do not require model weights or send requests to
external services.
