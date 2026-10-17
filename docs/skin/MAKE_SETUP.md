# Selected MAKE backend — local educational demo

MAKE achieved209/259correct (80.69%, macro-F1.6943) on the existing five-disease
comparison cohort, versus162/259(62.55%) for the preserved baseline. This cohort
was used for model selection, so it is not a new independent final test. Publisher
training-image overlap is unknown. No training, prompt tuning or threshold tuning
was performed. [Results and confusions](PRETRAINED_MODEL_EVALUATION.md).

This is a fixed six-candidate **zero-shot** classifier using actual pretrained
image/text encoders, not a newly trained six-class head. Candidates: Acne, Eczema,
Psoriasis, Rosacea, Ringworm, Normal. Normal is always withheld because healthy
skin recognition has not been validated.

## Install and obtain the pinned checkpoint

Use the existing Python3.12 environment. For a fresh checkout, create a Python3.12
virtual environment and install existing requirements.txt first; MRI still needs
TensorFlow. Use CPU wheels on this setup.

```powershell
Set-Location 'D:\fetaure_skin\cliniq-ai-service'
$skinPython = '.\.venv-skin-six\Scripts\python.exe'
& $skinPython -m ensurepip --upgrade
& $skinPython -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
& $skinPython -m pip install -r skin/requirements-pretrained.txt
& $skinPython -m pip check
& $skinPython -m skin.download_make
```

Tested optional runtime: Torch2.14.1+cpu, torchvision0.29.1+cpu, OpenCLIP3.3.0,
timm1.0.30, Transformers4.57.6, safetensors0.8.0. Existing TensorFlow/Keras/Voice
packages are retained. No paid service or HF token is needed.

Checkpoint: [MAKE pinned revision](https://huggingface.co/Xieji-Li/MAKE/tree/5efa1915f1d0128451ddd61242aad8475bfb8e1a),
598,516,980bytes, SHA256
`4ed233070183558b9f9e4572bdb2676f7da0f2c463d9b4d436b169e62700bf47`.
Cache: `workspaces/pretrained-make/5efa1915f1d0128451ddd61242aad8475bfb8e1a/`.
All three files, including architecture/preprocessing config, are pinned by size
and hash. Correct cache files are reused; different existing files are preserved
and cause an error. Serving is offline/local-only, safetensors only, strict loading.
No request-time downloads or silent fallback.

The HF README states MIT; underlying Derm1M data has noncommercial research terms.
This is an attributed local educational research demo, not evidence of commercial
image rights or medical suitability. [Official implementation](https://github.com/SiyuanYan1/MAKE).

## Start the shared API

If already running, use its Swagger page. For a new start, stop the earlier server
in its own terminal first. Use one worker without reload on this8GB laptop:

```powershell
Set-Location 'D:\fetaure_skin\cliniq-ai-service'
$env:SKIN_BACKEND = 'make'
$env:SKIN_MAKE_DIR = ''
$env:SKIN_ABSTAIN_THRESHOLD = ''
$env:SKIN_PRETRAINED_MIN_CONFIDENCE = '0.70'
$env:SKIN_PRETRAINED_MIN_MARGIN = '0.15'
$env:SKIN_PRETRAINED_THREADS = '2'
$env:SKIN_TOP_K = '3'
$env:SKIN_MAX_UPLOAD_BYTES = '10485760'
$env:SKIN_MAX_PIXELS = '120000000'
& '.\.venv-skin-six\Scripts\python.exe' -m uvicorn main:app --host 127.0.0.1 --port 8080
```

Existing main.py, Voice and MRI feature code are unchanged. The model and text
embeddings load once per process. Configuration changes need a restart. Source
default remains `keras` for existing installs; the delivered local server selects
`make` explicitly. Restore the old model with `SKIN_BACKEND=keras` and a restart.
The weaker native22model remains an [optional research backend](PRETRAINED_SETUP.md).

## Swagger test

1. Open [Swagger](http://127.0.0.1:8080/docs).
2. GET `/api/ai/skin/health` → **Try it out** → **Execute**. Expect ready,
   `classCount:6`, and `xieji-x/MAKE@5efa1915...` in `modelVersion`.
3. POST `/api/ai/skin/predict` → **Try it out** → choose a JPEG/PNG in **file**
   → **Execute**.
4. Read `predictedClass`, `confidence`, `abstained`, `abstainReason`,
   `topPredictions`, six `probabilities`, `modelVersion`, `inferenceMs`, `disclaimer`.
   A null prediction means withheld; top scores remain visible as model outputs.

```powershell
curl.exe -X POST 'http://127.0.0.1:8080/api/ai/skin/predict' `
  -F 'file=@D:\fetaure_skin\cliniq-ai-service\workspaces\skin-external-samples\Eczema.png;type=image/png'
```

JPEG/PNG images may have any aspect ratio within the resource budgets:10MiB,
120million original pixels and120million pixels at the intermediate resize.
Extremely thin images that would exceed that resize budget return413 before
the resize allocates memory; crop them before retrying.
Official preprocessing resizes the shortest side to224, center-crops224², converts
RGB and uses CLIP normalization. Extreme aspect ratios can lose peripheral lesions
in the center crop. Dimension acceptance does not ensure useful image content.
EXIF orientation is applied first. Corrupt/empty uploads return400, unsupported or
signature mismatch415, limits413, unavailable model503, inference failure500.

Confidence is an **uncalibrated softmax similarity score conditional on six fixed
descriptions**, not diagnostic correctness. Fixed rejection: top score<.70,
top-two margin<.15, or winning Normal (`normal_unverified`). These were fixed before
evaluation. Non-skin and unsupported diseases can still receive confident wrong
scores. No prescription or clinical diagnosis is provided.

## Reproduce tests and evaluation

```powershell
& '.\.venv-skin-six\Scripts\python.exe' -m pytest -v -m 'real_model or not real_model' --basetemp .test-tmp/make-local-tests
& '.\.venv-skin-six\Scripts\python.exe' -m skin.evaluate_make --output workspaces/make-evaluation-new
```

Use a new output directory. Evaluation requires the existing reviewed local259image
workspace; images/case responses are not committed. It verifies image/split hashes
and never trains or tunes. Compare the delivered backend against frozen259×6scores:

```powershell
& '.\.venv-skin-six\Scripts\python.exe' -m skin.evaluate_make `
  --output workspaces/make-parity-new `
  --reference-scores workspaces/make-skin-evaluation-2026-10-17-run2/native_scores.npy
```

Missing optional weights cause real-model tests to skip explicitly in a fresh
checkout. Unit tests alone do not prove actual inference. The recorded local run
includes checkpoint verification and real inference.
