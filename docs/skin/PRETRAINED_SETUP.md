# Ready-to-use pretrained Skin backend

The selected demo now uses MAKE: follow [MAKE setup](MAKE_SETUP.md). The native22
backend below remains an optional reproducible research comparison; its39.38%
cohort accuracy did not justify replacing the baseline.

This adds local CPU inference using `LaurianeMD/vit-skin-disease` at revision
`1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c`. It performs no training. Existing
six-class Keras weights, maps, splits and reports remain in their original paths.
The source default remains `SKIN_BACKEND=keras`; select `pretrained` explicitly.

## Tested Windows environment and installation

Run from the existing repository/environment. The following installs only optional
pretrained packages; it does not upgrade the existing TensorFlow/Keras/Voice stack:

```powershell
Set-Location 'D:\fetaure_skin\cliniq-ai-service'
$skinPython = '.\.venv-skin-six\Scripts\python.exe'
& $skinPython -m ensurepip --upgrade
& $skinPython -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
& $skinPython -m pip install -r skin/requirements-pretrained.txt
& $skinPython -m pip check
& $skinPython -m skin.download_pretrained
```

For a fresh checkout, first create an isolated Python3.12 environment and install
the existing `requirements.txt`; then install the optional packages above. This
project already needs TensorFlow for MRI and the preserved Keras backend.
The tested interpreter is Python3.12.14, with PyTorch2.14.1+cpu,
Transformers4.57.6 and safetensors0.8.0. The shared optional requirements also
install CPU torchvision/OpenCLIP/timm for the selected MAKE backend. The native
ViT runtime itself does not call OpenCLIP. No CUDA, paid inference API or Hugging
Face token is required.

The checkpoint is343,285,496bytes (327.38MiB). The separate real smoke test measured
43.01seconds to load and653.71MiB peak process working set; these are observations
on this laptop, not minimum-RAM guarantees. The shared API also loads MRI and has
additional memory overhead. CPU inference is serialized and uses two Torch threads.

## Download and offline cache

The downloader uses the immutable revision above and verifies byte size/SHA256 for
`config.json`, `preprocessor_config.json`, `README.md`, and `model.safetensors`.
Weights SHA256: `3d85c0ef41fc65937401ae1fa5cc37a8ec139754ef5e1e21539603417c6ae29c`.
The default cache is `workspaces/pretrained-skin/<revision>/`, ignored by Git.
Existing correct files are reused. A different existing file causes an error;
choose another directory instead of deleting or replacing an old artifact.
Partial files created by a failed download are cleaned up. Interrupted downloads
restart from the beginning; they do not overwrite the preserved Keras model.

For a different local cache directory:

```powershell
& $skinPython -m skin.download_pretrained --output 'D:\fetaure_skin\pretrained-cache'
$env:SKIN_PRETRAINED_DIR = 'D:\fetaure_skin\pretrained-cache'
```

Serving loads verified local files only, with safetensors and no remote model code.
There are no per-request downloads. A missing checkpoint, checksum mismatch or
missing dependency makes Skin unavailable/HTTP503; Voice/MRI remain isolated.
It never silently substitutes the Keras model for a failed pretrained backend.

## Start the existing central API

Use one server process on this8GB laptop. Avoid `--reload`, which can increase
memory use. Stop an earlier copy in its own terminal before starting this command:

```powershell
Set-Location 'D:\fetaure_skin\cliniq-ai-service'
$env:SKIN_BACKEND = 'pretrained'
# Empty selects the verified default cache under this repository.
$env:SKIN_PRETRAINED_DIR = ''
$env:SKIN_ABSTAIN_THRESHOLD = ''
$env:SKIN_PRETRAINED_MIN_CONFIDENCE = '0.70'
$env:SKIN_PRETRAINED_MIN_MARGIN = '0.15'
$env:SKIN_PRETRAINED_THREADS = '2'
$env:SKIN_MAX_UPLOAD_BYTES = '10485760'
$env:SKIN_MAX_PIXELS = '120000000'
& '.\.venv-skin-six\Scripts\python.exe' -m uvicorn main:app --host 127.0.0.1 --port 8080
```

`main.py` is unchanged. It initializes the cached optional Skin predictor during
startup; the first load can take tens of seconds. Configuration changes require a
restart. Existing Voice provider configuration remains separate; Skin inference
does not call Groq or another paid API. No `.env` secrets are changed by this work.

To select the preserved model, set `SKIN_BACKEND=keras` and restart. Its default
artifacts remain `skin/artifacts/six-class/skin_six.keras` and
`skin/artifacts/six-class/model_manifest.json`. Existing path overrides still apply
only to that backend. Restoring it requires no download or retraining.

## Swagger manual test

1. Open [Swagger](http://127.0.0.1:8080/docs).
2. Expand **GET /api/ai/skin/health**, click **Try it out**, then **Execute**.
   A ready pretrained backend reports `classCount:22` and the pinned Hugging Face
   identifier in `modelVersion`.
3. Expand **POST /api/ai/skin/predict**, click **Try it out**.
4. For multipart field **file**, select a JPEG or PNG photograph, then **Execute**.
5. Expect the existing JSON fields: `predictedClass`, `confidence`, `abstained`,
   `abstainReason`, `topPredictions`, `probabilities`, `modelVersion`, `inferenceMs`
   and `disclaimer`. All22 native label scores appear in `probabilities`; up to
   three appear in `topPredictions`. Existing clients must use model metadata
   rather than assume six fixed label values.

The exact prediction URL is `http://127.0.0.1:8080/api/ai/skin/predict`.

```powershell
curl.exe -X POST 'http://127.0.0.1:8080/api/ai/skin/predict' `
  -F 'file=@D:\path\to\skin.jpg;type=image/jpeg'
```

Supported uploads remain JPEG/PNG with matching extension/signature, at most10MiB
and120,000,000 original pixels, any aspect ratio. Corrupt/empty images return400,
unsupported or mismatched types415, limits413, inference failures500 and unavailable
models503. Uploads stay in memory, are decoded once under the inference queue, and
are not persisted or logged as images. Existing upload/error response contracts
remain unchanged.

## Interpret output correctly

`confidence` and `probability` retain existing field names but represent ordinary,
uncalibrated softmax scores. They are not measured diagnostic correctness.
`predictedClass:null` with `abstained:true` means the API withholds a class; the
unfiltered top scores remain visible for inspection. Reasons are:

- `low_confidence`: top score below.70 by default.
- `ambiguous_prediction`: top two scores differ by less than.15.
- `unknown_or_normal`: the winning native label is `Unknown Normal`; always reject.

These thresholds were fixed before evaluation, not fitted on the259images. They
cannot detect every unsupported condition or non-skin image and do not make
high-confidence predictions reliable. Setting `SKIN_ABSTAIN_THRESHOLD` overrides
the confidence threshold for the selected backend; leave it empty for defaults.

The actual classes are: Acne, Actinic Keratosis, Benign Tumors, Bullous,
Candidiasis, Drug Eruption, Eczema, Infestations Bites, Lichen, Lupus, Moles,
Psoriasis, Rosacea, Seborrh Keratoses, Skin Cancer, Sun Sunlight Damage, Tinea,
Unknown Normal, Vascular Tumors, Vasculitis, Vitiligo and Warts.
`Unknown Normal` combines unknown/normal and is not a verified healthy-skin class.
`Tinea` is broader than strict ringworm. Labels including `Skin Cancer` are model
categories, not diagnoses. The API never supplies a prescription.

## Reproduce evaluation and tests

```powershell
& $skinPython -m pytest -v -m 'real_model or not real_model' --basetemp .test-tmp/pretrained-full-local
& $skinPython -m skin.evaluate_pretrained --output workspaces/pretrained-evaluation-new
```

Use a new evaluation output directory; existing data/reports are never overwritten.
Evaluation requires the existing reviewed259image workspace and the locked v1
prediction cache. It verifies split/image hashes and preserves native predictions;
it does not restrict softmax to the five reference diseases or tune thresholds.
The separate254image comparison omits the approximate Tinea/Ringworm alias.
The publisher's training image inventory is unavailable, so overlap with this
cohort cannot be ruled out. Do not describe these images as independently unseen
to the downloaded model. See [actual results](PRETRAINED_MODEL_EVALUATION.md).

Publisher model/license sources: [pinned primary repository](https://huggingface.co/LaurianeMD/vit-skin-disease/tree/1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c),
[official ViT implementation](https://huggingface.co/docs/transformers/v4.57.6/en/model_doc/vit),
[CPU PyTorch installation](https://pytorch.org/get-started/locally/).
The model card declares Apache-2.0 and educational/research intent; upstream
photograph rights and clinical labels are not independently established.
