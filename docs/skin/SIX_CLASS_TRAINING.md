# Fresh six-class Skin training

The active Skin project starts again with **Acne, Eczema, Psoriasis, Rosacea,
Ringworm and Normal**. No old 22-class images, weights, cached features, split
files, calibration or scores are used. Runtime stays in the existing `skin/`
module and central FastAPI application. Voice and MRI contracts are unchanged.

## Data protocol

Source: [Cuebot/skin-assessment-13-class-filtered](https://huggingface.co/datasets/Cuebot/skin-assessment-13-class-filtered),
revision `41c48b0b0a3b9f36e645db494a0563f7cc770ebd`.
The downloader pins that revision and verifies the archive SHA256 and every
selected original image against the publisher manifest before extraction.

| Exported class | Original source folder | Published files | Excluded `aug_*` files | Originals before duplicate audit |
|---|---|---:|---:|---:|
| Acne | acne | 1,185 | 527 | 658 |
| Eczema | Eczema | 1,192 | 0 | 1,192 |
| Psoriasis | Psoriasis | 908 | 0 | 908 |
| Rosacea | Rosacea | 1,211 | 427 | 784 |
| Ringworm | Ringworm | 1,025 | 0 | 1,025 |
| Normal | normal | 1,240 | 274 | 966 |
| Total | | 6,761 | 1,228 | 5,533 |

These counts came from the pinned publisher manifest, not a count of patients or
verified diagnoses. Filename detection cannot establish absence of all prior
image transformations. Images are decoded and checked for exact byte/pixel
duplicates and near duplicates (pHash distance <=4 AND dHash <=6). Conflicting
cross-class clusters are excluded entirely; same-class exact duplicates are
removed; similar retained images stay in one group. The deterministic 70/15/15
split is stratified by class and indivisible similarity cluster, before online
augmentation. Patient independence is **not** established without patient IDs.
The final measured inventory is `workspaces/skin-six/provenance.json`.

The executed local audit retained **5,372 images** after excluding 1,228 supplied
augmentations and 161 exact duplicates: 3,761 train / 807 validation / 804 test.
No detected byte/pixel/similarity-cluster overlap remains. The portable final
local protocol is under `workspaces/skin-six-v1/`; its aggregate provenance is
checked in as [six_class_dataset.json](six_class_dataset.json). See
[verification evidence](SIX_CLASS_VALIDATION.md) for code checks and pending GPU work.

Do not initialize or run the legacy `skin.training.train`, `finetune`,
`prepare_artifact` or `evaluate` commands for this project. The new commands below
use their own workspace, class map, experiments and finalization lock.

## GPU notebook

Use [the notebook](../../skin/training/skin_six_colab.ipynb) in a fresh Colab GPU
runtime or Kaggle GPU notebook with Internet enabled. It installs training-only
dependencies with the same TensorFlow/Keras versions as ClinIQ. It checks GPU
availability and stops if unavailable. A runtime restart may be needed if the
notebook environment previously imported different TensorFlow/NumPy versions.

The setup cell downloads the verified public source commit directly, without
GitHub credentials, local file uploads or MRI LFS weights. After installing the
pinned dependencies, restart the runtime once if Colab had already imported
NumPy; this avoids mixing preloaded native NumPy code with newly installed files.
Then rerun setup and the GPU check. Command output streams into the notebook.
In Colab the command runner also downloads checkpoint ZIP backups after head
epochs 1/5/8, every fifth fine-tuning epoch, and successful completion. These
contain weights and protocol metadata, never source photographs. Keep the
downloads: temporary Colab storage is lost when a runtime expires. Checkpoint
backups are explicitly marked incomplete until training ends and are never a
substitute for the locked held-out evaluation/export.
Colab transfers these downloads through kernel comm messages. During the
foreground training cell the download can remain queued until the kernel is
idle; a visible download progress bar alone does not establish a local backup.
Verify the downloaded ZIP on disk. These snapshots do not guarantee recovery
from an expired runtime before transfer; persistent storage is still needed for
that guarantee.

For manual source transfer from a local checkout, a small source bundle is also available:

```powershell
python scripts/package_skin_six.py
```

Open `skin/training/skin_six_colab.ipynb` from the feature branch in Colab, or
import it into Kaggle with Internet and GPU enabled. No source ZIP upload is
required by the default notebook. The optional ZIP contains only source/docs;
unpack it and use the CLI commands below when working outside the notebook.
The dataset ZIP download is approximately 903 MiB; preparation extracts only the
six original-image classes. Use persistent storage or download results before
the GPU session ends.

The notebook supports validation-only comparison of B0 and B2; by default it
trains B2. Every run initializes **fresh ImageNet** features and a new six-output
head. This is a new Skin training experiment, not random initialization of all
convolutional features. The head is trained first, then the upper 35% of backbone
layers are fine-tuned at a low learning rate with BatchNorm frozen. Horizontal
flips, modest rotation/zoom/contrast run only during training. Training-only
class weights, dropout, label smoothing, reduced learning rate and early stopping
limit imbalance and overfitting. Selection always uses validation macro-F1.
Every training epoch globally shuffles row indexes before decoding images,
avoiding class blocks from the sorted manifests without buffering all decoded
photographs in memory. Validation/test retain their frozen manifest order.

Implementation follows the official Keras
[transfer-learning guide](https://keras.io/guides/transfer_learning/) and
[EfficientNet preprocessing contract](https://keras.io/api/applications/efficientnet/).
All paths use shared EXIF correction, RGB, 224x224 Pillow bilinear resize and
float32 [0,255]; EfficientNet supplies normalization. No additional /255 scaling.

Equivalent commands from the repository root:

```bash
python -m pip install -r skin/training/requirements-cloud.txt
python -m skin.data.prepare_six --download --workspace workspaces/skin-six
python -m skin.training.six_class train --workspace workspaces/skin-six \
  --run workspaces/skin-six/runs/b2 --architecture efficientnetb2
# Optional B0 comparison, BEFORE finalization; choose by validation.json only:
python -m skin.training.six_class train --workspace workspaces/skin-six \
  --run workspaces/skin-six/runs/b0 --architecture efficientnetb0
# Optional validation-only original/mirror average, BEFORE the test lock:
python -m skin.training.six_class flip-candidate --workspace workspaces/skin-six \
  --source-run workspaces/skin-six/runs/b2 --run workspaces/skin-six/runs/b2-flip
# Compare validation.json; finalize only the selected candidate.
python -m skin.training.six_class finalize --workspace workspaces/skin-six \
  --run workspaces/skin-six/runs/b2 --output workspaces/skin-six/export
```

Defaults: batch 24, head up to 8 epochs, fine-tuning up to 25 epochs. Reduce
`--batch-size` to 12 if GPU memory is insufficient. Existing run/workspace/export
directories are never silently overwritten. Interrupted training keeps its
checkpoint but needs a new run directory; automatic optimizer resume is not
implemented. `--allow-cpu` exists solely for random-weight synthetic smoke tests
and marks their artifacts ineligible for finalization/deployment.

## Accuracy and selection

85% accuracy and 0.85 macro-F1 with >=0.70 recall in every class are **engineering
targets, not measured results or a clinical acceptance standard**. They are
checked after finalization; a failed gate exits nonzero and retains the actual
results for inspection. It never changes the split or hides difficult classes.
Do not lower targets or select another model after viewing held-out test results.

Finalize only after choosing the architecture/checkpoint using validation. A
workspace-level lock binds the test protocol to the chosen model and split
hashes and prevents any further training commands on that workspace. Temperature
and an optional low-confidence abstention threshold use validation only. Primary
test accuracy and macro-F1 include **all** test images; abstention accuracy is
reported separately with coverage. The exported `evaluation.json` includes the
confusion matrix and every class's precision, recall, F1 and support.

`flip-candidate` optionally wraps a completed eligible six-class model in one
serializable Keras graph that averages original/mirrored logits. It uses the
built-in [Keras flip operation](https://keras.io/api/ops/numpy/#flip-function),
with shared weights and inference mode for both views; no custom/Lambda layer
is needed. It reads validation images only, writes a new protected run directory,
and refuses a locked test protocol, smoke-test source or nested two-view model.
Compare validation macro-F1 against the original before selecting it. This
doubles forward passes, so accuracy benefit and serving latency must be measured;
neither improvement nor clinical validity is assumed.

No real six-class model accuracy is claimed until GPU training and this held-out
evaluation have actually run. Technical unit/smoke-test results are not model
performance evidence. Do not compare scores to the old dataset: the classes,
images and split protocol differ.

## Export and ClinIQ integration

Finalization writes `skin_six.keras`, `model_manifest.json`, `class_names.json`,
`train_config.json`, `evaluation.json`, `evaluation.md` and `runtime_smoke.json`.
It loads the saved artifact through the actual `SkinPredictor` and makes a real
validation-image prediction before declaring the export compatible.

After inspecting a passing evaluation, copy the three model/metadata files into
`skin/artifacts/six-class/`. This is the new default artifact location. Alternatively
set `SKIN_MODEL_PATH` and `SKIN_MANIFEST_PATH` to files from the same export and
restart the service. Legacy artifacts in the parent directory are not loaded by
default; absent six-class weights mean Skin health is unavailable and prediction
returns 503, while Voice/MRI continue to start. Model files and source images are
Git-ignored. No fake model manifest or invented accuracy is shipped.
Serving rejects any class names or index order other than the agreed six, even
when an incompatible model is configured manually.

```powershell
python -m uvicorn main:app --port 8080
curl.exe http://127.0.0.1:8080/api/ai/skin/health
curl.exe -X POST http://127.0.0.1:8080/api/ai/skin/predict `
  -F "file=@C:\path\to\skin.jpg;type=image/jpeg"
python -m pytest -v
```

Response fields remain `predictedClass`, `confidence`, `abstained`,
`abstainReason`, `topPredictions`, `probabilities`, `modelVersion`, `inferenceMs`
and `disclaimer`. The class map has exactly the six agreed names. `Normal` is an
inherited dataset label; it does not establish that a patient is healthy. A
low-confidence threshold is not a reliable detector for unseen diseases or
non-skin images. Model confidence is not the probability of a correct diagnosis.

The release has no repository-wide license; its card notes unverified upstream
photograph rights and clinically unverified labels. Preserve that provenance.
This work is a research/technical prototype, with no clinical, external-site or
demographic validation.
