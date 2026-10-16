# Mixed-source six-class research training

Keep the existing six-class runtime and API. This experiment adds unused
[SCIN](https://github.com/google-research-datasets/scin) cases and four mapped
diseases from [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k).
The latter is CC-BY-NC-SA-3.0 according to its authors: this mixed experiment
is restricted to noncommercial educational research. The original dataset's
upstream image rights and labels remain incompletely verified.

Use an isolated Colab/Kaggle GPU runtime with `skin/training/requirements-cloud.txt`.
Fetch an exact reviewed feature-branch commit and retain its SHA with the run.
No model, photograph, case-level CSV or prediction file belongs in Git.

## Freeze the data

First prepare the pinned original six-class workspace using
`python -m skin.data.prepare_six --download --workspace workspaces/skin-six-v1`.
Obtain the original-author `fitzpatrick17k.csv` and raw image ZIP separately.
The preparation tool checks the author metadata SHA256 and every admitted image's
MD5 against that metadata; a background-removed mirror is incompatible.

```text
python -m skin.data.prepare_v2 --original workspaces/skin-six-v1 --workspace workspaces/skin-six-v2 --cache workspaces/skin-v2-cache --atlas-zip PATH_TO_RAW_ZIP --atlas-metadata PATH_TO_AUTHOR_CSV
```

The atlas arguments are optional. SCIN metadata is downloaded with pinned
checksums. All photos of the 141 historical audit cases are protected. Before
partitioning, duplicate components include encoded/pixel identities, perceptual
matches and whole SCIN cases; components touching any protected reference are
excluded, including transitive relatives. Conflicting duplicate labels are
excluded. New atlas overlaps with SCIN are also excluded. Original test images
never enter training or validation.

SCIN training requires a unique top differential label of at least .4;
validation/test require .5. Map Tinea to Ringworm with that approximation
recorded. Weak cases are training-only. All photos of a training case share its
loss mass; evaluation uses one fixed close-up-preferred image per case.
Class and source loss mass are balanced without manufacturing extra cases.
Original Normal remains a weak inherited label; no verified healthy external
controls are available. Grouping cannot prove patient independence.

## Train, lock, evaluate

```text
python -m skin.training.v2 train --workspace workspaces/skin-six-v2 --run workspaces/skin-v2-run --batch-size 32 --head-epochs 3 --finetune-epochs 8
python -m skin.training.v2 finalize --workspace workspaces/skin-six-v2 --run workspaces/skin-v2-run --output workspaces/skin-v2-export --batch-size 32
```

Training starts from ImageNet EfficientNetB2. Checkpoint selection uses
validation macro-F1 weighted .4 original six classes, .4 SCIN five diseases,
.2 atlas four diseases; without atlas it uses .5/.5. Temperature and abstention
are fitted only on validation. Finalization rejects a different training/data
protocol and freezes the exported model, manifest/calibration and split hashes
before blind predictions. Do not tune on the new blind cases or historical audit.

Use the historical baseline and unchanged 138-image audit manifest for comparison:

```text
python -m skin.training.compare_v2 --workspace workspaces/skin-six-v2 --candidate workspaces/skin-v2-export --baseline skin/artifacts/six-class --audit workspaces/skin-external-scin-v1/evaluation_manifest.json --cache workspaces/skin-v2-cache/pictures --output workspaces/skin-v2-comparison
```

The comparison verifies all scored image bytes and evaluation locks. It reports
all-image accuracy, supported-class macro-F1, recalls, confusion matrices,
image-level Wilson intervals and abstention coverage/accepted accuracy separately.
SCIN labels are dermatologist differential labels, not definitive diagnoses.
Small per-class samples and source bias limit interpretation of intervals.

Educational promotion requires all four predeclared conditions: historical
audit accuracy and five-disease macro-F1 each improve at least .05, new blind
accuracy does not fall, and original validation macro-F1 drops at most .02.
Failure keeps the existing artifact active. A full six-class external gate
cannot pass without clinically verified Normal support. No command silently
installs or clinically approves the candidate.

## Recover a lost GPU session

`python -m skin.training.recovery_v2 --help` exposes a separate continuation
command. It requires the saved compiled v2 checkpoint, its recorded SHA256,
the original training configuration, expected optimizer iteration, and identical
unfinalized data splits. It restores Adam instead of restarting the head or
using legacy weights. New checkpoint selection still uses validation only.
Record changes of batch size/device and completed versus additional epochs;
CPU continuation is not a bitwise replay of the cloud job. Finalize and compare
the recovered run using the same commands and unchanged promotion conditions.

Before finalizing, `python -m skin.training.blend_v2 --help` can select a graph
from the fixed baseline/recovered blend grid on validation only. Each branch's
logits are temperature-scaled from validation; candidate weights are 0/.25/.5/.75/1.
Selection retains the .4/.4/.2 domain objective and requires original validation
macro-F1 within .02 of v1. A tied single model is preferred. The built-in
Rescaling/Add graph remains compatible with production safe-mode loading.
A mixed graph evaluates both parents (including v1's flip-average passes),
increasing model size and inference work. Finalize the selected run, then apply
the unchanged external promotion checks; do not adjust the blend after scoring.

For the reviewed local recovery workspace, selection and export use these separate
directories. Keep the recovered checkpoint and selected graph for provenance:

```text
python -m skin.training.blend_v2 --workspace workspaces/skin-six-v2-reviewed --candidate-run workspaces/skin-v2-recovered-run --baseline skin/artifacts/six-class --run workspaces/skin-v2-selected-run --batch-size 16
python -m skin.training.v2 finalize --workspace workspaces/skin-six-v2-reviewed --run workspaces/skin-v2-selected-run --output workspaces/skin-v2-selected-export --batch-size 16
```

Selection must precede external finalization. A workspace selection lock prevents
choosing another graph after opening its test results. Do not delete that lock to
repeat model selection. Comparison must use the preserved historical baseline,
even if a later deployment replaces the active artifact directory.
