# Six-class implementation verification — updated 2026-10-16

Branch: `feature/skin-six-class`, based on latest fetched `origin/main` at
`0434192`. Active implementation: `skin.data.prepare_six` and
`skin.training.six_class`. All paths and results below concern the new protocol
except explicitly identified synthetic functional fixtures.

## Executed dataset audit

The 946,838,775-byte pinned archive SHA256 matched
`d0c5d8019d960ceccabf7dcf1dddf66ebacc0c1d7393b7340b0dbdee6b5259b1`.
Manifest SHA256 matched
`1105026c7d9b1af8939862de7e5783d883f910af94410e8bf0c436cf9da529d4`.
Every selected original image was compared to the publisher's size/SHA256 and
decoded before audit. Raw images remain ignored and were not published.

6,761 selected files → exclude 1,228 supplied `aug_*` images → 5,533 originals →
remove 161 exact duplicates → **5,372 retained images**. No unreadable images or
cross-class conflicting similarity clusters were found in this selected set.
Similar retained images stay in an indivisible split group. This is an automated
image audit, not a clinical or comprehensive manual review.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Acne | 450 | 96 | 96 |
| Eczema | 805 | 173 | 172 |
| Psoriasis | 615 | 132 | 132 |
| Rosacea | 548 | 118 | 117 |
| Ringworm | 696 | 149 | 149 |
| Normal | 647 | 139 | 138 |
| Total | 3,761 | 807 | 804 |

All three manifests are relative/portable and use explicit numeric class indexes.
Final ordering uses POSIX path strings, avoiding Windows/Linux case-sort changes.
The final frozen local run is `workspaces/skin-six-v1/`. An earlier audit folder
is retained unchanged; it is not used by the new default training commands.
Aggregate provenance and frozen split checksums: [six_class_dataset.json](six_class_dataset.json).
No real test image was used for model training or selection. The final locked
evaluation subsequently predicted all 804 held-out images once.
Patient independence remains unverified because patient IDs are unavailable.

Seeded training-only contact sheets were inspected locally for Normal and
Psoriasis (20 images each). Normal includes selfies and cosmetic/stock-style
photographs; Psoriasis includes source watermarks. These create opportunities
for background/source shortcuts. This small visual audit does not validate
diagnoses, and no clinical photographs or contact sheets were published.

## Executed code/integration checks

Isolated workspace environment: Python 3.12.14, TensorFlow 2.21.0, Keras 3.15.1,
NumPy 2.5.3, Pillow 12.3.0, Starlette 1.7.0. Runtime dependency pins were preserved.

| Check | Observed result |
|---|---|
| New data/protocol tests before implementation | Failed for missing implementation; path/checksum/isolation regressions subsequently passed |
| Default config tests before changing the active artifact location | 2 failures, then passed with six-class defaults |
| Final full shared working-tree suite: `.venv-skin-six/Scripts/python.exe -m pytest -v --basetemp .test-tmp/six-final-suite` | **113 passed, 1 real-model test deselected, 3 upstream deprecation warnings; 15.14s** |
| Full suite after replacing bounded image-buffer shuffle with global index shuffle | **114 passed, 1 deselected, 3 warnings; 81.96s**; diagnostic repeat also passed in 24.00s |
| Exact staged source exported to an isolated clean copy; full `python -m pytest -v` run from inside that copy | **106 passed, 1 deselected, 3 warnings; 6.51s**; no legacy training scripts/data needed |
| Two-stage EfficientNetB0 CPU smoke on 72 synthetic random images, one epoch per stage | Both stages completed; selected `.keras` checkpoint saved/reloaded and validation selection reproduced |
| Separate tiny synthetic model through finalization and the actual `SkinPredictor` | Six output labels/probabilities verified; deliberately weak classifier failed the default quality gate, exited nonzero, and retained reports/artifact for inspection |
| Live central Uvicorn app on local port 8086 | `/`, `/docs`, `/openapi.json`, MRI health and Skin health all returned HTTP 200; Voice/MRI/Skin routes present; MRI ready; Skin correctly unavailable without new weights |
| Dependency check using bundled pip with `--python .venv-skin-six/Scripts/python.exe check` | No broken requirements |
| Python compile checks; notebook JSON and all Python code-cell ASTs | Passed |
| Source ZIP inspection | Only 13 allowlisted source/document files; no images, weights, credentials or private split paths |
| `git diff --check`; `git diff --stat -- voice mri web` | Passed; no Voice/MRI/web changes |

The shared working tree also contains seven retained unit tests for earlier
generic training helpers. Those old training entry points and their tests are
not required by the fresh six-class source bundle. The real-model marker is
deselected by default; this is not evidence of six-class model readiness.
The three warnings are Starlette HTTPX TestClient deprecation and existing MRI
HTTP-status aliases. They are recorded without changing unrelated MRI behavior.

Windows sandbox execution stalled TestClient/native checkpoint writes; native
local execution with a workspace basetemp completed. Those restricted attempts
are not counted as passing tests. No test was removed to conceal a failure.
Synthetic fixture weights live only under ignored test/run folders and are not
copied into the active artifacts directory.

## Measured model and activation status

The final held-out run measured **82.96% accuracy, 0.8323 macro-F1 on all 804 test
images**. Minimum per-class recall was 0.7093 (Eczema). The engineering gate
therefore **failed** its accuracy and macro-F1 targets, while every class passed
the recall target. Full results and limits are recorded in
[SIX_CLASS_MODEL_CARD.md](SIX_CLASS_MODEL_CARD.md) and
[six_class_evaluation.json](six_class_evaluation.json).

The artifact was exported and reloaded successfully by the real `SkinPredictor`.
It is retained under the ignored `workspaces/skin-six-v1/export/` directory as a
research prototype, with a private Drive copy of the complete results bundle.
Default activation was initially withheld after the failed gate. After reviewing
the measured result and successful live preview, the requester explicitly
approved default local activation for educational use. The exact model and
manifest were copied into `skin/artifacts/six-class/`, their checksum/class map
were verified, and an ignored activation-decision record was saved. The measured
quality gate remains false; no clinical deployment is approved. Earlier Skin
weights are not substituted. The test lock is retained;
thresholds and model selection were not changed after viewing these results.

The 85% accuracy / 0.85 macro-F1 / 0.70 minimum per-class recall gate is an
engineering target only. Synthetic smoke-test scores are not disease-classifier
performance evidence. The verified source branch is published to origin as
`feature/skin-six-class`. [Draft PR #9](https://github.com/MohamedElsayedElbadwy/cliniq-ai-service/pull/9)
was created through the owner's signed-in browser session. A private Colab copy
provided a T4 GPU, with TensorFlow 2.21.0/Keras 3.15.1 GPU discovery verified after
restarting the kernel to clear preloaded NumPy. The execution/recovery history
below explains the completed runs and artifact provenance.

The first GPU run was interrupted before fine-tuning because inspection showed
the frozen train CSV was class-sorted and its 512-image shuffle buffer could not
mix whole-class blocks. No test prediction was made. The corrected sampler
globally permutes lightweight row indexes before image decoding, once per epoch,
using a reproducible seed; validation/test order is unchanged. A regression test
checks complete image coverage, class mixing at both ends of an epoch, and
repeatable/changing epoch order. A fresh ImageNet run in a new directory uses
source commit `0a1261057304a8aade474569efc0eac79eca8184`; no interrupted weights
are resumed. The published notebook pins this corrected source.

## GPU session recovery and review — 2026-10-16

The corrected T4 run completed all eight head epochs and logged fine-tuning
through epoch 8: validation accuracy **0.8042**, macro-F1 **0.8071**. These are
validation scores only. The temporary Colab runtime expired during an account
usage interruption; reconnecting allocated a new machine, and a notebook check
confirmed the previous workspace and checkpoint were absent. No final model or
held-out test report was recovered, and none of these scores is presented as
test accuracy. A fresh run is required on the same frozen protocol.

The notebook now downloads periodic training backups to the local browser.
Checkpoint weights and JSON metadata are published by atomic replacement;
backup polling failures cannot terminate healthy training. Manual checks
verified archive SHA256/incomplete markers, absence of source photographs,
streamed subprocess output/nonzero propagation, and successful child completion
despite deliberately malformed backup metadata. A two-stage synthetic B0 smoke
also completed with atomic checkpoint saves/reload; its scores are not disease
accuracy.

An independent read-only review found that the generic serving layer could
accept a manually configured artifact with another class map. It now requires
the exact six agreed labels in index order, before loading weights. Regression
cases reject 22 classes, three classes, swapped labels and an unknown label.
The full suite after that contract fix passed **118 tests**, with one real-model
test deselected and three upstream warnings (64.32s). The final suite including
atomic metadata publication also passed **118 tests**, one deselected and three
warnings (62.63s). The restarted notebook pins atomic-save source commit
`45ace6fb1c37b9d7defef915cf7e2d33938af69c`.

During the restarted run, the first epoch reproduced validation accuracy
0.6171/macro-F1 0.6115 and created its checkpoint archive. Colab displayed a
download progress bar, but the transfer remained pending while the foreground
kernel was occupied. No local backup is claimed until the ZIP actually reaches
disk and its embedded model checksum is verified. This limitation is documented
in the training guide; periodic archive creation alone is not durable storage.

An optional validation-only original/mirror logits-average candidate is also
available. Its six-output Keras graph reproduced a known asymmetric fixture's
exact two-view mean after safe-mode save/reload. The full suite with locked-test
and ineligible-source regressions passed **120 tests**, one deselected and three
upstream warnings (11.96s). Independent read-only review found no actionable
issue in this candidate path.

## Completed replacement GPU run — 2026-10-16

The replacement T4 run used source `45ace6fb1c37b9d7defef915cf7e2d33938af69c`,
fresh ImageNet weights, and the same verified portable split hashes. It completed
eight frozen-head epochs and 17 fine-tuning epochs, stopping after five epochs
without an improvement in validation macro-F1. The best checkpoint was selected
at fine-tuning epoch 12 and reproduced its validation metrics after reload:
**accuracy 0.83023544, macro-F1 0.83226249 on all 807 validation images**.
Checkpoint SHA256:
`1bbdfb2aec3c312a80cf580ca6095c1279227786e1b5046c8f3bd04906a57a1c`.

The optional original/mirror candidate was built and reloaded with source
`f44cf6c79e994f31b1764c96c8b9b2da79061fd6`. On the same validation images it
measured **accuracy 0.83271375, macro-F1 0.83545463**. Its checkpoint SHA256 is
`333593ce7099aaa79b5abeed364dce9d5001f7b5b32db80c8a05750e4e3a6331`.
Eczema validation recall remains 0.68786, so the improvement does not establish
that the engineering targets will pass. These are validation results, not
held-out test scores.

The fresh B0 comparison completed eight head and all 25 fine-tuning epochs.
Its best checkpoint (fine-tuning epoch 24) reproduced validation accuracy
0.80297398 and macro-F1 0.80501642. Its mirror candidate measured accuracy
0.80421314 and macro-F1 0.80551704. B2 with mirror averaging remained the best
of these four candidates by validation macro-F1. No test image was used to
choose among them. The Colab runtime subsequently disconnected while the
browser's download action was waiting for the user's save interaction.

After the user completed the browser save action, the 76,736,681-byte completed
B2 ZIP reached local disk. Its embedded best checkpoint SHA256 matched the
completed run, with `incomplete=false`. The exact class map, pinned provenance
and all three portable split checksums also matched the frozen local protocol.
This supersedes the earlier incomplete first-head-epoch backup. A copy of the
completed ZIP was uploaded into the user's authorized private `ClinIQ/skin-six`
Drive folder; metadata readback confirmed the exact size, correct parent and
`shared=false`. No source photographs were included.

The mirror graph was recreated from the recovered B2 weights without retraining.
A local batch-24 validation reproduction was interrupted after high memory use;
a separate batch-4 run reproduced the exact Colab validation metrics on all 807
images, retaining the interrupted directory and all source weights. Its exported
artifact SHA256 is
`d3aa8ee9417f57e323465d5d36f4162a6f5b23bc57f26064942f8852c033a375`.
Finalization froze that hash, all split hashes, exact class order and unchanged
85%/0.85/0.70 targets before reading test images. Validation was then used for
temperature/abstention calibration, followed by the complete 804-image test run.
Finalization exited nonzero because of the measured quality-gate failure,
after retaining the report and successfully reloading/predicting through the
production service. This is a quality-target failure, not an export/runtime crash.

The private final bundle contains only the model, metadata, measured evaluation,
runtime prediction, frozen manifests and model card; no source photographs.
Its size is 29,428,793 bytes and SHA256 is
`f74525def567da62ee8e81a082c3fdb3c4181724c31f8c3346d408a5eae8a9ca`.
Drive metadata readback confirmed exact size, expected private folder and
`shared=false`. The fresh full shared suite after delivery passed **120 tests**,
one real-model test deselected and three upstream warnings (53.75s).

## Actual exported-model HTTP verification

The existing central Uvicorn application was started with process-local
`SKIN_MODEL_PATH`/`SKIN_MANIFEST_PATH` overrides pointing at the research export;
default active artifacts were not changed. Both MRI and Skin reported ready,
with Skin reporting exactly six classes and the expected model version.
`/`, `/docs`, `/openapi.json` and both health routes returned HTTP 200; the
Voice/MRI/Skin prediction routes were present in OpenAPI. One fixed validation
image per class was uploaded using its actual JPEG/PNG media type. All six
requests returned 200 with the exact six probability keys, normalized scores
and matching model version. A corrupt JPEG returned 415. These six samples
verify the API contract and are not another accuracy estimate.

Reported CPU inference times for these sequential requests were 1.07–1.94s;
this small functional check is not a capacity benchmark. The temporary server
was stopped after verification. The earlier full suite deselected the separate
real-model marker; actual readiness was also established by explicit
non-skipping service readiness checks and live HTTP requests above.

After explicit user approval of educational default activation, the same live
checks were repeated against `skin/artifacts/six-class/` with no research-export
path override. Both models were ready, all six validation-image requests
returned 200 and the corrupt JPEG returned 415. The exact measured model hash
and unchanged failed gate were retained. A final full suite explicitly included
all markers: **121 passed, no skipped/deselected tests, three upstream warnings
(14.21s)**. The application server was stopped after verification. The weights
and ignored activation record are local; Git contains source and aggregate
evidence only. Clinical deployment was not approved.

## Upload-size regression verification (2026-10-16)

A 7,733,630-byte JPEG (3333 by 5000 pixels) was rejected by the former
5 MiB file limit despite being below the unchanged 20-million-pixel limit.
The Skin default is now 10 MiB, with bounded multipart ingestion and in-memory
uploads retained. Oversized-file errors state the configured limit, and
OpenAPI/Swagger documents HTTP 400, 413, 415, 500 and 503 with the existing
`detail` error body. Successful prediction responses are unchanged.

Regression tests first reproduced the rejection and missing error documentation.
After the fix, all six focused upload checks passed. The full shared suite,
including the real-model marker, passed **124 tests, no skipped/deselected tests,
three upstream warnings (24.38s)**. Live HTTP verification against the restarted
central application accepted the same JPEG with HTTP 200 and normalized six-class
probabilities. A synthetic file above 10 MiB returned 413 with the configured
limit in its message. Root, Swagger, OpenAPI and MRI health returned 200; Skin
reported ready with six classes. Swagger visibly showed the new limits and
documented 413 response. No input photograph was added to Git, and model weights,
measured evaluation and educational activation were unchanged. The local API
was left running on port 8080 for continued testing.
