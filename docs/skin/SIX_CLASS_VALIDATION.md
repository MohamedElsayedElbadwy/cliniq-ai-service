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
No real test image has been used for model training, selection or prediction.
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

## Pending work and performance claims

Held-out six-class disease accuracy is **not yet measured**. TensorFlow reports
no usable GPU on native Windows here; use
the provided Colab/Kaggle GPU notebook. No active six-class model manifest or
weights are fabricated, and the old classifier is not loaded by default.

The 85% accuracy / 0.85 macro-F1 / 0.70 minimum per-class recall gate is an
engineering target only. Synthetic smoke-test scores are not disease-classifier
performance evidence. The verified source branch is published to origin as
`feature/skin-six-class`. [Draft PR #9](https://github.com/MohamedElsayedElbadwy/cliniq-ai-service/pull/9)
was created through the owner's signed-in browser session. A private Colab copy
now has a T4 GPU, with TensorFlow 2.21.0/Keras 3.15.1 GPU discovery verified after
restarting the kernel to clear preloaded NumPy. Actual GPU training, final
held-out evaluation and artifact delivery are in progress; no accuracy is claimed yet.

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
issue in this candidate path. Disease-accuracy benefit remains unmeasured until
the trained source model is compared on validation; no test images are used.
