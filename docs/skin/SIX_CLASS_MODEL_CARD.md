# Six-class Skin model: measured research prototype

Model: `skin-six-efficientnetb2-flip-average-d3aa8ee9417f`. Evaluated on all **804 held-out images**.

Accuracy: **82.96%**; macro-F1: **0.8323**.
Engineering quality gate passed: **False** (accuracy >=85%, macro-F1 >=0.85, every class recall >=0.70).

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Acne | 0.783 | 0.865 | 0.822 | 96 |
| Eczema | 0.819 | 0.709 | 0.760 | 172 |
| Psoriasis | 0.778 | 0.795 | 0.787 | 132 |
| Rosacea | 0.857 | 0.872 | 0.864 | 117 |
| Ringworm | 0.776 | 0.812 | 0.793 | 149 |
| Normal | 0.964 | 0.971 | 0.968 | 138 |

## Protocol

Fresh ImageNet B2 training: 8 head + 17 fine-tuning epochs; epoch 12 selected. A separate fresh B0 run completed 8 + 25 epochs and scored lower. Original/mirrored mean logits were selected using validation macro-F1; the recreated local graph reproduced validation metrics on all 807 images. The finalization lock bound the selected model, class order, split hashes and quality thresholds before any held-out prediction. Calibration used validation only.

A fixed undo-class-weight offset was examined on cached validation logits before the lock. It left validation accuracy unchanged and changed macro-F1 by less than 0.001, so it was not added to the serving graph. No test tuning was performed.

Abstention coverage on test: **77.86%**. Accepted-image accuracy: **91.69%**. These selective metrics do not replace the primary metrics above.

## Artifact and compatibility

Artifact SHA256: `d3aa8ee9417f57e323465d5d36f4162a6f5b23bc57f26064942f8852c033a375`.

The export includes the Keras model, exact six-label map, manifest, actual evaluation, training configuration and a prediction produced by the real `SkinPredictor`. Weights and source images are excluded from Git. Use the artifact setup and quality-gate guidance in [SIX_CLASS_TRAINING.md](SIX_CLASS_TRAINING.md).

## Limits

After reviewing the measured failed quality gate and live API preview, the
requester explicitly approved local default activation for educational use.
The artifact's measured quality gate remains false. Weights remain outside Git;
the local activation record does not constitute clinical deployment approval.

Research prototype only. Diagnoses and upstream photograph rights were not independently verified. Patient IDs are unavailable; grouped image deduplication does not establish patient independence. No demographic or clinical validation was performed. Normal is an inherited dataset label. Training-only inspection found stock/selfie imagery and watermarks, which permit source shortcuts.

## Independent external evaluation (2026-10-16)

The unchanged model was tested through the live API on 138 frozen SCIN cases
with top dermatologist differential-label weight >=0.6, one image per case.
No encoded/pixel exact match or configured perceptual near match was detected
against all 5,372 original train/validation/test images. Agreement was
**47.83% (66/138)**, balanced accuracy **45.44%**, and macro-F1 across the five
evaluated diseases **0.4818**. Normal lacked a verified external label and was
not evaluated; Tinea-to-Ringworm mapping is approximate. This is research-label
agreement, not clinical accuracy. The >=0.8 label-weight subset also performed
poorly (49.35%, 77 cases). The existing abstention rule answered 62/138 cases
with only 54.84% agreement among answered cases, so its same-source validation
target does not transfer to this external source.

The model failed the external disease gate and needs improved training data
and retraining followed by an additional untouched external holdout. No model,
threshold or class-score adjustment was made from these test results.
See [the complete protocol, per-class results and limits](EXTERNAL_SCIN_EVALUATION.md).

[Full aggregate metrics and confusion matrix](six_class_evaluation.json).
