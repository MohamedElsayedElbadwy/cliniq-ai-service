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

Research prototype only. Diagnoses and upstream photograph rights were not independently verified. Patient IDs are unavailable; grouped image deduplication does not establish patient independence. No external-site, demographic or clinical validation was performed. Normal is an inherited dataset label. Training-only inspection found stock/selfie imagery and watermarks, which permit source shortcuts.

[Full aggregate metrics and confusion matrix](six_class_evaluation.json).
