# Validation-selected mixed-source v2 evaluation — 2026-10-16

**Decision: Retain v1; the validation-selected candidate did not meet the predeclared educational promotion rule.**

Both artifacts were fixed before this comparison. Primary accuracy includes every retained image, including predictions that the API abstains from displaying. SCIN labels are top dermatologist differential labels, not confirmed diagnoses. No clinically verified Normal controls are available; this is not a six-class clinical accuracy estimate.

| Identical evaluation cohort | Images | v1 accuracy | v2 accuracy | v1 macro-F1 | v2 macro-F1 |
|---|---:|---:|---:|---:|---:|
| New mixed-source blind images | 259 | 62.55% | 65.64% | 0.5518 | 0.5664 |
| Historical frozen SCIN audit | 138 | 47.83% | 50.00% | 0.4818 | 0.4789 |
| Original-source validation | 807 | 83.27% | 82.03% | 0.8355 | 0.8239 |

Macro-F1 covers supported true labels: five diseases on the blind/audit cohorts, six inherited labels on original validation. The 807-image original validation cohort is not the historical 804-image test set. Original validation participated in selection and is not an independent performance estimate. The historical audit motivated retraining and is a repeat development audit; the new 259-image holdout supplies the fresh blind comparison.

| New blind source | Images | v1 accuracy | v2 accuracy | v1 macro-F1 | v2 macro-F1 |
|---|---:|---:|---:|---:|---:|
| scin | 55 | 56.36% | 63.64% | 0.3474 | 0.3564 |
| atlas | 204 | 64.22% | 66.18% | 0.6495 | 0.6509 |

## Per-disease recall on external images

| Cohort / disease | Support | v1 recall | Candidate recall |
|---|---:|---:|---:|
| blind / Acne | 78 | 70.51% | 73.08% |
| blind / Eczema | 68 | 64.71% | 69.12% |
| blind / Psoriasis | 92 | 50.00% | 53.26% |
| blind / Rosacea | 16 | 100.00% | 100.00% |
| blind / Ringworm | 5 | 20.00% | 20.00% |
| historicalAudit / Acne | 32 | 65.62% | 59.38% |
| historicalAudit / Eczema | 40 | 55.00% | 75.00% |
| historicalAudit / Psoriasis | 32 | 34.38% | 25.00% |
| historicalAudit / Rosacea | 16 | 50.00% | 43.75% |
| historicalAudit / Ringworm | 18 | 22.22% | 27.78% |

## Historical Eczema errors

| Model | Eczema top predictions | Other labels predicted Eczema | True Eczema cases |
|---|---:|---:|---:|
| v1 | 49 | 27 | 40 |
| v2 | 63 | 33 | 40 |

## Uncertainty and abstention

| Model / cohort | Image-level Wilson 95% interval | Coverage | Accuracy among answered |
|---|---:|---:|---:|
| v1 / blind | 56.51%–68.22% | 60.62% | 71.97% |
| v1 / historicalAudit | 39.66%–56.11% | 44.93% | 54.84% |
| v2 / blind | 59.66%–71.15% | 53.67% | 81.29% |
| v2 / historicalAudit | 41.77%–58.23% | 40.58% | 62.50% |

Intervals describe image-level sampling uncertainty only. Source/label bias, small classes, and possible repeated patients are not captured. New SCIN test support is particularly small for Acne/Rosacea (two cases each) and Ringworm (five). No patient-independence claim is made.

## Frozen protocol and recovered training

The corrected preparation produced 5,313 training images, 1,067 validation images and 259 new blind images. All three split CSV hashes match the original Colab configuration. Duplicate/case components were audited with protected references before exclusions; the transitive-protection fix changed no actual partition. All 141 historical SCIN case selections and all their photos were excluded from additions. Original test images were protected and never added to training/validation.

The atlas audit admitted 1,446 author-MD5-verified images after excluding 20 path/label mismatches, then excluded 79 protected overlaps, retaining 1,367 images in grouped partitions. Atlas labels cover Acne, Eczema, Psoriasis and Rosacea only. SCIN supplies five disease labels; Normal remains inherited from the original dataset.

Colab visibly completed the planned job, but its final checkpoint was not retrieved: a notebook export syntax error preceded any candidate external prediction, followed by unavailable browser tooling. The recovered branch continues the locally saved five-fine-epoch checkpoint with Adam at iteration 835, for three additional CPU epochs at batch size 8. It is not the unavailable final Colab artifact or a bitwise replay of that job.

Initial recovered checkpoint SHA256: `cc62a3cb85f9defe4dcbd24dd44098a5eed389f35bb00b1e52bb3abbc1ffd253`.
Chosen recovered checkpoint SHA256: `80a261baa2ef9724958875be0522cb9dcba08e2ae2d90eda6176d7bf361ac0d3`.
Validation-selected graph SHA256: `75bd920dc80b661af87878e4e0428c6aa84a467062f305dbdb25ab60dd8aa24e`.
Exported v2 SHA256: `096154581e8bd576dc6253b6324b08152950fa42f249d11f215be83924dd0fe5`.
Historical v1 SHA256: `d3aa8ee9417f57e323465d5d36f4162a6f5b23bc57f26064942f8852c033a375`.

Checkpoint choice used validation-domain macro-F1 weights .4 original/.4 SCIN/.2 atlas. Before external predictions, a fixed recovered-branch weight grid of 0/.25/.5/.75/1 was evaluated on validation only, after branch temperature calibration. Candidates losing more than .02 original-validation macro-F1 were rejected; ties preferred a single model. Weight 0 retains the baseline, weight 1 retains the recovered branch, and an intermediate weight combines calibrated logits. Temperature and abstention were calibrated only on validation. The export lock binds the model, data split hashes and manifest/calibration before blind predictions; comparison verifies those identities and every scored image byte.

Selected recovered-branch weight: **0.25**; validation selection score: **0.675029**. Branch temperatures (baseline, recovered): `[1.034398719612927, 0.9572534164308573]`. An intermediate-weight graph performs three EfficientNetB2 passes per image, compared with two for the baseline, and stores both branches.

## Predeclared educational promotion checks

- `auditAccuracyImprovesAtLeastFivePoints`: **False**
- `auditMacroF1ImprovesAtLeastFivePoints`: **False**
- `blindAccuracyDoesNotFall`: **True**
- `originalValidationMacroF1DropsAtMostTwoPoints`: **True**

The rule requires at least .05 improvement in both historical audit accuracy and five-disease macro-F1, no loss in new blind accuracy, and at most .02 loss in original validation macro-F1. These are engineering criteria, not clinical approval. The full six-class external gate remains false without verified Normal support.

## API and engineering verification

The full shared suite passed **152 tests, zero skips**, including the real active
artifact and safe-mode Keras blend round-trip. The central app was restarted with
the retained v1 artifact. Skin/MRI health, Swagger/OpenAPI, Voice request validation,
invalid uploads and six actual sample predictions were checked. Voice provider
behavior is covered by mocked tests; no live Groq speech call was made.

JPEG at 12000 x 10000 and 20000 x 1200, and PNG at 3000 x 8000 returned HTTP 200
through actual inference. Images above 120 million pixels and uploads above 10 MiB
returned the documented HTTP 413. These are upload/contract checks, not accuracy
measurements. The attributed manual sample bundle contains historical SCIN examples
and one clinically unverified Normal illustration; it is not a set of successful
classification examples. Its actual model responses are included without filtering.

## Sources, rights and reproduction

See [mixed-source training commands](SKIN_V2_TRAINING.md), [SCIN documentation](https://github.com/google-research-datasets/scin), and [Fitzpatrick17k author metadata](https://github.com/mattgroh/fitzpatrick17k). Atlas data is CC-BY-NC-SA-3.0; this experiment is for noncommercial educational research. Original-source upstream rights/labels remain incompletely verified. Raw photographs, case-level manifests/predictions and weights stay outside Git.

The complete aggregate confusion matrices, recalls, F1 scores, supports, uncertainty and promotion checks are in [v2_comparison.json](v2_comparison.json).
