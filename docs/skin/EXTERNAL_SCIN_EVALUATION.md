# Independent SCIN evaluation — 2026-10-16

**Decision: Needs improved training data and a new independently held-out evaluation.**

The unchanged active model agreed with the top dermatologist-derived label on **66/138 images (47.83%)**. The Wilson 95% interval is 39.66%–56.11%. Balanced accuracy is **45.44%**; macro-F1 over the five evaluated diseases is **0.4818**.
This is agreement with research differential labels, not a clinical diagnostic-accuracy estimate. Normal was not evaluated, so this is not a six-class accuracy figure.

| Disease | Correct / cases | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| Acne | 21 / 32 | 65.62% | 65.62% | 0.6562 |
| Eczema | 22 / 40 | 55.00% | 44.90% | 0.4944 |
| Psoriasis | 11 / 32 | 34.38% | 68.75% | 0.4583 |
| Rosacea | 8 / 16 | 50.00% | 57.14% | 0.5333 |
| Ringworm | 4 / 18 | 22.22% | 33.33% | 0.2667 |

Eczema was the top prediction on **49** images; **27** were other labeled conditions. The true Eczema count was **40**.
The existing abstention rule answered **62/138** cases (44.93% coverage). Accuracy among those answered was **54.84%**; abstentions are still included in the primary argmax score above.
A prespecified sensitivity subset with dermatologist label weight >=0.8 contained **77** cases: accuracy **49.35%**, balanced accuracy **47.04%**, macro-F1 **0.4948**.

## How the independent test was frozen

[SCIN](https://github.com/google-research-datasets/scin) was collected independently from volunteer contributions with dermatologist differential labels. No SCIN image was used to fit this model. [Official label schema](https://github.com/google-research-datasets/scin/blob/main/dataset_schema.md).

Unique top dermatologist weighted label >=0.6; max 40 cases/class; SHA256(42/case_id) order; one CLOSE_UP image preferred, otherwise first image. No model predictions used.

The initial selection contained 141 cases. Two exact duplicates within SCIN and one missing source image were excluded before scoring. All remaining 138 images were compared against all 5,372 original train/validation/test images using encoded SHA256, EXIF-corrected RGB SHA256, and perceptual matches (pHash <=4 AND dHash <=6). No overlap was detected. This does not establish patient independence.

Frozen external manifest SHA256: `9ef5efb5e367467e80b5a09be99d12fddf28071fc7036d370db4a33979f2b27a`.
Frozen model SHA256: `d3aa8ee9417f57e323465d5d36f4162a6f5b23bc57f26064942f8852c033a375`.
Metadata content hashes and the complete aggregate confusion matrix are in [external_scin_evaluation.json](external_scin_evaluation.json). Raw photographs, case-level predictions and metadata remain in the ignored local workspace.

The model, class order, preprocessing, temperature and abstention threshold were unchanged. The test thresholds were recorded before scoring: accuracy >=85%, macro-F1 >=0.85, each evaluated disease recall >=0.70. This external disease gate passed: **False**. The previous 804-image same-source result (82.96%) remains unchanged; it does not measure external generalization.

## API and training implications

A separate live API parity audit on 30 fixed validation samples (five/class) matched saved evaluation probabilities within 0.000000638. The active six-class artifact and label order were correct. This supports an external-generalization problem rather than an API class-order mismatch.
Training-only inspection had already identified stock/selfie images and watermarks in the original dataset. Source cues, acquisition differences and label quality are plausible contributors; this test does not isolate their individual causal effects.

Do not raise reported accuracy by discarding errors, penalizing Eczema scores, or retuning thresholds on this holdout. For a new model, curate clinically reviewed examples across conditions, skin tones and body sites; obtain verified healthy controls for Normal; split by person/source before augmentation; retrain and calibrate on separate train/validation data; and evaluate on an additional untouched external holdout. More epochs on the same source alone are not evidence of improved generalization. No retraining or accuracy improvement is claimed by this audit.

## External images for manual testing

The local `workspaces/skin-external-samples.zip` contains Acne, Eczema, Psoriasis, Rosacea and Ringworm/Tinea SCIN examples, plus an illustrative hand photograph for Normal. Disease examples use the first eligible >=0.8-weight case in the frozen manifest, excluding genital/buttock metadata, without inspecting predictions. The Normal image has no clinician-verified normal label and is excluded from accuracy.
Individual sources, unmodified-file hashes and attribution are in the ZIP. SCIN images are under the [SCIN Data Use License](https://github.com/google-research-datasets/scin/blob/main/LICENSE). The hand reference is [Evan-Amos, Human-Hands-Front-Back.jpg](https://commons.wikimedia.org/wiki/File:Human-Hands-Front-Back.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

## Limits

- Dermatologist differential labels are not definitive clinical diagnoses.
- SCIN Tinea is broader than body ringworm; mapping is approximate.
- Normal has no verified target label; six-class generalization is not established.
- Perceptual duplicate checks cannot prove patient independence.
- Small per-class sample sizes give wide uncertainty; the sample is selected by label criteria, not representative of all patients.
- The model still has only six outputs; out-of-scope conditions and non-skin images can receive confident class scores.
- User-approved educational activation remains in place; this audit does not approve clinical use.
