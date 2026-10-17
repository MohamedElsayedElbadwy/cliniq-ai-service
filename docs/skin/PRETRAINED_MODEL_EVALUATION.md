# Pretrained skin model evaluation — 17 October 2026

## Decision and measured result

**Selected: MAKE,209/259correct (80.69%), macro-F1.6943, compared with baseline162/259(62.55%).** The shipped fixed six-candidate backend reproduces all259frozen candidate score vectors within rtol1e-5/atol1e-6. No training, prompt tuning or threshold tuning was performed. This is exploratory model selection on a reused cohort, not an independent final test or a guarantee for new photos. MAKE is zero-shot text/image similarity with fixed descriptions, not a trained six-class head. Normal is always withheld. The first two pretrained classifiers did not improve the requested task.

The change reduces Eczema false positives on this cohort from27to10(58/68true Eczema recognized versus44/68baseline). It still has weak Ringworm precision3/17(17.65%), Rosacea precision14/28(50%), and no verified healthy/non-skin controls. Acceptance coverage is203/259(78.38%);173accepted images are correct and30are wrong (85.22% selective accuracy). Those limitations prevent treating the demo as a dependable diagnostic system.

## Identical-cohort comparison

| Model / output task | Correct | Wrong | Top-1 accuracy | Macro-F1 | Accepted /259 | Coverage | Accuracy among accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved six-class baseline | 162 | 97 | 62.55% | 0.5518 | 157 | 60.62% | 71.97% |
| LaurianeMD native22 | 102 | 157 | 39.38% | 0.4141 | 150 | 57.92% | 54.67% |
| 0xnu native184 | 80 | 179 | 30.89% | 0.3145 | 40 | 15.44% | 90.00% |
| MAKE fixed six candidates | 209 | 50 | 80.69% | 0.6943 | 203 | 78.38% | 85.22% |

Top-1 accuracy includes all259images, even when the API withholds the class. Selective accuracy applies only to accepted predictions: 0xnu's90% applies to40images and rejects219; it is not90% overall accuracy. These models solve different native output tasks. MAKE and baseline choose among six candidates, while the other checkpoints compete across22/184native labels. This table measures utility for the requested five diseases; it does not establish global superiority across all skin diseases.

## Cohort and leakage limits

- The exact existing reviewed test.csv order was retained:259images,204atlas and55SCIN; Acne78, Eczema68, Psoriasis92, Rosacea16, Ringworm5, Normal0. Every scored image byte hash and split hashes were verified.
- The baseline cached logits reproduce its locked62.5483% accuracy and.551779macro-F1 on the identical rows. No baseline retraining or refitting occurred.
- The cohort is now an exploratory model-selection/development comparison. Reusing it to choose checkpoints removes its status as a fresh independent final test. Publisher training inventories are incomplete; overlap with their pretraining cannot be ruled out.
- No candidate was trained, calibrated, or prompt/threshold-tuned on these images. MAKE protocol-before-evaluation.json was written before any MAKE image prediction. Its single prompt follows the author quick-start: `This is a skin image of {condition}`, with descriptions acne, eczema, psoriasis, rosacea, ringworm, healthy skin; normalized embeddings, fixed scale100.
- Native22mapping only aliases broader Tinea to Ringworm, explicitly approximate. Native184mapping only uses exact named diseases plus tinea/ringworm; composite/subtype outputs remain errors. No restricted-label renormalization or invented labels.
- SCIN uses top dermatologist differential labels, not independently confirmed diagnoses. Atlas/Fitzpatrick17k source bias and possible repeated patients remain. No claim of patient independence or skin-tone fairness.
- There are no clinically verified Normal, non-skin or unsupported-disease controls. Healthy-skin recognition and unknown rejection accuracy are unmeasured. Ringworm has only5images.
- Macro-F1 averages the five supported reference labels. Predictions outside those references count as errors; it is not a native22/184class macro-F1. All259images completed inference for each evaluated candidate;0failures,0skips.

Split SHA256 values:
```json
{
  "train": "f32de1281e6b43a2d916a3aafa31d830f32f587c1d3bdd0de814a2a679491038",
  "val": "5a8f850af60e90c5b6ee43b777ebac51cdfd43f9b02733027dcaeadaab1ced16",
  "test": "f5fb13fd6c7c000a7dee09d7eb91b683285bfb6d7b758d54ac6e30d8d078a2eb"
}
```

## Compatible-label check

The254image exact-name check excludes Ringworm because native22Tinea is broader; it does not remove competitors or renormalize outputs. Baseline63.39%/.6759macro-F1; native2240.16%/0.5177macro-F1; native18431.50%/0.3931macro-F1. MAKE has exact descriptions for all five requested diseases; no Tinea alias is used.

## Per-class metrics and confusion matrices

### Preserved six-class baseline

| Reference disease | Images | Correct | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acne | 78 | 55 | 0.8088 | 0.7051 | 0.7534 |
| Eczema | 68 | 44 | 0.6197 | 0.6471 | 0.6331 |
| Psoriasis | 92 | 46 | 0.8070 | 0.5000 | 0.6174 |
| Rosacea | 16 | 16 | 0.5161 | 1.0000 | 0.6809 |
| Ringworm | 5 | 1 | 0.0455 | 0.2000 | 0.0741 |

Confusion: rows=true disease, columns=Acne/Eczema/Psoriasis/Rosacea/Ringworm/Other. Other includes Normal and unmatched native outputs.

| True \ Predicted | Acne | Eczema | Psoriasis | Rosacea | Ringworm | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Acne | 55 | 4 | 5 | 11 | 3 | 0 |
| Eczema | 6 | 44 | 6 | 0 | 4 | 8 |
| Psoriasis | 7 | 19 | 46 | 4 | 14 | 2 |
| Rosacea | 0 | 0 | 0 | 16 | 0 | 0 |
| Ringworm | 0 | 4 | 0 | 0 | 1 | 0 |

Largest errors: Psoriasis → Eczema: 19; Psoriasis → Ringworm: 14; Acne → Rosacea: 11; Eczema → Other native label: 8; Psoriasis → Acne: 7.

### LaurianeMD native22

| Reference disease | Images | Correct | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acne | 78 | 55 | 0.9016 | 0.7051 | 0.7914 |
| Eczema | 68 | 3 | 0.6000 | 0.0441 | 0.0822 |
| Psoriasis | 92 | 34 | 0.8095 | 0.3696 | 0.5075 |
| Rosacea | 16 | 10 | 0.7692 | 0.6250 | 0.6897 |
| Ringworm | 5 | 0 | 0.0000 | 0.0000 | 0.0000 |

Confusion: rows=true disease, columns=Acne/Eczema/Psoriasis/Rosacea/Ringworm/Other. Other includes Normal and unmatched native outputs.

| True \ Predicted | Acne | Eczema | Psoriasis | Rosacea | Ringworm | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Acne | 55 | 0 | 5 | 3 | 1 | 14 |
| Eczema | 2 | 3 | 3 | 0 | 3 | 57 |
| Psoriasis | 0 | 2 | 34 | 0 | 2 | 54 |
| Rosacea | 4 | 0 | 0 | 10 | 0 | 2 |
| Ringworm | 0 | 0 | 0 | 0 | 0 | 5 |

Largest errors: Eczema → Other native label: 57; Psoriasis → Other native label: 54; Acne → Other native label: 14; Ringworm → Other native label: 5; Acne → Psoriasis: 5.

### 0xnu native184

| Reference disease | Images | Correct | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acne | 78 | 39 | 0.8125 | 0.5000 | 0.6190 |
| Eczema | 68 | 7 | 0.5385 | 0.1029 | 0.1728 |
| Psoriasis | 92 | 31 | 0.8378 | 0.3370 | 0.4806 |
| Rosacea | 16 | 3 | 0.7500 | 0.1875 | 0.3000 |
| Ringworm | 5 | 0 | 0.0000 | 0.0000 | 0.0000 |

Confusion: rows=true disease, columns=Acne/Eczema/Psoriasis/Rosacea/Ringworm/Other. Other includes Normal and unmatched native outputs.

| True \ Predicted | Acne | Eczema | Psoriasis | Rosacea | Ringworm | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Acne | 39 | 1 | 3 | 0 | 0 | 35 |
| Eczema | 2 | 7 | 3 | 1 | 1 | 54 |
| Psoriasis | 4 | 5 | 31 | 0 | 0 | 52 |
| Rosacea | 3 | 0 | 0 | 3 | 0 | 10 |
| Ringworm | 0 | 0 | 0 | 0 | 0 | 5 |

Largest errors: Eczema → Other native label: 54; Psoriasis → Other native label: 52; Acne → Other native label: 35; Rosacea → Other native label: 10; Ringworm → Other native label: 5.

### MAKE fixed six candidates

| Reference disease | Images | Correct | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acne | 78 | 62 | 0.9538 | 0.7949 | 0.8671 |
| Eczema | 68 | 58 | 0.8529 | 0.8529 | 0.8529 |
| Psoriasis | 92 | 72 | 0.9114 | 0.7826 | 0.8421 |
| Rosacea | 16 | 14 | 0.5000 | 0.8750 | 0.6364 |
| Ringworm | 5 | 3 | 0.1765 | 0.6000 | 0.2727 |

Confusion: rows=true disease, columns=Acne/Eczema/Psoriasis/Rosacea/Ringworm/Other. Other includes Normal and unmatched native outputs.

| True \ Predicted | Acne | Eczema | Psoriasis | Rosacea | Ringworm | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Acne | 62 | 0 | 2 | 13 | 1 | 0 |
| Eczema | 0 | 58 | 5 | 0 | 5 | 0 |
| Psoriasis | 1 | 8 | 72 | 1 | 8 | 2 |
| Rosacea | 2 | 0 | 0 | 14 | 0 | 0 |
| Ringworm | 0 | 2 | 0 | 0 | 3 | 0 |

Largest errors: Acne → Rosacea: 13; Psoriasis → Ringworm: 8; Psoriasis → Eczema: 8; Eczema → Ringworm: 5; Eczema → Psoriasis: 5.

## Source-specific results

| Source | Images | Baseline accuracy | Native22accuracy | Native184accuracy | MAKE accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| atlas | 204 | 64.22% | 49.02% | 37.25% | 78.92% |
| scin | 55 | 56.36% | 3.64% | 7.27% | 87.27% |

The previous138SCINaudit47.83% and804image original-source test82.96% refer to the preserved baseline on different cohorts. They must not be mixed with259image results or presented as this checkpoint's external accuracy.

## Preprocessing and error investigation

The native22checkpoint was loaded with its official slow ViTImageProcessor:224×224direct bilinear RGB, rescale1/255, mean/std.5. Tensor shape and constant-color normalization were checked numerically. Exact class map matches loaded model config; any missing/mismatched/unexpected loading keys are rejected. Its Eczema recall3/68 remains poor despite these checks; common outside-reference predictions include Infestations Bites31, Lichen30 and Bullous24. Therefore no simple label-order or normalization fix is established.

The184checkpoint's documented HF loader conflicts with its actual torchvision parameter names. Investigation loaded the actual standard torchvision ViT-B/16, strict184-class head with no random/missing parameters, and declared ViT normalization. Its poor results persist. Broad/composite taxonomy differences explain part of the requested-task mismatch; these categories were not silently collapsed.

MAKE uses the official OpenCLIP local config, strict safetensors checkpoint loading,224shortest-side bicubic resize, center crop and CLIP mean/std. Non-square upload acceptance does not mean every part of the image survives the model's official center crop. Text embeddings are fixed and computed once. Confidence is conditional on six candidate descriptions and cannot certify an unsupported image.

Image-source shift, ambiguous labels, different class definitions, image framing, lighting and absent negative controls remain substantial issues. The evidence does not justify claiming an accuracy target or clinical readiness. A new independently collected/cohort-separated labeled set would be needed for an unbiased estimate after model selection.

## Timing and size

| Checkpoint | Weight bytes | Observed load | Median inference | P95 inference |
| --- | ---: | ---: | ---: | ---: |
| Native22 |343285496|6.19s warm cache|257.1ms|319.7ms|
| Native184 |343778400|not separately recorded in aggregate|220.0ms|260.0ms|
| MAKE |598516980|1.98s warm cache|238.7ms|297.8ms|

Native22separate cold smoke:43.01s load,653.71MiB peak process working set; first inference3202.53ms. Timing depends on disk cache, imports and other laptop load. These are local CPU observations, not minimum memory guarantees. The central API also loads MRI. Inference is serialized; use a single server worker and avoid reload on8GBRAM.

## Checkpoint provenance and native labels

- Native22revision: `1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c`; weights SHA256 `3d85c0ef41fc65937401ae1fa5cc37a8ec139754ef5e1e21539603417c6ae29c`. All four pinned artifact files are size/hash-verified, including processor and class-map config.
- Native184revision: `24d4efecac881e2d483a9539887d7795e2e06259`; weights SHA256 `f67481f3ba29f335f835d26e20f87596f3382228700c4e32164b9a6182161038`; labels/config mapping verified, strict native architecture loading.
- MAKE revision: `5efa1915f1d0128451ddd61242aad8475bfb8e1a`; weights SHA256 `4ed233070183558b9f9e4572bdb2676f7da0f2c463d9b4d436b169e62700bf47`; public safetensors and official local config.
- Native22labels: Acne, Actinic Keratosis, Benign Tumors, Bullous, Candidiasis, Drug Eruption, Eczema, Infestations Bites, Lichen, Lupus, Moles, Psoriasis, Rosacea, Seborrh Keratoses, Skin Cancer, Sun Sunlight Damage, Tinea, Unknown Normal, Vascular Tumors, Vasculitis, Vitiligo, Warts.
- Native184full184labels and all machine-readable metrics appear in [pretrained_evaluation.json](pretrained_evaluation.json). They are not exposed as newly validated disease coverage.
- MAKE zero-shot candidates are Acne, Eczema, Psoriasis, Rosacea, Ringworm, Normal; healthy-skin status has not been validated.

## Sources investigated and licensing

| Candidate / primary source | Availability and task evidence | Decision |
| --- | --- | --- |
| [LaurianeMD/vit-skin-disease](https://huggingface.co/LaurianeMD/vit-skin-disease) | Public ungated; all 343,285,496 weight bytes downloaded and SHA256 verified; standard ViT-B/16, official slow ViTImageProcessor; 22 native labels; card declares Apache-2.0. Actual full-cohort accuracy39.38%. | Implemented optional backend, but measured below preserved62.55% baseline. |
| [0xnu/skincare-detection](https://huggingface.co/0xnu/skincare-detection) | Public ungated; all343,778,400 weight bytes downloaded/hash verified;184-label mapping verified. Actual state is torchvision ViT-B/16 despite incompatible HF architecture instructions. Strict native loading succeeds. Apache-2.0 declared. Full-cohort accuracy30.89%. | Rejected as replacement; no overlapping/composite subtype merging to inflate performance. |
| [MAKE](https://github.com/SiyuanYan1/MAKE), [checkpoint](https://huggingface.co/Xieji-Li/MAKE), [paper](https://arxiv.org/abs/2505.09372) | Clinical dermatology vision-language model; public ungated598,516,980-byte safetensors; native OpenCLIP ViT-B/16+text encoder; official normalization/shortest-side bicubic resize and center crop. HF README states MIT, while underlying Derm1M data is CC-BY-NC-4.0. | Promising CPU candidate selected for actual evaluation, no training. Fixed author quick-start prompt and six candidate descriptions saved before evaluation. Zero-shot scores are conditional on supplied candidate list, not clinical probabilities or a trained six-class head. |
| [DermFM-Zero](https://github.com/SiyuanYan1/DermFM-Zero), [checkpoint](https://huggingface.co/redlessone/DermFM-Zero) | Public metadata revision2039b405065ff5ead7e58c04ec5604bd1e4a7769, actual1,659,299,303-byte weight URL returns401 without authenticated acceptance. Card names public-data Open release but downloadable repo remains gated. Requires bundled custom OpenCLIP fork; PanDerm-Large+PubMedBERT. Card tags CC-BY-NC-4.0 but text requests CC-BY-NC-ND4.0/no redistribution. | Access restriction confirmed; do not bypass gate or invent credentials/affiliation. Alternative sought as instructed. No claimed local accuracy. |
| [DermLIP-PanDerm](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256) | Correct official link from PanDerm, revisionb455d4ca3b39a742ae9a8ea7d3246216c6d4bf4f. Gated auto-approval with terms/account required,783,694,172-byte safetensors. | Screened out for token-free workflow; no local inference or accuracy claim. |
| [PanDerm](https://github.com/SiyuanYan1/PanDerm) | Official repo explicitly requires fine-tuning or linear probing for a specific task; Base/Large released representations are not the required ready-to-use common-disease head. CC-BY-NC-ND4.0 research license. | No random/untrained classifier head or unauthorized new training. |
| [MONET](https://github.com/suinleelab/MONET), [checkpoint](https://huggingface.co/chanwkim/monet) | Public ungated revision1d1efd0b8d61bde82cde22d2e93c28d73441f29e, actual weight HEAD200,1,710,537,716-byte safetensors, CLIP ViT-L/14. Concept annotation/zero-shot text similarity rather than a trained disease head. Intended research only; card warns against unconstrained deployment. | Not downloaded: substantially larger than MAKE, published downstream zero-shot comparisons favor MAKE, 8GB laptop and shared MRI runtime favor smaller candidate. No claim of measured CPU memory or local accuracy. |
| [RevelaCap/clinical-skin-condition-v1](https://huggingface.co/RevelaCap/clinical-skin-condition-v1) | Public ungated revisiona614ae0c12e8590f06d4f25b529b6b568a517903; downloadable16,350,585-byte EfficientNet-B0 state dict HEAD200; clinical224 RGB ImageNet normalization; MIT. Taxonomy merges Eczema/dermatitis, Folliculitis/acne-like, Psoriasis/papulosquamous and adds Urticaria/allergic and lesion routing. Author reports SCIN macro-F1.43 and overall.6527 on different1515 samples. | Small but taxonomy is incompatible with required five-disease truth; no Rosacea/Ringworm/Normal. Mapping Folliculitis to Acne or routing to a diagnosis would be invalid. Screened out before downloading/evaluating. |
| [User-supplied Nature HPDT paper](https://www.nature.com/articles/s41746-024-01395-z), [official code/checkpoints](https://github.com/Zakiyi/Hierarchical-skin-lesion-image-classification-with-prototypeical-decision-tree) | Official repository supplies Drive links labelled ResNet34 and ViT-B (both resolve to same best_tree_acc_model.pth file ID11w6_3kdFReIP0jS6017VwXVa555A1qKn). Paper describes65 benign/malignant lesion categories on235,268 dermoscopic images,320px input; full dataset private, subset by academic request. No broad clinical inflammation classifier established. | Modality/taxonomy mismatch; do not present dermoscopy lesion performance as common-condition smartphone accuracy. No local download/evaluation claim. |
| [OmerKurtulus/skin-disease-classification](https://github.com/OmerKurtulus/skin-disease-classification) | Actual checkpoint links exist, but five categories are Acne, Vitiligo, Hyperpigmentation, Nail Psoriasis, SJS/TEN, with dermatoscopic data described. MIT code. Its99.61% is its own test result. | Only one exact common disease; Nail Psoriasis is not interchangeable with the present cutaneous Psoriasis label. Rejected for the requested task regardless of advertised accuracy. |
| [Kaggle MobileNetV3 clinical notebook](https://www.kaggle.com/code/amritanshukush/skin-disease-classification-mobilenetv3-large), [VGG16 notebook](https://www.kaggle.com/code/karma12/skin-disease-classification-vgg16), [HAM10000 notebook](https://www.kaggle.com/code/dhruv1234/ham10000-skin-disease-classification) | Public search confirms notebook/output listings, not a verified packaged checkpoint with reliable mapping/preprocessing. Attempted unauthenticated MobileNet output endpoint returned404. HAM10000 is dermoscopy lesion taxonomy. | No unverified model weight is advertised as usable. Established public research checkpoint preferred. |
| [Papers With Code skin task](https://paperswithcode.com/task/skin-disease-classification) | Browser fetch unavailable; native GET200 returned a different generic AI research page rather than a reliable task/checkpoint listing. | Used direct papers and their official repositories instead. |

SCIN and Fitzpatrick17k resources were already inspected and are present in the reviewed local evaluation cohort. Their diagnostic uncertainty, source overlap, image rights and absent clinically verified Normal controls remain limitations. No new training dataset was added. A generic CLIP or heuristic skin gate cannot be presumed to solve unknown rejection without its own suitable non-skin/unsupported-condition evaluation; no such unsupported claim is introduced.


## Delivery

The shared central API is running on `http://127.0.0.1:8080` with explicit
`SKIN_BACKEND=make`, one worker and two Torch threads. Skin health reports ready,
six candidates and `xieji-x/MAKE@5efa1915f1d0128451ddd61242aad8475bfb8e1a`.
The old Keras artifact is preserved, not silently substituted. Source default stays
Keras for existing installs; use the documented explicit MAKE startup command.

### Automated and actual-model checks

- Final complete suite: **177 passed, zero failures/errors/skips, five upstream
  warnings,121.76seconds**. Command:
  `.venv-skin-six/Scripts/python.exe -m pytest -v -m "real_model or not real_model" --basetemp .test-tmp/make-final-all --junitxml .test-tmp/make-final-all.xml`.
  The explicit marker expression includes all four actual-artifact tests rather
  than the repository's default exclusion. MRI/Voice regression tests also passed.
- Frozen candidate versus shipped MAKE implementation: all259score vectors match
  at rtol1e-5/atol1e-6, with identical209correct/.694254macro-F1. Reproducible runner:
  `python -m skin.evaluate_make --output workspaces/make-evaluation-new`.
- Review found intermediate-resize memory amplification on extremely thin images.
  Two regression tests first failed, then passed after adding the integer
  projected-resize120MP check before OpenCLIP and preserving HTTP413 error mapping.
  Ordinary-image official preprocessing is unchanged. Both original pixels and
  projected resize must fit the resource caps; an extreme100000×1PNG is rejected
  before the huge resize. Confidence thresholds were not changed.
- Optional package installation passes `pip check`. No dependency conflicts found.
- `git diff --check` passes. HEAD remains `fec81cb` on `feature/skin-six-class`;
  main.py has no diff and the original six-class SHA256 remains
  `d3aa8ee9417f57e323465d5d36f4162a6f5b23bc57f26064942f8852c033a375`.
  No commit, push, merge or PR action was performed.

### Live HTTP and Swagger evidence

Twenty-three real HTTP requests completed against the actual shared application:

| Check | Actual result |
| --- | --- |
| `/`, `/docs`, `/openapi.json` |200; Swagger HTML present, multipart upload and413schema documented |
| Skin health |200, ready, MAKE revision above, classCount6 |
| MRI health |200, ready, modelAvailable true |
| Voice missing-input contract |422 as expected; no live Groq/provider conversation was sent |
| Five existing public SCIN smoke examples |200 with actual model scores; Acne/Eczema/Psoriasis/Rosacea/Ringworm respectively |
| User-supplied Dermatitis image |200; case response retained only in ignored local workspace |
|3000×75and75×3000PNG |200, actual inference |
|12000×9000JPEG (108MP),1689629bytes |200, actual decode/inference |
| Exact10MiB valid PNG with trailing padding |200 |
| Empty file / corrupt JPEG |400 /400 |
| Unsupported file / extension-signature mismatch |415 /415 |
|100000×1PNG / file over10MiB |413 /413 |
| Header-only original-dimension probe above120MP |413 before decoding |

The five smoke examples are functional checks, not an additional accuracy estimate.
Swagger was verified by HTTP delivery and OpenAPI inspection; an interactive
browser click-through was not performed. Automated unit/integration and actual
model tests cover the endpoint contract. A live Voice provider call and new MRI
image accuracy evaluation are outside this skin task; readiness/regression were
verified instead.

Measured shared server process after these checks: peak working set2,007,121,920
bytes (about1.87GiB), current working set736,796,672bytes at that observation.
This includes MRI and the108MP upload, not a peak guarantee under concurrent load.
Startup on this laptop can take over a minute for TensorFlow and optional imports.

Actual Eczema smoke response, unchanged response schema:

```json
{
  "predictedClass": "Eczema",
  "confidence": 0.9862341841705285,
  "abstained": false,
  "abstainReason": null,
  "topPredictions": [
    {"label": "Eczema", "probability": 0.9862341841705285},
    {"label": "Psoriasis", "probability": 0.0069607301061234625},
    {"label": "Acne", "probability": 0.0033635503795051633}
  ],
  "probabilities": {
    "Acne": 0.0033635503795051633,
    "Eczema": 0.9862341841705285,
    "Psoriasis": 0.0069607301061234625,
    "Rosacea": 0.00016493426034381022,
    "Ringworm": 0.00247390903738848,
    "Normal": 0.0008026920461103816
  },
  "modelVersion": "xieji-x/MAKE@5efa1915f1d0128451ddd61242aad8475bfb8e1a",
  "inferenceMs": 223.961,
  "disclaimer": "Educational zero-shot image classification only. Not a medical diagnosis or prescription. Scores are uncalibrated and conditional on six fixed candidate descriptions. Unsupported conditions and non-skin images can receive confident scores. Normal is not clinically verified and is always withheld."
}
```

### Changed files

Only the following source/documentation files belong to this pretrained task.
Preexisting untracked training/report artifacts remain untouched.

| File | Purpose |
| --- | --- |
| `skin/config.py` | Explicit MAKE/native22/Keras selection and separate artifact locations, limits/threshold configuration |
| `skin/service.py` | Cached backend dispatch; invalid configuration fails closed without loading backup weights |
| `skin/schemas.py` | Existing fields retained; confidence/candidate semantics documented |
| `skin/pretrained_artifact.py` | Immutable native22metadata, class map and byte/hash validation |
| `skin/pretrained.py` | Native22official runtime and shared serialized inference/rejection/error behavior |
| `skin/download_pretrained.py` | Pinned, bounded, atomic non-overwriting acquisition reused by MAKE |
| `skin/evaluate_pretrained.py` | Native output evaluation with locked baseline and exact-name subset |
| `skin/make_artifact.py` | Selected checkpoint pins, architecture/preprocessing and fixed descriptions |
| `skin/make.py` | Official local OpenCLIP inference and once-computed text embeddings; resize guard; Normal abstention |
| `skin/download_make.py` | Reproducible public checkpoint acquisition |
| `skin/evaluate_make.py` | Hash-verified fixed-cohort evaluation and optional numerical parity check |
| `skin/requirements-pretrained.txt` | Optional pinned CPU-compatible dependencies |
| `skin/tests/test_pretrained.py` | Native map/config/download/failure/endpoint/scoring/actual-artifact tests |
| `skin/tests/test_make.py` | MAKE selection/mapping/uncertainty/resize/failure and actual-checkpoint parity tests |
| `.env.example` | Explicit optional backend configuration; no secret values |
| `README.md` | Selected demo and honest comparison; historical results identified |
| `docs/skin/MAKE_SETUP.md` | Selected installation, exact PowerShell startup, Swagger, evaluation commands |
| `docs/skin/PRETRAINED_SETUP.md` | Reproducible weaker native22research backend, shared CPU dependency installation |
| `docs/skin/PRETRAINED_MODEL_EVALUATION.md` | This report |
| `docs/skin/pretrained_evaluation.json` | Aggregate machine-readable metrics, all native labels and provenance |
| `docs/superpowers/plans/2026-10-17-pretrained-skin.md` | Execution/verification plan including expanded research scope |

Image cohorts, case responses, optional virtualenv packages, local research
downloads, backup checkpoints and test logs stay outside Git in ignored paths.
After the user's explicit release request, the selected MAKE safetensors checkpoint
is distributed through Git LFS in its pinned artifact directory; its normal Git
blob is only an LFS pointer. The architecture config and upstream model card are
included as text. Other model weights retain their existing storage policy.
Original weights/maps/splits/reports, MRI, Voice and central main.py are preserved.
No private images or credentials are committed. The MAKE weight publication is
explicitly authorized and retains the original bytes, identity and model card.

### Exact PowerShell startup and Swagger procedure

Installation and download commands: [MAKE_SETUP.md](MAKE_SETUP.md). In a new
PowerShell terminal, after stopping an earlier server in its own terminal:

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

Open [Swagger](http://127.0.0.1:8080/docs), expand **POST /api/ai/skin/predict**,
click **Try it out**, choose a JPEG/PNG in multipart field **file**, and click
**Execute**. GET **/api/ai/skin/health** should name MAKE and six candidates.
The server is already running at delivery; do not launch a second copy on8080.

Known unverified items remain clinical/generalization quality outside this reused
cohort, publisher image overlap, verified Normal/non-skin/OOD sensitivity,
skin-tone/patient independence and live interactive-browser/provider testing.
Gated DermFM-Zero/DermLIP were not accessed, downloaded or evaluated; no access
restriction was bypassed. These do not block the reproducible local educational
feature and its measured comparison; they do prevent a clinical-quality claim.

## Later Git LFS release authorization

After approving the measured MAKE implementation, the requester explicitly asked
to publish the model using Git LFS. This changes artifact distribution only; it
does not introduce new training, weights, class descriptions, thresholds or image
processing. The selected bundle now lives at
`skin/artifacts/make/5efa1915f1d0128451ddd61242aad8475bfb8e1a/`.

- LFS object: `open_clip_model.safetensors`,598,516,980bytes,
  OID `sha256:4ed233070183558b9f9e4572bdb2676f7da0f2c463d9b4d436b169e62700bf47`.
- The Git blob contains only the standard three-line LFS pointer. The matching
  architecture/preprocessing JSON and upstream model card are normal Git files
  with line-ending conversion disabled to preserve checksum-pinned bytes on Windows.
- This adds the three artifact files and updates .gitattributes/.gitignore,
  the default MAKE directory in skin/config.py, the example environment and setup
  instructions. The existing MRI LFS entry remains intact; old MAKE cache and
  baseline weights remain untouched.
- Run `git lfs install` and `git lfs pull` when hydrating a checkout. Source default
  backend compatibility is unchanged: select MAKE explicitly as documented.

The upstream card is preserved verbatim, including attribution and its MIT license
statement. Its existing trailing spaces are retained because they are part of the
pinned byte hash; the per-file whitespace attribute is scoped to this vendored card.
