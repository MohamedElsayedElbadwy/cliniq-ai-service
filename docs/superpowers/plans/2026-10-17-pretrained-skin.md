# Pretrained Skin Integration Implementation Plan

> Execute natively using superpowers:executing-plans. User explicitly requests the current feature branch, no training, no commits/push/merge/PR, and completion without another approval round. These override skill workflow defaults.

**Goal:** Research broadly, measure promising pretrained checkpoints, and serve the best practical verified choice through the existing Skin API while preserving every existing artifact. The expanded autonomous mission takes precedence over the original two-candidate scope.

**Architecture:** Add an optional `pretrained` backend behind the existing cached predictor factory. Keep `keras` as the code default, existing routes/response fields, bounded uploads, and the central application. Select the pretrained backend explicitly for the delivered local server.

**Tech stack:** Existing FastAPI/Pillow/NumPy/httpx, optional CPU PyTorch and Transformers. No changes to TensorFlow/MRI/Voice dependencies or weights.

**Final measured selection:** MAKE public checkpoint, fixed six-candidate zero-shot task,209/259correct80.69%, macro-F1.694254. Integrated behind explicit `SKIN_BACKEND=make`; normal_unverified always withholds Normal. Both weaker native classifiers remain separate research comparisons. Source default stays keras for compatibility. All259integrated score vectors match frozen candidate scores at rtol1e-5/atol1e-6.

**Spec:** Expanded mission `C:/Users/MOHAMED ELSAYED/.codex/attachments/cede27ef-f592-48fd-a995-4e83b0b441af/pasted-text-1.txt`; original implementation brief `C:/Users/MOHAMED ELSAYED/.codex/attachments/a684565d-6554-4668-9a31-dac7bcd349e2/Pasted text.txt`.

## Global constraints and fixed decisions

- Repository inspected: feature/skin-six-class, existing untracked files preserved, current main already an ancestor. No Git writes beyond normal working-file changes.
- Implemented first candidate: LaurianeMD/vit-skin-disease, revision 1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c, ViTForImageClassification, 22 actual native labels. Final serving selection remains evidence-dependent. Alternative 0xnu has184 labels but `model_type=image-classification` conflicts with its documented HF ViT loader.
- Weight343285496bytes SHA2563d85c0ef41fc65937401ae1fa5cc37a8ec139754ef5e1e21539603417c6ae29c. Verify all pinned files and use safetensors/local-only/no remote code.
- Use official ViTImageProcessor: RGB,224x224 bilinear,1/255 rescale,mean/std[.5,.5,.5]. Do not reuse Keras tensor normalization.
- Fixed BEFORE evaluation: confidence threshold.70, top-two margin.15; always abstain for native `Unknown Normal`. Heuristics are uncalibrated, cannot reliably detect all unsupported inputs, and will not be tuned on259images.
- Unknown Normal combines unknown/normal and cannot certify healthy skin. Native Tinea is broader than strict ringworm; report this mapping approximation.
- 10MiB/120million original pixels, no image persistence/logging, load once, CPU inference. Missing pretrained dependencies/weights returns503 without silently loading another model.
- Existing 259-image data is only an evaluation cohort; publisher training-image overlap is unknown. Do not assert it is independently unseen to the external checkpoint.
- Primary measured39.38% on the existing cohort. Investigate the named alternative using its actual standard torchvision ViT-B/16 state, strict safetensors loading and the declared official ViTImageProcessor metadata. Both fixed candidates keep.70/.15 rejection; only exact common label names and the explicit tinea/ringworm alias may map to references. Choosing between these explored checkpoints makes the cohort a development comparison, not a new independent final holdout.

## Review focus

- Expanded search includes HF, actual GitHub checkpoints, published clinical foundation models, Kaggle outputs, Papers With Code, and the supplied Nature paper. Record access restrictions and modality/taxonomy mismatches. No training or new user choices.
- MAKE zero-shot evaluation is fixed prospectively to the original six candidate labels and the author's quick-start prompt, scale100, confidence.70/margin.15, Normal always abstains. This is a closed candidate comparison, not a native trained classifier or validated unknown detector. Preserve the protocol file written before evaluating images; never tune prompts on this cohort.

- A missing/bad checksum or map fails closed, no runtime downloads or fallback misidentification.
- Both backends keep the same API fields; clients must inspect modelVersion and classCount for native label sets.
- Low score, small margin and Unknown Normal produce explicit null prediction/abstention; corrupt images retain400/415 and pixel caps413.
- Model loader stays cached, safetensors only; no random missing classifier weights accepted.
- Evaluation counts predictions into native classes outside the five reference diseases as errors, with no restricted-class renormalization.

### Task1: Artifact and cached backend

Files: new skin/pretrained_artifact.py, skin/pretrained.py, skin/download_pretrained.py, skin/requirements-pretrained.txt, skin/tests/test_pretrained.py; modify skin/config.py,service.py,schemas.py,.env.example.

- [x] Write tests for configuration selection, correct native label order, checksums, missing artifacts, low-confidence/margin/unknown abstention, inference failure and real route compatibility; observe RED.
- [x] Implement `verify_artifact(directory)`, atomic nonoverwriting pinned `download(directory)`, `PretrainedSkinPredictor(settings,runtime_loader=None)` and backend factory dispatch.
- [x] Install only resolved optional CPU dependencies, download verified model and run GREEN tests.
- [x] Run actual official preprocessing/CPU inference and record latency/memory and dependency compatibility.

### Task2: Fixed evaluation and delivery

Files: new skin/evaluate_pretrained.py, docs/skin/PRETRAINED_MODEL_EVALUATION.md, docs/skin/PRETRAINED_SETUP.md; small README link.

- [x] Test native-to-reference evaluation, outside-reference error counting, supported subset handling and no unsupported Normal score; observe RED then GREEN.
- [x] Verify split/image hashes and run fixed checkpoint on all259images without tuning. Compare cached locked v1 predictions on identical rows, plus exact four-label254subset.
- [x] Publish accuracy/macro-F1/precision/recall/F1/confusion/correct-errors, coverage and accepted accuracy, native labels, model audit, source links, uncertainty and overlap caveat.
- [x] Run complete shared suite including real-model tests. Review the scoped diff; no Git commit/push/PR.
- [x] Start owned central API on127.0.0.1:8080 with explicit selected MAKE config; verify live valid/corrupt/oversized uploads, Swagger HTTP/OpenAPI, health, schema and unrelated features.
- [x] Document every changed file, exact PowerShell install/download/start commands and actual response. Leave server running and deliver honest measured result.

### Expanded task3: Selected MAKE delivery

- [x] Broad source/checkpoint audit; no restriction to two initial models; screened modality, taxonomy, gates, CPU cost and license.
- [x] Freeze official single quick-start prompt, six descriptions, scale100, confidence.70/margin.15, Normal always withheld before evaluation. No prompt or threshold tuning.
- [x] Actual259image candidate evaluation;209correct50wrong, macroF1.694254, accepted203with173correct,0failures/skips.
- [x] MAKE feature tests RED2failed4errors before implementation; add pinned artifact/config, official local runtime, cached factory dispatch, generic shared policy, separate downloader, reproducible evaluator. No old artifact changes.
- [x] Re-evaluate shipped implementation on all259images and verify all score vectors match frozen candidate output.
- [x] Full native regression suite177passed0skips121.76s including actual artifacts; MRI/Voice tests passed. Fresh final review followed by one RED→GREEN fix pass for resize amplification and corrected setup dependencies.
- [x] Live central API23HTTPchecks; Swagger HTML/OpenAPI, upload errors/dimensions,108MPJPEG/exact10MiB and existing feature health/contracts; actual Eczema response and peak1.87GiBserverworking-set recorded. Interactive browser click-through and live paid-provider calls explicitly unverified.
- [x] Delivery report,21file inventory and exact setup/start/Swagger commands completed. No training, Git commit/push/merge or changes to central main.py/oldweights/maps/data/reports.

### Explicit later authorization: GitHub and Git LFS delivery

The user subsequently authorized committing/pushing the approved implementation,
then explicitly requested publication of the selected MAKE model through Git LFS.
These later instructions supersede the earlier no-commit/no-weight-publication
constraints for this release only. Preserve all previous model bytes and behavior.

- Publish only the pinned MAKE bundle, not images, credentials, old checkpoints or
  unrelated local experiments. Keep the existing MRI LFS rule.
- Default MAKE location becomes skin/artifacts/make/<pinned-revision>/ so a hydrated
  clone is usable without the old local cache. Original cache remains untouched.
- Record actual LFS SHA256 OID/size; verify normal Git stores a pointer.
- Preserve exact upstream card/config bytes across Windows line-ending conversion.
  Checkout regression first failed for README and passed after scoped attributes.
- Run the full suite from the published bundle path, verify remote LFS retrieval,
  update the existing PR metadata, and retain main without merging.
