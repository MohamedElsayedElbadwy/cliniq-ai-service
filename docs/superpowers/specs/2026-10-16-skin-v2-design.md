# Skin v2: improved data and measured generalization

The requester authorized execution of the complete data audit, retraining,
independent evaluation and final local API verification, and delegated routine
choices without additional approval questions. Execute inline in the existing
Skin feature branch; preserve all other developers' untracked work.

Keep the six labels in their existing order, preprocessing semantics (EXIF,
RGB, direct bilinear 224x224, float32 [0,255]), response schema, central app,
120 MP/10 MiB caps and optional model initialization. Do not use old 22-class
data/weights. Retain v1 and its reports for rollback and comparison.

Build an immutable v2 workspace from the original TRAIN/VALIDATION partitions
plus eligible SCIN cases. Exclude ALL cases selected for the prior external
audit (including missing/duplicate exclusions), all their images and detected
near duplicates. Reserve new SCIN cases for validation and blind test after
fingerprinting but before model prediction/training. Group by case and duplicate cluster. A person identifier is
not available: never claim patient independence. Reject ambiguous tied top
labels. Exact mapping: Acne, Eczema, Psoriasis, Rosacea, Tinea -> Ringworm.
Record differential-label weight and source for every image. Lower confidence
cases can enter training with downweighted loss; validation/test require a
unique top label >=0.5. Use only source-reviewed, nonconflicting original
images; Normal remains a weak inherited research label, not clinically verified.

SCIN has few unused Rosacea/Acne cases after protecting the old audit. Report
these counts and uncertainty rather than manufacture extra independent samples.
Fitzpatrick17k was investigated, but direct image URLs return 406 and candidate
mirrors are metadata-only or generated VQA; do not admit them without verified
original image identities and compatible labels. A subsequently verified raw
asosenge mirror has 1,466 eligible images matching author MD5, zero mismatches.
Admit only individually verified, technically readable, QC/path-aligned images
of Acne, Eczema, Psoriasis and Rosacea. Exclude overlaps with all original,
frozen and new SCIN images; cluster before splitting. Record CC-BY-NC-SA-3.0
noncommercial educational limitations. Do not repurpose cancer labels
or ungradable cases as healthy controls.

Train a fresh ImageNet EfficientNetB2 on the available Colab T4; retain periodic
checkpoints and a reproducible source commit. Choose checkpoints by equal-weight
mean of original-domain six-class validation macro-F1 and SCIN five-disease
validation macro-F1 when only those sources are available. With verified atlas
images, predeclare selection weights .4 original, .4 SCIN and .2 atlas four-class
macro-F1 before any candidate prediction. Use 3 head and 8 fine-tuning epochs
within the available free T4 runtime, retaining the best validation checkpoint.
Calibrate temperature/abstention on validation only. Freeze
model hash and calibration BEFORE scoring new blind cases and the old audits.
No external-test score or threshold tuning after scoring.

Report primary accuracy on every included image, per-class counts/recall/F1,
confusion matrix, coverage and accepted accuracy separately. Compare v1/v2 on
identical blind cases and the frozen 138-image audit after model selection.
85% accuracy, .85 macro-F1 and .70 minimum recall remain goals, not promises.
Do not label the six-class external gate passed without verified Normal support.
Promote an educational candidate only if the frozen 138-image accuracy and
five-disease macro-F1 each improve by at least .05 and blind-case accuracy is
not below v1, while original-domain validation macro-F1 falls by no more than
.02. Otherwise leave v1 active and give the actual result; do not claim success.

Complete shared tests, actual-artifact load/inference and central API checks;
leave port 8080 running and provide external image samples with attribution.
No clinical approval, image/model files in Git, paid compute purchases, shared
branch force pushes or merges are authorized by this design.
