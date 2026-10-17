# Fine-tuning data analysis, 18 October 2026

The useful supervised signal is currently **40 source video hash groups**, with **21 positive and 19 negative video labels**. The 360 files in `clips/` provide appearance variants of candidate source footage. They do not establish 360 new procedures, camera angles, transactions, or independent examples. The immediate opportunity is to improve observation and event labeling, then train small, measurable components. The available labels do not yet support supervised detector, pose, scene, or evidence-grounded VLM training.

The audit was executed locally on all **402 videos**. Every file was hashed and five temporal samples were decoded. No media was uploaded and no source workbook was edited. Reproduce the aggregate results in [finetuning-data.json](finetuning-data.json) with:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/audit_finetuning.py
.\.venv\Scripts\python.exe -m pytest tests/test_augmentation_data.py -q
```

## What the data actually contains

| Source | Audited result | Training meaning |
| --- | --- | --- |
| Original videos | 42 MOV files, 40 SHA-256 groups; two groups have byte-identical copies | Copies are one source, not extra observations |
| Original labels | 42 explicit column C verdicts; 21 positive and 19 negative hash groups | Usable for exploratory video-level binary supervision |
| Weak timestamps | 22 positive point anchors across 21 positive source groups | Points are neither event intervals nor complete annotation coverage |
| Unique source duration | Approximately 1,299.867 seconds, or 21.66 minutes; 11.633–142.533 seconds per group | Estimated from OpenCV metadata, not authoritative source presentation timestamps |
| Workbook agreement | All three workbook variants agree on matched column C labels; no formula input cells | Column C remains authoritative. Empty timestamps never create negative labels |
| Possible category hints | 13/40 groups have hints, covering seven category IDs | Hints are incomplete. Unlisted categories are unknown |
| Assisted annotations | 13 event rows and 115 observation rows; all 13 events and 42 label-review rows explicitly await human review | Review queue material, not independently reviewed VLM targets |
| Augmented clips | 360 MP4s, 360 distinct byte hashes, 40 filename families with nine variants each | Family names group variants but never identify an original or its label |

The seven hinted categories have only **one to three positive source groups each**. One category also appears on three explicitly negative sources. This is direct evidence that a possible-category mention cannot become a positive category target. The audit exports no category supervision from these hints.

The nine filename transform hints are `brighter`, `dimmer`, `camera_noise`, their supplied combinations, and horizontal flips. Each hint occurs 40 times. Temporal image comparison tests both original and horizontally reversed orientation, using five fraction-aligned grayscale correlations and perceptual hashes with a duration check. Top source-candidate mean correlations range from **0.999677 to 0.999990**. A conservative margin check accepts **342 files in 38 complete families** as strong *candidates*. The remaining 18 files have close alternatives against the same fixed background. A local image inspection of representative candidates and their alternatives confirms that repeated backgrounds can score highly despite different action states. These are source suggestions, not attested provenance.

Representative source/variant images were inspected locally, including brightness, dimming, noise and a flip combination. These samples support the interpretation of appearance transformations and repeated source actions. Five samples cannot verify every frame, event visibility, or a complete unchanged timeline. Paired samples have median mean-luminance changes of **+8.19/255 for brighter** and **−12.72/255 for dimmer**. All 40 unique original groups decode at 464×848 pixels. Small-object crops and independently recorded viewpoints deserve more attention than increasing the appearance-variant count.

There is no supplied, reviewed mapping from `cashier_NN` to original cases. **All 360 augmented files therefore remain quarantined from supervised training.** No labels are inferred from numeric ordering, Cyrillic/Latin filename similarity, the most similar frame, or an augmentation's name.

## What is implemented for training

`worker/augmentation_data.py` and `scripts/audit_finetuning.py` implement the audit and reusable exports. Private artifacts are written under ignored `data/training/augmentation-audit/`:

- `manifest.json` contains immutable media identities, explicit source labels, weak anchors, source suggestions and split status.
- `augmentation-review.jsonl` contains one pending provenance/visibility task per unverified derivative.
- `reviewed-provenance.template.json` describes the explicit human-attested source mapping required for label transfer.
- `binary/train.jsonl`, `binary/validation.jsonl` and `binary/test.jsonl` currently contain **24, 8 and 8 original source groups**, respectively.
- `binary/robustness-validation.jsonl` and `binary/robustness-test.jsonl` are currently empty. Reviewed derivatives of held-out sources go there, separately from clean holdout examples.

These JSONL exports have video-level labels and weak anchors. They do **not** contain fabricated assistant responses, category IDs, visibility answers, boxes, poses, or event intervals. They are suitable inputs to a future binary/weak-supervision experiment, not drop-in evidence-grounded VLM SFT data.

To release reviewed derivatives, fill a separate provenance file with actual source and clip SHA-256 values from the private queue, reviewer identity, timezone-aware review time, and an explicit human-review attestation. Each link must attest that the full timeline is retained, its binary label remains visually supportable, its time offset is zero, its time scale is one, and its horizontal orientation was checked. The loader rejects foreign hashes, duplicate clip links, absent attestation, incomplete timelines and unsupported timing changes. Then rerun:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/audit_finetuning.py `
  --reviewed-provenance data/training/augmentation-audit/reviewed-provenance.json
```

Byte-identical copies are grouped automatically. Reviewed source links and common augmentation families are unioned **transitively before splitting**, including families joining differently labeled sources. A derivative follows its source component. If a family has unresolved strong candidate links crossing partitions, its derivatives remain quarantined. A dataset with too few separable components is quarantined instead of silently becoming an all-training dataset. Training weights sum to one per original source, so nine derivatives cannot multiply that source's influence. Clean holdout metrics and augmentation robustness metrics must be reported separately and aggregated by source group.

The default 24/8/8 partition preserves the earlier seeded source-hash holdout when every source component is a singleton. It is **exploratory**: every existing source has already been inspected or used during previous model development. Neither these holdouts nor the earlier nested folds are an untouched test. Camera and recording-session IDs are absent. Forty distinct byte hashes do not establish forty independent scenes. Collect camera/session metadata and a new held-out recording set before making generalization claims.

## Fine-tuning priorities

| Component | Opportunity with current media | Missing supervision and next experiment |
| --- | --- | --- |
| Scene calibration | High value: improve uncertainty and propose reusable zones across viewpoints | Review counter/window/safe/cash-zone polygons, allowed staff zones and `not_visible`. Include frames where a requested zone is absent. Measure zone validity and overlap on a new camera before training a small segmentation/grounding head |
| Event proposal and decision thresholds | Highest immediate value: tune temporal windows, dwell thresholds, missing-person persistence and sampling against explicit events | Review positive event starts/ends, preconditions, occlusion intervals and hard negative episodes. Positive video windows outside weak anchors remain unknown. Measure proposal recall, timing and false alerts/hour before tuning final verdict thresholds |
| Person detector and tracker | Adaptation may help occlusion, small people and boundary tracks | No boxes or track IDs are supplied. Label error-focused frames and short sequences, including entrants/exits and occlusion. Measure person recall, identity switches and missed occupancy before changing detector weights. ByteTrack association itself is configured/calibrated rather than fine-tuned as a neural ReID model |
| Pose / hand proximity | Potentially useful for proposal generation, but the original camera geometry and small hands limit evidence | No keypoints or visibility flags are supplied. Review wrists/elbows and person boxes in relevant counter crops. Keep hand contact unknown under occlusion. Measure keypoint quality and interaction proposal recall before investing in pose training |
| Cash, bag, phone, drawer, safe and counter objects | Targeted object adaptation could reduce domain misses and false positives | No reviewed boxes/masks or exhaustive negative images are supplied. Annotate tiny objects at original resolution, with occluded/ambiguous states and confusing paper/equipment. Evaluate per-object precision/recall and small-object performance. Do not label cash solely from a positive video verdict |
| VLM verifier | LoRA could improve the chosen rule predicates, source-frame grounding and structured `not_visible` behavior after review | Human-reviewed candidate clips, supplied frame IDs, tri-state predicates, exceptions and evidence answers are missing. A binary source label cannot train the requested explanation or justify a rule ID. Freeze the vision tower initially and train a small adapter only after these targets exist |
| Video-level binary head | Technically available now, with augmentation invariance as an exploratory experiment after provenance review | Previous heads generalized poorly. Keep group-aware evaluation and training-only threshold selection. Treat any gain on this repeatedly inspected corpus as provisional |

For boxes, use a reviewed detection format such as COCO with categories and image geometry; for poses, include a defined keypoint schema and visibility. These are distinct tasks with different target requirements. [MMDetection dataset guidance](https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_dataset.html), [MMPose dataset guidance](https://github.com/open-mmlab/mmpose/blob/main/docs/en/guide_to_framework.md)

For eventual VLM training, the official Qwen workflow expects media plus question/answer conversations. Train toward human-reviewed physical observations and evidence references, then let the rule registry and deterministic conditions decide a violation. Qwen supports image/video training and selectively frozen components. PEFT supports adapter training on a quantized base, including four-bit QLoRA. These are candidate techniques, not measured gains or confirmed memory fits for this workstation. [Qwen training format and controls](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md), [PEFT quantized adapter training](https://huggingface.co/docs/peft/developer_guides/quantization)

The existing four-bit GGUF inference artifact is not the same training setup as a Transformers/PEFT base checkpoint. Keep training dependencies isolated, pin the matching model revision, benchmark memory with the actual image/frame budget, and revalidate any adapter conversion for deployment. The previous 4B inference smoke already sampled **7,688 MiB of an 8,151 MiB GPU**; that measurement gives no assurance that 7B multimodal fine-tuning will fit. [Recorded hardware/runtime measurements](hybrid-validation.md)

## Augmentation use and evaluation gates

Use reviewed brightness/noise variants to test robustness, with augmentation strength varied on training sources only. A horizontal flip needs transformed spatial labels: normalized boxes become `(1 - x2, y1, 1 - x1, y2)`, polygons mirror their x coordinates, and pose annotations also exchange semantic left/right joints. Flip any scene zones with the image. Do not carry fixed left/right text instructions across a flip unchanged.

Appearance changes can hide the only visible small object. In those cases an evidence-verifier target should become `not_visible` or be excluded after review; it cannot automatically inherit the original evidence answer. A flip does not simulate a new physical camera angle, and global brightness/noise does not add new causal counterexamples. Add independent recordings of compliant and noncompliant actions, interrupted transactions, obscured objects, absent zones, benign dwell and multiple-person overlaps. Keep paired recordings from the same session together.

Earlier local experiments already expose the limits of binary-label fine-tuning:

| Existing experiment | Recorded result | Implication |
| --- | --- | --- |
| DINOv2 head | Training accuracy 1.00, held-out 8-group balanced accuracy 0.50 and specificity 0.00 | The fitted training score is not useful evidence of detection quality |
| Nested frozen DINOv2 baseline | Balanced accuracy 0.401 on 40 inspected hash groups | Appearance alone was weak under the declared grouped protocol |
| Nested R3D-18 event model | Balanced accuracy 0.393; recall 0.524, specificity 0.263 | Motion features and weak anchors did not solve event supervision |
| Nested semantic-observable head | Balanced accuracy 0.477; recall 0.429, specificity 0.526 | More semantic features still failed the declared inference gate |
| Existing hybrid 3.0 smoke evaluation | One positive missed, one negative correct; only two groups evaluated | Completed jobs establish operation, not useful violation recall |

Sources: [DINOv2 head](trained-case-evaluation.json), [nested event comparison](event-model-evaluation.json), [nested semantic comparison](semantic-model-evaluation.json), [hybrid validation](hybrid-validation.md). These reports predate the current architecture changes and are not accuracy measurements of the new implementation.

Predeclare a small set of observable rules, review the existing event queue, and collect new independent sources for those rules. First compare the frozen pipeline against tuned proposal/threshold settings. Then compare each trained component with the same frozen evaluation set and candidate budget. Report per-rule precision/recall, proposal misses, false alerts per hour, abstention/visibility coverage, event timing, evidence relevance and source-group confidence intervals. Keep the trained artifact disabled until its declared quality gate is passed. No detector, pose or VLM weights were fine-tuned in this audit, and no new quality claim is made from the augmented file count.

The [executed architecture 4.0 checks](architecture-v4-validation.md) add an immediate error-analysis queue. The short positive reference still returned no automatic finding, despite detected objects, temporal proposals and four review items. Its uncertainty occurred in visible-action verification before any context-verification call. Review proposal timing, the actual action crops and `not_visible` labels first; this observation alone cannot distinguish a missed event proposal, insufficient pixels or a verifier error. These examples can become focused VLM/pose/object supervision after review. Increasing legal-rule text or training on the binary verdict does not resolve that distinction.
