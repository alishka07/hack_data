# Local classifier training

The supervised models learn the simplified workbook's binary case labels. Version 1 uses frozen DINOv2 image features. Version 2 uses pretrained action-video features and weak timestamp supervision. Version 3 learns from physical observations extracted by the existing local visual-language model. These tasks are separate from the legal rule gate.

## Semantic model (v3)

Stop the local web launcher before extracting a new snapshot: the single GPU model must have one owner. Training owns and stops its own model server, and refuses attachment to an existing server.

```powershell
.\.venv\Scripts\python.exe scripts/train_semantic_detector.py
.\.venv\Scripts\python.exe scripts/train_semantic_detector.py --train-only
.\.venv\Scripts\python.exe scripts/predict_semantic.py C:\path\to\video.mov --experimental
```

The first command extracts features and trains; `--train-only` reuses a complete matching snapshot without starting a GPU model. Prediction owns its own server by default. `--attach` is an explicit option for a trusted idle local server; it does not stop that server. Restart the normal launcher after training or prediction if needed.

The encoder sees only eight uniformly sampled video frames and a fixed 26-question physical-observation prompt. It receives no labels, case IDs, category annotations, rule decisions or timestamp anchors. Visible yes/no observations and unknown observations have separate numeric encodings. The training head jointly selects regularization and either the 52 main features or 377 features including pair interactions. No PCA discards the semantic axes.

[The fixed semantic protocol](semantic-model-protocol.json) retains the same five outer and four inner grouped folds as the previous comparison. Every setting, transform and threshold is chosen using inner folds. Outer predictions use fold-specific thresholds. The corpus has informed several model attempts; these results remain exploratory and can include shared cameras or sessions.

The encoder recipe hashes the exact prompt, schema, sampling, local model, projector, runtime binary, runtime settings and implementation. Private pixel-only answers are cached under that fingerprint. Old mixed-version screening outputs are not reused. Inference verifies the complete recipe and never reads labels, training caches or known-video lookup tables.

The semantic CLI and trainer use `worker/semantic_runtime.py`, an exact preserved copy of the runtime used to extract this snapshot. Its byte hash matches the trained recipe. This isolates frame-text formatting from concurrent website/runtime changes. Schema-validation compatibility is also checked before accepting the recipe.

The final artifact is `models/case-detector-v3/model.json`; the aggregate report is `docs/semantic-model-evaluation.json`. Default inference refuses a failed artifact; `--experimental` allows research use while retaining integrity checks. The binary head does not assign a category, box, legal finding or event timestamp.

The completed semantic comparison scored 47.5% accuracy (19/40), versus 40% (16/40) for the original baseline on the same folds. It missed 12 of 21 positive videos and falsely flagged nine of 19 clean videos. This is inadequate, below chance in balanced accuracy, and fails the fixed gate. The artifact remains experimental and is not enabled in the website. The temporal experiment also failed; these outcomes do not validate a reliable detector from this corpus.

## Temporal event model (v2)

Use the prepared training environment, or install its dependencies and pinned local action encoder:

```powershell
.\.venv-train\Scripts\python.exe -m pip install -r requirements.training.txt
.\.venv-train\Scripts\python.exe scripts/setup_action_encoder.py
.\.venv-train\Scripts\python.exe scripts/train_event_detector.py
.\.venv-train\Scripts\python.exe scripts/predict_event.py C:\path\to\video.mov --experimental
```

Setup downloads the official [Torchvision R3D-18](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.video.r3d_18.html) Kinetics-400 weights once and checks their SHA-256. Normal training and inference use only local files. The encoder reads 16 source frames per eight-second window, with four-second overlap. Full-frame letterboxing preserves the complete view; its crop and sampling rate differ from Kinetics evaluation.

Version 2 retains each window's action features and its difference from the current video's mean. Training uses windows near positive timestamp anchors and every window of explicit negative videos. Other windows in positive videos remain unknown. Balanced video weights prevent long videos from dominating. Weighted PCA fitted inside each training fold reduces the head to eight dimensions before regularized logistic training. Inference scans uniform windows without reading labels or known timestamps, and uses the highest window score.

[The fixed protocol](event-model-protocol.json) defines five outer video-hash-group folds and four inner folds. Both pipelines use the same folds. Inner predictions select settings and thresholds; each outer prediction uses that fold's fixed threshold. [The comparison report](event-model-evaluation.json) aggregates those outer decisions. This corpus has already been inspected and reused, so the comparison is exploratory; exact-hash grouping does not separate cameras or sessions.

The final artifact, `models/case-detector-v2/model.json`, is a separate fit on all 40 groups after evaluation. Its training selection scores are not test accuracy. Default inference refuses an artifact that failed the fixed gate; `--experimental` explicitly allows a research prediction while retaining integrity checks. The selected input window is an estimate, not a reviewed event boundary. No category or bounding-box model is trained.

Version 2 stores private numeric features under `data/training/action_features/` and private manifests/predictions under the corresponding timestamped event run. `models/action-encoder/manifest.json` retains the exact package version, weights fingerprint and source provenance. The setup retains Torchvision's license beside its assets; the previous application dependency audit does not cover this added environment.

The completed comparison found 40% accuracy for both pipelines across 40 outer predictions. The temporal model produced 11 true positives, 14 false positives, five true negatives and ten false negatives. It did not improve balanced accuracy and failed the fixed gate. It is retained as an unsuccessful research attempt, not enabled in the website.

## Whole-video baseline (v1)

## Data and split

`open/open_labels.simplified.xlsx` supplies the exact video stem, weak timestamp, explicit binary verdict and optional possible categories. The explicit verdict is authoritative. Possible categories are sparse annotations; an absent category is unknown, and a category mentioned on a negative row does not turn that row positive. The workbook is read without modification.

The 42 rows represent 40 unique video byte hashes. Exact duplicates stay in one group. Seed 17 assigns 24 groups to training, eight to validation and eight to the final test; each holdout contains four positives and four negatives. This corpus was previously inspected, so these are exploratory group holdouts, not an untouched external test set. Different-byte clips from the same camera or session can still share visual context.

## Train

Use the separate CPU training environment, local DINOv2 assets and the existing FFmpeg/ffprobe installation:

```powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install -r requirements.training.txt
.\.venv-train\Scripts\python.exe scripts/train_case_detector.py
```

Dependency installation needs network access. Feature extraction uses pinned local assets and does not download models. Training defaults to at most 24 source frames per video, full-frame letterboxing to 224 pixels, and 384 frozen features per frame. Their temporal mean, standard deviation and last-minus-first difference form a 1,152-value clip vector.

The logistic head fits standardization and balanced, regularized coefficients on the 24 training groups. Four training folds select regularization. Validation predictions calibrate the decision threshold; final test labels do not select features, coefficients, regularization or threshold. Training selection metrics and validation calibration metrics must not be presented as independent test accuracy.

Outputs:

- `models/case-detector/model.json`: encoder/sampling fingerprints, learned head and threshold.
- `docs/trained-case-evaluation.json`: aggregate split metrics and inference approval status, written after training completes.
- `data/training/features/`: private numeric feature cache keyed by encoder/sampling fingerprint and video hash.
- `data/training/<run timestamp>/manifest.json`: private source snapshot metadata and grouped split manifest.

The completed baseline scored 50% on the eight test groups: four true positives, four false positives, zero true negatives and zero false negatives. It failed the predeclared release gate and is not used by the website. The fitted model and measured report remain available for research; do not change settings against these same test results to seek a passing score.

## Predict

The prediction CLI uses the same environment and matching local encoder. It refuses this failed model by default; research predictions require an explicit flag:

```powershell
.\.venv-train\Scripts\python.exe scripts/predict_case.py C:\path\to\video.mov --experimental
```

Prediction uses video-derived features and the saved transform/head/threshold. It does not consult the workbook or match a filename or hash to a known label. Its numeric score is uncalibrated.

## Limits and asset provenance

Named categories have only one to three positive video groups each, so validated category heads are not trained. The workbook contains no bounding-box annotations. Binary classification does not establish a category, exact event time, object location or legal violation; existing visual boxes are a separate model output.

`models/dinov2/manifest.json` records the upstream source repository/revision, weights URL, SHA-256 and local license path. The pinned source revision is `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; its local `source/LICENSE` contains Apache License 2.0. Retain the license and provenance with the assets. Models, videos, features and virtual environments are ignored by Git. The earlier audit in [licenses.md](licenses.md) does not cover the new Torch training environment.
