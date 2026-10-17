# Local video architecture 4.0

Implemented on 18 October 2026. This version preserves the authenticated upload UI, `POST /api/v1/analyze`, SQLite job leases/deadlines, persistent worker and exact jury result schema. The worker now implements the proposed detector/tracker/pose/object/event/verifier/fusion pipeline. It remains a review prototype whose detection quality needs independently reviewed evaluation.

| Stage | Implemented behavior |
| --- | --- |
| Ingest | FFprobe presentation timestamps, FFmpeg 5 fps source-index sampling including the final frame, disk-backed original/detail cache; audio excluded |
| People | Official YOLOX Nano ONNX on CPU; genuine confidence-aware ByteTrack with Kalman/Hungarian association, no ReID; explicit adjacent-IoU fallback capability |
| Pose | Local RTMPose COCO-17 ONNX through CPU ONNX Runtime, sparse 2 fps plus every object sample; source-normalized joints and explicit `not_visible` |
| Objects | Local OWLv2 with fixed cash/bag/phone/drawer/safe/counter/window prompts; sparse 0.2 fps source-detail cache; isolated CPU Torch process |
| Scene | Separate schema-bound local VLM mapping for workstation/customer/counter/storage/window/safe/cash-zone regions; source-supported normalized coordinates, unknown regions and cut invalidation |
| Events | Entries/exits, sampled dwell, wrist/object proximity, stationary unattended-object candidates, previously occupied workstation without detections, multiple detections in a cash-zone proposal |
| Verification | Uniform timeline inspection plus every additional trigger; focused action/context/exception verification; at most eight images including detail crops, closed rule IDs and explicit `not_visible`/unknown states |
| Fusion | Conservative minimum of correlated action/context scores; version/model-bound empirical threshold artifacts; same-episode duplicate merging; cited action and before/after evidence selected with sharpness tie-breaking |
| Registry/output | Existing reviewed JSON rule registry, deterministic prerequisites/exceptions, 1–3 original-dimension source JPEGs, exact binary jury result and separate internal review/coverage details |

`worker/events.py` creates proposals, not rule findings. A disappearance, zero detection count, pose point near an object, or open-vocabulary cash box cannot establish a person's role, authorization, ownership, actual contact or continuous absence. An automatic finding still needs every existing rule predicate and exception check. Omission rules remain review-only. No learned model was trained or enabled merely because the architecture changed.

## Install and run

The API environment stays separate from heavy model dependencies. Install the base environment using the existing README. For the full CPU perception profile:

```powershell
# Base .venv already contains the application's cv2 and numpy versions.
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements.perception.txt
.\.venv\Scripts\python.exe scripts/setup_perception.py

# Existing .venv-train has CPU Torch. For a fresh CPU environment, follow
# requirements.training.txt before installing the object adapter dependencies.
.\.venv-train\Scripts\python.exe -m pip install transformers==4.57.6 safetensors
.\.venv\Scripts\python.exe scripts/setup_architecture_models.py --pose --objects --check
.\.venv\Scripts\python.exe scripts/run_local.py
```

Setup downloads public weights once, captures immutable provenance and per-file SHA-256 values, and runs optional local synthetic inference. Production model loading is offline. `--check-only --pose --objects` verifies the installed assets without downloading. Missing optional models appear in the capability report; explicitly requested missing/corrupt models fail. The UI coverage section reports whether hand/object analysis was available and how many activity proposals were considered. See [perception configuration](perception.md).

The default GGUF verifier remains Qwen3-VL-4B Q4_K_M. Compatible Qwen2.5-VL-7B Q4_K_M weights and their Q8 projector are also installed, independently SHA-256 verified and tested with four images on CPU. Reproduce installation with `scripts/setup_verifier_gguf.py`; the installer verifies an immutable official runtime-maintainer revision, exact asset sizes, LFS SHA-256 values and license provenance, then prints `FORTE_MODEL_PATH` and `FORTE_MMPROJ_PATH` configuration. Interrupted downloads resume and final model names are published atomically after verification. The CPU compatibility check does not establish a successful GPU fit or video accuracy; see [the recorded validation](architecture-v4-validation.md).

An isolated local Qwen2.5-VL-7B Transformers NF4 server is also implemented, with strict JSON constrained decoding, context admission checks and local-only weights. Its setup, immutable checkpoint revision and memory limitations are in [the verifier guide](vlm-runtime.md). Model metadata records actual backend, input budgets and asset-bound identity when the worker owns the runtime; an attached generic server remains explicitly unattested. This pipeline's eight-frame screening requires `FORTE_VLM_MAX_FRAMES=8`; lower generic-server limits are rejected at pipeline initialization.

OWL's native 960-pixel input was too slow on this CPU for the planned cadence. The default CPU profile uses 480 with positional-embedding interpolation and four Torch threads. Both the altered resolution and uncalibrated recall are reported in provenance. Original source pixels remain cached; resizing never invents missing banknote detail. Perception models use CPU and close before the verifier's GPU work. A full-object profile has a 90-second perception budget; the accepted-job deadline remains 300 seconds. The system fails incomplete work rather than silently dropping the ending or reporting a completed clean verdict.

## Calibration

No calibrated artifact is supplied from the current binary workbook labels. Those labels do not tell whether an individual verifier candidate or rule application is correct. The default policy preserves the deterministic evidence gate and explicitly reports uncalibrated scores.

`scripts/calibrate_verifier.py` accepts reviewed candidate outcomes as JSON or JSONL:

```json
{"event_id":"original-event-id","group_id":"original-source-family","partition":"calibration","rule_id":"active_intake_departure","score":0.81,"label":true,"reviewed":true}
```

Use a separate `validation` partition. Every source/augmentation family must stay in one partition; repeated event IDs are rejected. The default requires at least 20 source groups and five positive/negative groups per rule in each partition. It chooses a cutoff on calibration data and accepts it only if the disjoint validation data meet the target precision and positive-source requirements. These exploratory metrics do not constitute a production release guarantee or calibrated probabilities.

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_verifier.py data/reviewed-verifier-outcomes.jsonl `
  --model-id 'sha256:<exact runtime model_id>' --output data/calibration/verifier-v4.json
$env:FORTE_CALIBRATION_PATH = 'data/calibration/verifier-v4.json'
```

The artifact binds the exact model/configuration identity, pipeline version and score method. Unattested attached servers cannot load a calibrated policy. Change the pipeline version and refit after prompts, rules, sampling or the deployed model change. Rejected and insufficient-data policies are never deployed; correlated repeated model calls do not add confidence.

## Validation and data

The Python suite covers real ByteTrack low-score recovery, uncertainty after track gaps, SimCC decoding, sparse source bindings, event duration/cut behavior, same-frame wrist/object matching, missed-object persistence, calibration split leakage and source evidence selection. The browser regression uses synthetic requests to verify review decisions, authenticated JPEGs, source seeking, coverage and legacy results. Current executed results and real-model smoke measurements are recorded in [architecture-v4-validation.md](architecture-v4-validation.md).

Run a real local pipeline check without passing labels or filenames to inference:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_architecture.py C:\path\to\source.mov `
  --output docs/architecture-v4-smoke.json --attach-model
```

Omit `--attach-model` to let the script own and stop its model process. An explicitly attached server is never stopped. Private job observations stay in `data/architecture-smoke`; the published smoke output contains aggregate diagnostics. For API/upload/authentication checks against a running service, use `scripts/check_hybrid_local.py --pipeline-version 4.0`.

[The executed fine-tuning audit](finetuning-analysis.md) covers 42 original files / 40 unique source hashes and 360 appearance variants. It exports source-grouped exploratory binary splits and a provenance-review queue. Detector boxes, pose labels, scene masks, reviewed event intervals and grounded verifier answers are still missing. No new accuracy, camera generalization or fine-tuning improvement is inferred from passing tests or the augmented file count.
