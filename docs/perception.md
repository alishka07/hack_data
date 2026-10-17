# Local perception, architecture 4.0

The current pipeline uses CPU YOLOX person detection, genuine ByteTrack from `trackers.ByteTrackTracker`, local RTMPose ONNX and sparse OWLv2 object detection. Models are loaded from local, integrity-checked files. Perception releases its optional models before the verifier runs. The existing adjacent-frame IoU association remains an explicitly reported fallback when ByteTrack dependencies are unavailable; it is never labeled ByteTrack.

Run setup once with network access:

```powershell
.\.venv\Scripts\python.exe scripts/setup_perception.py
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements.perception.txt
.\.venv\Scripts\python.exe scripts/setup_architecture_models.py --pose --objects --check
```

OWLv2 uses an isolated CPU Torch/Transformers environment. The already installed `.venv-train` is discovered automatically; `FORTE_OBJECT_PYTHON` selects another interpreter. It communicates only a fixed vocabulary and JPEG pixels over an owned local pipe. Each supplied object frame has matching pose sampling and source-index metadata, including frames where no object is detected. No interpolation fills missing detections. Original pixels remain available for evidence and focused crops.

| Setting | Default / meaning |
| --- | --- |
| `FORTE_PERCEPTION_FPS` | 5 source-PTS samples/sec; final source frame included |
| `FORTE_TRACKER` | `auto`; choose `bytetrack` to require it or `adjacent_iou` explicitly |
| `FORTE_PERSON_THRESHOLD` / `FORTE_PERSON_LOW_THRESHOLD` | 0.35 activation / 0.1 association recovery |
| `FORTE_POSE_BACKEND` | `auto`, `rtmpose`, or `disabled` |
| `FORTE_POSE_MODEL` / `FORTE_POSE_FPS` / `FORTE_POSE_THRESHOLD` | Local ONNX / 2 fps plus object samples / 0.3 raw SimCC peak |
| `FORTE_OBJECT_BACKEND` | `auto`, `owlv2`, or `disabled` |
| `FORTE_OBJECT_MODEL` / `FORTE_OBJECT_PYTHON` | Local manifest directory / isolated CPU interpreter |
| `FORTE_OBJECT_FPS` / `FORTE_OBJECT_THRESHOLD` | 0.2 fps / 0.15 uncalibrated detection score |
| `FORTE_OBJECT_INPUT_SIZE` | 480, positional embeddings interpolated; native 960 configurable |
| `FORTE_OBJECT_THREADS` | 4 CPU threads |
| `FORTE_PERCEPTION_SECONDS` | 90 seconds with OWLv2 assets enabled, otherwise 45; accepted-job deadline remains 300 |

These are operating settings, not empirically calibrated detection thresholds. Native-960 OWLv2 was too slow for this laptop's proposed CPU cadence; 480 preserves the functioning local path with unvalidated small-object recall. Sparse object inputs use a separate source-detail cache capped at height 1080. The detailed architecture and current validation are in [architecture-v4.md](architecture-v4.md).

Every capability is reported as enabled, disabled or unavailable with actual model/package provenance. Explicitly requested missing or corrupt models fail; automatic discovery can report unavailable without inventing detections. A missing person, hidden wrist, object persistence or zone count never establishes an absence, cash ownership, actual contact or rule breach.

## Historical 3.0 baseline and diagnostics

The remaining sections record the earlier baseline. Its model scope, tracker selection and timing results do not describe architecture 4.0.

`worker/perception.py` samples authoritative source PTS around 5 fps, includes
the actual final source frame, obtains one disk-backed cache batch, and reads
one reduced image at a time. It computes grayscale frame motion and anonymous
person boxes. Normalized boxes retain `source_index` and `timestamp_sec` in the
same shape as the existing localization output, with geometric track metadata.

The installed default is official YOLOX Nano ONNX through the existing OpenCV
DNN backend on CPU. COCO person is the only exposed class. No banknote model,
face recognition, identity matching, or role classification is implemented.
Frame-to-frame IoU association is **not ByteTrack**. It does not bridge a missed
detection, interpolate hidden people, or infer absence from an empty frame.
Track IDs are temporary visual associations, not identities.

Every disappearance becomes a source-linked unknown visibility interval. Scene
cuts/exposure jumps invalidate geometry across segments. `normalize_scene` and
the pure `apply_scene(report, scene, info)` accept source-linked per-video VLM
region hypotheses. No workstation/customer/counter/storage coordinates are
hardcoded. Unsupported or missing regions stay unknown; zone occupancy alone
cannot establish a transaction, a person's role, or cash storage.

Motion, displacement, and zone changes produce unconfirmed candidate intervals.
Same-trigger intervals merge only within 12 seconds. No motion threshold proves
that an event did not happen. Regular VLM timeline inspection remains necessary
when all proposal routes return zero candidates.

## Setup and provenance

Run once with internet available:

```powershell
.\.venv\Scripts\python.exe scripts/setup_perception.py
```

The setup script retrieves only public model/license bytes. Inference never
downloads assets or transmits frames. The local ignored asset directory contains
the ONNX, its manifest, and the upstream Apache license. Setup validates the
reviewed asset size and SHA-256; the inference loader checks the manifest hash
before loading ONNX.

| Asset | Value |
| --- | --- |
| Official release | `0.1.1rc0`, `yolox_nano.onnx` |
| Bytes | 3,659,407 |
| SHA-256 | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |
| Input | BGR, 416 × 416, 114-filled top-left letterbox |
| Output | Raw COCO head, 80 classes; only class 0 person retained |
| Code/model repository license | Apache-2.0 |

Primary references: [official YOLOX ONNX model table](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime),
[official preprocessing](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py),
[official raw-head decode](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py),
and [YOLOX license](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE).

YOLOX plus ByteTrack was genuinely replayed against geometric association in an
isolated evaluation environment. [The measured comparison](tracking-comparison.json)
uses the current `trackers.ByteTrackTracker` implementation, version 2.6.1
([Apache-2.0](https://github.com/roboflow/trackers/blob/release/stable/LICENSE)),
with compatible `supervision` 0.26.1
([MIT](https://github.com/roboflow/supervision/blob/0.26.1/LICENSE.md)). The original
[ByteTrack repository](https://github.com/FoundationVision/ByteTrack) is MIT.
This is an evaluated alternative; production still uses adjacent-frame IoU.
The current inference environment has OpenCV 5.0.0 and no Torch or ONNX Runtime.
Its OpenCV build lacks HOG/CascadeClassifier APIs. Builds with HOG can use the
actual built-in pedestrian detector. On a fresh checkout without installed ONNX
or HOG, auto mode returns explicit `motion_only` backend and unknown person
visibility. Configuring a missing, corrupt, or incompatible model explicitly
fails; it never silently produces empty detections as a successful detector run.

## Settings

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `FORTE_PERCEPTION_BACKEND` | `auto` | `auto`, `yolox`, `hog`, `motion_only` |
| `FORTE_PERCEPTION_FPS` | `5` | Requested source timeline sample rate |
| `FORTE_PERCEPTION_HEIGHT` | `320` | Decode height, further bounded by aspect ratio |
| `FORTE_PERSON_MODEL` | `models/perception/yolox_nano.onnx` | Explicit local ONNX path |
| `FORTE_PERSON_MANIFEST` | Model basename with `.json` | Provenance/format manifest |
| `FORTE_PERSON_THRESHOLD` | `.35` | Uncalibrated object × person score threshold |
| `FORTE_MOTION_THRESHOLD` | `.018` | Changed fraction above grayscale delta 22 |
| `FORTE_TRACK_MAX_GAP` | `.8` | Maximum adjacent-detection time association gap |
| `FORTE_PERCEPTION_SECONDS` | `45` | Complete perception stage budget including cache decode |

The stage checks both its own wall time and the global job deadline. An overrun
or unreadable source frame fails without a completed perception report. It does
not discard the video ending to meet a timing budget.

## Measured smoke checks, 2026-10-17

These are technical CPU smoke measurements on two private local source clips.
Only aggregate measurements are published. Labels/known event timestamps never
entered inference. These clips have not been scored for detector accuracy.

| Source duration | Sampled frames | Person boxes | Geometric tracks | Triggers | Decode | Perception | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25.933 s | 131 | 360 | 18 | 2 | 0.860 s | 2.047 s | 2.907 s |
| 142.533 s | 714 | 773 | 21 | 0 | 4.157 s | 13.250 s | 17.407 s |

Both runs used YOLOX CPU at 5 fps and included the final source frame. No scene
regions were supplied, so all four semantic regions remained unknown. The zero
triggers on the longer clip show why regular inspection cannot depend on motion.
As an unvalidated sensitivity experiment, `.005` would select 72 motion samples
there before merging. This does not establish useful events, and that setting
has not replaced the default because added VLM requests affect end-to-end time.

The measured component fits its provisional 45-second stage target on these two
runs. It does not establish the full 300-second deadline, event recall, camera
generalization, detection calibration, or sufficient visibility for omissions.
Track fragmentation is expected with occlusion and low-resolution/seated views.

Synthetic checks cover PTS-bound proposals, ending retention, unknown camera
regions, scene invalidation, source bindings, missed detections, track expiration,
raw-model format errors, corrupt assets, and explicit deadline/stage failures:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_perception.py -q --basetemp tmp/pytest-perception
```

## Actual ByteTrack comparison

`scripts/evaluate_tracking.py` replays cached person boxes from the two smoke
runs without reading images/videos, loading a detector, using the GPU, or making
network calls. It calls the genuine
[current ByteTrack API](https://trackers.roboflow.com/develop/api/trackers/#trackers.core.bytetrack.tracker.ByteTrackTracker)
with source timestamps. Three timing repetitions follow a warm-up. The isolated
environment uses Python 3.12.14, trackers 2.6.1, supervision 0.26.1, NumPy 2.5.4,
SciPy 1.18.1, and read-only access to the existing OpenCV 5.0.0 installation.
Production and training environments were not modified. Modern Supervision
removed its old `supervision.ByteTrack` API, so the maintained `trackers` API is
used instead.

| Cached clip | IoU IDs / assigned boxes | ByteTrack 0.8 s buffer IDs / assigned boxes | ByteTrack resumed gaps | IoU / ByteTrack median tracking time |
| --- | --- | --- | ---: | --- |
| 131 frames / 360 boxes | 18 / 360 | 5 / 345 | 6 | 0.0018 / 0.0318 s |
| 714 frames / 773 boxes | 21 / 773 | 4 / 743 | 6 | 0.0038 / 0.1096 s |

These counts do **not** establish tracking accuracy. Fewer IDs can indicate false
merges or suppressed detections. The buffered tracker reused IDs over 10 and 7
unobserved samples respectively; every such gap remains unknown visibility and
cannot establish presence, absence, cash lineage, or identity. The replay cannot
exercise YOLOX detections below .35 because the cache had already filtered them.

The one-frame (0.2 s) buffer diagnostic produced 96/188 IDs and assigned 203/263
boxes, indicating strong sensitivity near the sampled frame cadence. The zero
buffer configuration assigned no tracks in this implementation's timestamp mode;
the JSON records this degenerate result explicitly. Neither diagnostic is a
recommended setting. No ground-truth identities exist here, so HOTA, IDF1, MOTA,
and identity switches were not reported. Keep the conservative production IoU
default until reviewed association labels establish a benefit.

To create a separate evaluation environment, install `trackers==2.6.1` and
`supervision==0.26.1` there with their CPU dependencies. The successful environment
is `tmp/tracking-eval-env`; this exact local replay command regenerates the report:

```powershell
.\tmp\tracking-eval-env\Scripts\python.exe scripts/evaluate_tracking.py --input tmp/perception-smoke-short/perception.json --input tmp/perception-smoke-long/perception.json --output docs/tracking-comparison.json --repeats 3
```

The input caches are private ignored artifacts. An environment without them must
first run local perception on its own videos; the report contains only aggregate
diagnostics. The script never installs packages or downloads data at runtime.
