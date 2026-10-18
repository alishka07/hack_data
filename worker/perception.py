"""Local CPU perception proposals; detection gaps never establish a breach."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

from worker.video import AnalysisError, Budget, FrameCache, FrameRef, VideoInfo, nearest_index
from worker.perception_models import ByteTrackAdapter, dependency_available, optional_backend


ROOT = Path(__file__).resolve().parents[1]
ZONE_NAMES = ("workstation", "customer", "counter", "storage", "window", "safe", "cash_zone", "computer")


def _config_error(message: str):
    raise AnalysisError("perception_config_invalid", message)


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


@dataclass(frozen=True)
class PerceptionConfig:
    backend: str = "auto"
    sample_rate: float = 5.0
    height: int = 320
    motion_threshold: float = .018
    track_iou: float = .25
    track_max_gap: float = .8
    score_threshold: float = .35
    stage_seconds: float = 45.0
    model_path: Path | None = None
    manifest_path: Path | None = None
    tracker: str = "auto"
    low_score_threshold: float = .1
    pose_backend: str = "auto"
    pose_model: Path = ROOT / "models/perception/rtmpose_t.onnx"
    pose_fps: float = 2.
    pose_threshold: float = .3
    object_backend: str = "auto"
    object_model: Path = ROOT / "models/perception/owlv2"
    object_fps: float = .2
    object_threshold: float = .15

    @classmethod
    def from_env(cls):
        try:
            result = cls(
                backend=os.getenv("FORTE_PERCEPTION_BACKEND", "auto"),
                sample_rate=float(os.getenv("FORTE_PERCEPTION_FPS", "5")),
                height=int(os.getenv("FORTE_PERCEPTION_HEIGHT", "320")),
                motion_threshold=float(os.getenv("FORTE_MOTION_THRESHOLD", ".018")),
                track_max_gap=float(os.getenv("FORTE_TRACK_MAX_GAP", ".8")),
                score_threshold=float(os.getenv("FORTE_PERSON_THRESHOLD", ".35")),
                stage_seconds=float(os.getenv("FORTE_PERCEPTION_SECONDS", "45")),
                model_path=Path(os.environ["FORTE_PERSON_MODEL"]) if os.getenv("FORTE_PERSON_MODEL") else None,
                manifest_path=Path(os.environ["FORTE_PERSON_MANIFEST"]) if os.getenv("FORTE_PERSON_MANIFEST") else None,
                tracker=os.getenv("FORTE_TRACKER", "auto"),
                low_score_threshold=float(os.getenv("FORTE_PERSON_LOW_THRESHOLD", ".1")),
                pose_backend=os.getenv("FORTE_POSE_BACKEND", "auto"),
                pose_model=Path(os.getenv("FORTE_POSE_MODEL", str(ROOT / "models/perception/rtmpose_t.onnx"))),
                pose_fps=float(os.getenv("FORTE_POSE_FPS", "2")),
                pose_threshold=float(os.getenv("FORTE_POSE_THRESHOLD", ".3")),
                object_backend=os.getenv("FORTE_OBJECT_BACKEND", "auto"),
                object_model=Path(os.getenv("FORTE_OBJECT_MODEL", str(ROOT / "models/perception/owlv2"))),
                object_fps=float(os.getenv("FORTE_OBJECT_FPS", ".2")),
                object_threshold=float(os.getenv("FORTE_OBJECT_THRESHOLD", ".15")))
        except (ValueError, TypeError):
            _config_error("Perception settings must be finite numbers and local paths")
        result.validate()
        return result

    def validate(self):
        if self.backend not in ("auto", "yolox", "hog", "motion_only"):
            _config_error("Unknown perception backend")
        if self.tracker not in ("auto", "bytetrack", "adjacent_iou"):
            _config_error("Unknown tracking backend")
        if self.pose_backend not in ("auto", "disabled", "rtmpose") or self.object_backend not in ("auto", "disabled", "owlv2"):
            _config_error("Unknown optional perception backend")
        if (self.model_path or self.manifest_path) and self.backend not in ("auto", "yolox"):
            _config_error("Person model paths require the auto or yolox backend")
        for value, lower, upper in ((self.sample_rate, .2, 15), (self.height, 128, 640),
                                    (self.motion_threshold, .001, .5), (self.track_iou, .05, .95),
                                    (self.track_max_gap, 0, 5), (self.score_threshold, .01, .99),
                                    (self.stage_seconds, 1, 240), (self.low_score_threshold, .01, .99),
                                    (self.pose_fps, .01, 8), (self.object_fps, .01, 5),
                                    (self.pose_threshold, .001, 10), (self.object_threshold, .01, .99)):
            if not _number(value) or not lower <= value <= upper:
                _config_error("Perception setting is outside its supported range")
        if type(self.height) is not int:
            _config_error("Perception height must be an integer")
        if self.low_score_threshold > self.score_threshold:
            _config_error("The recovery threshold must not exceed the person activation threshold")


def sample_targets(info: VideoInfo, fps: float) -> list[float]:
    """Uniform source-index sampling, including the actual final source frame."""
    if not _number(fps) or not .2 <= fps <= 15:
        _config_error("Perception sample rate is outside its supported range")
    last = info.pts[-1]
    if math.ceil(last * fps) > 20000:
        raise AnalysisError("perception_sampling_limit", "The full perception timeline exceeds the supported sample limit")
    indices = {nearest_index(info.pts, index / fps) for index in range(math.ceil(last * fps) + 1)}
    indices.add(len(info.pts) - 1)
    return [info.pts[index] for index in sorted(indices)]


def _iou(first, second):
    intersection = max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(
        0., min(first[3], second[3]) - max(first[1], second[1]))
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union > 0 else 0.


def _valid_box(box):
    return (isinstance(box, list) and len(box) == 4 and all(_number(n) and 0 <= n <= 1 for n in box)
            and box[0] < box[2] and box[1] < box[3])


class IoUTracker:
    """Conservative geometric association with explicit, uninterpolated gaps."""
    name = "adjacent_iou"
    def __init__(self, threshold=.25, max_gap=.8):
        self.threshold, self.max_gap = threshold, max_gap
        self.tracks = {}
        self.next_id = 1
        self.provenance = {"backend": self.name, "association": "adjacent-frame geometric IoU",
                           "reidentification": False, "predicted_boxes_are_evidence": False}

    def update(self, boxes: list[dict], frame: FrameRef):
        active = {key: value for key, value in self.tracks.items()
                  if frame.timestamp - value["time"] <= self.max_gap}
        scores = sorted(((_iou(detection["bbox"], track["bbox"]), i, key)
                         for i, detection in enumerate(boxes) for key, track in active.items()), reverse=True)
        assigned, used = {}, set()
        for score, index, key in scores:
            if score >= self.threshold and index not in assigned and key not in used:
                assigned[index] = key
                used.add(key)
        gaps = []
        for key, track in self.tracks.items():
            if key not in used:
                gaps.append({"start_sec": track["time"], "end_sec": frame.timestamp,
                             "source_indices": [track["source_index"], frame.index],
                             "reason": "track_detection_gap", "track_id": f"track-{key}"})
        # Do not bridge one missed sample. A reappearing box starts a fresh track.
        tracks, output = {}, []
        for index, detection in enumerate(boxes):
            key = assigned.get(index)
            if key is None:
                key, self.next_id = self.next_id, self.next_id + 1
            tracks[key] = {"bbox": detection["bbox"], "time": frame.timestamp, "source_index": frame.index}
            output.append({**detection, "track_id": f"track-{key}", "continuity": "adjacent_detection" if index in assigned else "new"})
        self.tracks = tracks
        return output, gaps

    def reset(self):
        self.tracks = {}


class MotionOnlyDetector:
    name = "motion_only"
    def __init__(self, reason):
        self.provenance = {"backend": self.name, "person_detection_available": False,
                           "reason": reason, "model_download_during_inference": False}

    def detect(self, image):
        return []


class HOGPersonDetector:
    name = "opencv_hog"
    def __init__(self):
        if not hasattr(cv2, "HOGDescriptor") or not hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
            raise AnalysisError("person_backend_unavailable", "This OpenCV build does not include the HOG pedestrian detector")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.provenance = {"backend": self.name, "opencv_version": cv2.__version__,
                           "source": "https://github.com/opencv/opencv", "license": "Apache-2.0",
                           "person_detection_available": True, "confidence_calibrated": False,
                           "limitations": "Pedestrian detector; seated/occluded people may be missed",
                           "model_download_during_inference": False}

    def detect(self, image):
        height, width = image.shape[:2]
        if height < 128 or width < 64:
            return []
        rectangles, weights = self.hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.08)
        boxes = []
        for (x, y, w, h), weight in zip(rectangles, np.asarray(weights).reshape(-1)):
            confidence = 1. / (1. + math.exp(-max(-20., min(20., float(weight)))))
            boxes.append({"bbox": [max(0., x / width), max(0., y / height),
                                    min(1., (x + w) / width), min(1., (y + h) / height)],
                          "confidence": confidence})
        return _nms(boxes, .45)


def _nms(boxes: list[dict], threshold: float) -> list[dict]:
    result = []
    for box in sorted(boxes, key=lambda box: -box["confidence"]):
        if _valid_box(box["bbox"]) and not any(_iou(box["bbox"], old["bbox"]) > threshold for old in result):
            result.append(box)
    return result


class YOLOXPersonDetector:
    """CPU ONNX reader using existing OpenCV DNN; only COCO person is exposed.

    Implements YOLOX's documented BGR / 114 top-left letterbox and raw-head
    stride decode. No ONNX Runtime, Torch, network access, or cash classes.
    """
    name = "yolox_opencv_cpu"
    def __init__(self, model: Path, manifest_path: Path, threshold=.35):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            size = manifest["input_size"]
            if (manifest["format"] != "yolox-raw-coco-v1" or manifest["license"] != "Apache-2.0"
                    or not isinstance(size, list) or len(size) != 2
                    or any(type(value) is not int or not 128 <= value <= 1280 or value % 32 for value in size)
                    or len(manifest["sha256"]) != 64 or not manifest["source_url"].startswith("https://")):
                raise ValueError("Invalid manifest")
            with model.open("rb") as model_file:
                digest = hashlib.file_digest(model_file, "sha256").hexdigest()
            if digest != manifest["sha256"]:
                raise ValueError("Model hash mismatch")
        except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise AnalysisError("person_asset_invalid", "The local person model or provenance manifest is missing or invalid") from exc
        try:
            self.net = cv2.dnn.readNetFromONNX(str(model))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise AnalysisError("person_asset_invalid", "The local ONNX person detector cannot be loaded") from exc
        self.size, self.threshold = size, threshold
        self.provenance = {"backend": self.name, "model_sha256": digest, "input_size": size,
                           "source": manifest["source_url"], "license": manifest["license"],
                           "opencv_version": cv2.__version__, "person_detection_available": True,
                           "confidence_calibrated": False, "model_download_during_inference": False,
                           "classes_used": ["person"]}

    def detect(self, image):
        height, width = image.shape[:2]
        target_h, target_w = self.size
        ratio = min(target_h / height, target_w / width)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        resized = cv2.resize(image, (int(width * ratio), int(height * ratio)))
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        try:
            self.net.setInput(blob)
            predictions = self.net.forward()
        except cv2.error as exc:
            raise AnalysisError("person_inference_failed", "The local CPU person detector failed") from exc
        return decode_yolox(predictions, self.size, (height, width), ratio, self.threshold)


def decode_yolox(output, input_size, source_size, ratio, threshold):
    """Strict raw-head contract prevents silently using an incompatible export."""
    target_h, target_w = input_size
    grid, strides = [], []
    for stride in (8, 16, 32):
        x, y = np.meshgrid(np.arange(target_w // stride), np.arange(target_h // stride))
        grid.append(np.stack((x, y), axis=-1).reshape(-1, 2))
        strides.extend([stride] * (target_h // stride) * (target_w // stride))
    grid, strides = np.concatenate(grid), np.asarray(strides)[:, None]
    output = np.asarray(output)
    if output.shape != (1, len(grid), 85) or not np.isfinite(output).all():
        raise AnalysisError("person_output_invalid", "The person model output does not match the reviewed YOLOX raw COCO format")
    prediction = output[0]
    probabilities = prediction[:, 4:]
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise AnalysisError("person_output_invalid", "The person model returned invalid object/class probabilities")
    scores = prediction[:, 4] * prediction[:, 5]
    keep = np.flatnonzero(scores >= threshold)
    # Processing all surviving anchors avoids a hidden candidate truncation.
    center = (prediction[keep, :2] + grid[keep]) * strides[keep]
    dimensions = np.exp(np.clip(prediction[keep, 2:4], -20, 20)) * strides[keep]
    height, width = source_size
    boxes = np.concatenate((center - dimensions / 2, center + dimensions / 2), axis=1) / ratio
    boxes /= np.asarray([width, height, width, height])
    boxes = np.clip(boxes, 0., 1.)
    return _nms([{"bbox": box.tolist(), "confidence": float(scores[index])} for index, box in zip(keep, boxes)], .45)


def build_detector(config: PerceptionConfig):
    config.validate()
    default = ROOT / "models" / "perception" / "yolox_nano.onnx"
    model = config.model_path or default
    manifest = config.manifest_path or model.with_suffix(".json")
    if config.backend == "yolox" or config.model_path or (config.backend == "auto" and model.is_file()):
        return YOLOXPersonDetector(model, manifest, config.score_threshold)
    if config.backend == "hog":
        return HOGPersonDetector()
    if config.backend == "auto" and hasattr(cv2, "HOGDescriptor"):
        return HOGPersonDetector()
    return MotionOnlyDetector("Explicit motion-only configuration" if config.backend == "motion_only"
                              else "No installed YOLOX asset or OpenCV HOG API; person visibility is unknown")


def build_tracker(config: PerceptionConfig):
    if config.tracker == "bytetrack" or (config.tracker == "auto" and dependency_available("trackers")
                                         and dependency_available("supervision")):
        try:
            return ByteTrackAdapter(config.sample_rate, config.track_max_gap, config.score_threshold,
                                    config.low_score_threshold, config.track_iou)
        except AnalysisError:
            if config.tracker == "bytetrack":
                raise
    tracker = IoUTracker(config.track_iou, config.track_max_gap)
    tracker.provenance["reason"] = ("Configured explicitly" if config.tracker == "adjacent_iou"
                                     else "Optional ByteTrack runtime unavailable; install requirements.perception.txt")
    return tracker


def normalize_scene(scene: dict | None, info: VideoInfo):
    """Accept only source-linked per-video zone hypotheses; no fixed geometry."""
    zones, evidence = [], []
    supplied = scene.get("zones", []) if isinstance(scene, dict) else []
    if isinstance(supplied, dict):
        supplied = [{"name": key, **value} for key, value in supplied.items() if isinstance(value, dict)]
    if not isinstance(supplied, list):
        supplied = []
    for zone in supplied:
        if not isinstance(zone, dict):
            continue
        name = zone.get("name", zone.get("zone", zone.get("kind")))
        sources = zone.get("source_indices", [])
        refs = zone.get("refs", [])
        if not sources and isinstance(refs, list):
            sources = [ref["source_index"] for ref in refs if isinstance(ref, dict)
                       and type(ref.get("source_index")) is int and 0 <= ref["source_index"] < len(info.pts)
                       and _number(ref.get("timestamp_sec"))
                       and abs(info.pts[ref["source_index"]] - ref["timestamp_sec"]) < 1e-6]
        if (name not in ZONE_NAMES or not _valid_box(zone.get("bbox")) or not isinstance(sources, list)
                or zone.get("status") in ("unknown", "contradicted") or not sources
                or any(type(i) is not int or not 0 <= i < len(info.pts) for i in sources)):
            continue
        record = {"name": name, "bbox": list(zone["bbox"]), "source_indices": sorted(set(sources)),
                  "status": "source_linked_hypothesis"}
        zones.append(record)
        evidence.extend({"source_index": index, "timestamp_sec": info.pts[index]} for index in sorted(set(sources)))
    return {"zones": zones, "evidence": evidence,
            "unknown": [name for name in ZONE_NAMES if not any(z["name"] == name for z in zones)],
            "inference": "Per-video zone hypotheses only; geometry establishes no roles or storage actions"}


def _proposal(previous: FrameRef, frame: FrameRef, reason: str, duration: float, **extra):
    return {"start_sec": max(0., previous.timestamp - 1.), "end_sec": min(duration, frame.timestamp + 1.),
            "timecode_sec": frame.timestamp, "reason": reason,
            "source_indices": sorted({previous.index, frame.index}), "status": "unconfirmed_trigger", **extra}


def merge_proposals(proposals: list[dict], gap_sec: float = .5):
    """Merge only same-trigger neighboring intervals; preserve actual anchors."""
    result = []
    for proposal in sorted(proposals, key=lambda p: (p["start_sec"], p["reason"])):
        previous = next((item for item in reversed(result) if item["reason"] == proposal["reason"]
                         and item.get("track_id") == proposal.get("track_id")
                         and proposal["start_sec"] <= item["end_sec"] + gap_sec
                         and max(item["end_sec"], proposal["end_sec"]) - item["start_sec"] <= 12), None)
        if previous is None:
            result.append({**proposal, "source_indices": list(proposal["source_indices"])})
        else:
            previous["end_sec"] = max(previous["end_sec"], proposal["end_sec"])
            previous["source_indices"] = sorted(set(previous["source_indices"] + proposal["source_indices"]))
    return sorted(result, key=lambda p: p["timecode_sec"])


def apply_scene(report: dict, scene: dict | None, info: VideoInfo) -> dict:
    """Bind a later VLM scene hypothesis without rerunning CPU perception.

    Scene-change segments never share geometry. Even supported zone hypotheses
    cannot identify people, establish customer presence/absence, or prove cash
    storage. Returned proposals remain unconfirmed triggers.
    """
    result = deepcopy(report)
    scene_map = normalize_scene(scene, info)
    changes = sorted(m["timestamp_sec"] for m in result.get("motion", []) if m.get("scene_change_suspected"))
    def segment(timestamp):
        return sum(change <= timestamp for change in changes)
    proposals = [p for p in result.get("proposals", []) if p.get("reason") != "person_zone_transition"]
    observations, previous = [], {}
    for detection in sorted(result.get("detections", []), key=lambda item: (item["timestamp_sec"], item["id"])):
        if detection.get("kind") != "person" or not _valid_box(detection.get("bbox")):
            continue
        timestamp, source = detection["timestamp_sec"], detection["source_index"]
        center = [(detection["bbox"][0] + detection["bbox"][2]) / 2,
                  (detection["bbox"][1] + detection["bbox"][3]) / 2]
        zones = [zone["name"] for zone in scene_map["zones"]
                 if all(segment(info.pts[index]) == segment(timestamp) for index in zone["source_indices"])
                 and zone["bbox"][0] <= center[0] <= zone["bbox"][2]
                 and zone["bbox"][1] <= center[1] <= zone["bbox"][3]]
        track_id = detection.get("track_id", "unknown")
        observations.append({"track_id": track_id, "source_index": source, "timestamp_sec": timestamp,
                             "zones": zones, "role": "unknown", "state": "visible_detection"})
        prior = previous.get(track_id)
        if (prior and detection.get("continuity") == "adjacent_detection" and segment(prior["timestamp"]) == segment(timestamp)
                and zones != prior["zones"]):
            proposals.append(_proposal(FrameRef(prior["source"], prior["timestamp"], Path()),
                                       FrameRef(source, timestamp, Path()), "person_zone_transition", info.duration,
                                       track_id=track_id))
        previous[track_id] = {"zones": zones, "timestamp": timestamp, "source": source}
    if changes:
        scene_map["invalidated_by_scene_change"] = True
        scene_map["unknown"] = list(ZONE_NAMES)
    result["scene"], result["track_observations"] = scene_map, observations
    result["proposals"] = merge_proposals(proposals)
    return result


def analyze_perception(info: VideoInfo, cache: FrameCache, budget: Budget, directory: Path,
                       *, scene: dict | None = None) -> dict:
    config = PerceptionConfig.from_env()
    started = time.monotonic()
    if (config.object_backend != "disabled" and not os.getenv("FORTE_PERCEPTION_SECONDS")
            and (config.object_backend == "owlv2" or (config.object_model / "manifest.json").is_file())):
        config = replace(config, stage_seconds=max(90., config.stage_seconds))
    tracker = build_tracker(config)
    # ByteTrack must receive low-score boxes to exercise its second association
    # stage. Unmatched low-score detections cannot create tracks or evidence.
    detector = build_detector(replace(config, score_threshold=config.low_score_threshold)
                              if tracker.name == "bytetrack" else config)
    pose, objects = None, None
    try:
        pose, pose_capability = optional_backend("pose", config.pose_backend, config.pose_model, config.pose_threshold)
        remaining_stage = config.stage_seconds - (time.monotonic() - started)
        if remaining_stage <= 0:
            raise AnalysisError("perception_timeout", "Perception model loading exceeded its stage budget")
        objects, object_capability = optional_backend("objects", config.object_backend, config.object_model,
                                                     config.object_threshold, min(remaining_stage, budget.remaining(2)))
        loaded_at = time.monotonic()
        return _analyze_perception(info, cache, budget, directory, scene=scene, config=config, started=started,
                                   loaded_at=loaded_at,
                                   detector=detector, tracker=tracker, pose=pose, pose_capability=pose_capability,
                                   objects=objects, object_capability=object_capability)
    finally:
        # Includes deadline, decode, inference and model-load failure paths.
        if pose is not None:
            pose.close()
        if objects is not None:
            objects.close()


def _analyze_perception(info, cache, budget, directory, *, scene, config, started, loaded_at,
                        detector, tracker, pose, pose_capability, objects, object_capability):
    scene_map = normalize_scene(scene, info)
    targets = sample_targets(info, config.sample_rate)
    # These are tiny disk references, not decoded image arrays. One decode serves
    # the complete timeline; only one image and a small previous gray are held.
    height = max(128, min(config.height, int(640 * info.height / info.width)))
    frames = cache.get(targets, height=height)
    object_image_refs = {}
    if objects is not None:
        # A second sparse decode preserves detail that a 320px motion stream
        # cannot recover. These references share the authoritative source PTS.
        sparse_targets, next_target = [], -1.
        for frame in frames:
            if frame.timestamp + 1e-6 >= next_target:
                sparse_targets.append(frame.timestamp)
                next_target = frame.timestamp + 1. / config.object_fps
        object_image_refs = {frame.index: frame for frame in cache.get(sparse_targets, height=min(info.height, 1080))}
    decoded_at = time.monotonic()
    detections, proposals, gaps, motions, observations, poses, frame_samples, object_samples = [], [], [], [], [], [], [], []
    next_pose, next_objects = -1., -1.
    pose_frames, object_frames = 0, 0
    previous_gray, previous_frame, previous_tracks = None, None, {}
    zones_invalidated = False
    for frame in frames:
        budget.remaining()
        if time.monotonic() - started > config.stage_seconds:
            raise AnalysisError("perception_timeout", "The complete perception timeline exceeded its stage budget")
        if not frame.path.is_file():
            raise AnalysisError("perception_frame_invalid", "A cached source frame is missing")
        image = cv2.imread(str(frame.path))
        if image is None or image.size == 0:
            raise AnalysisError("perception_frame_invalid", "A cached source frame could not be read for perception")
        ratio = min(1., 640. / image.shape[1], 448. / image.shape[0])
        if ratio < 1:
            image = cv2.resize(image, (max(1, int(image.shape[1] * ratio)), max(1, int(image.shape[0] * ratio))))
        gray = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(image, (160, 96)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        scene_change = False
        if previous_gray is not None:
            if frame.timestamp - previous_frame.timestamp > max(1.5 / config.sample_rate, config.track_max_gap):
                gaps.append({"start_sec": previous_frame.timestamp, "end_sec": frame.timestamp,
                             "source_indices": [previous_frame.index, frame.index], "reason": "source_sampling_gap"})
            delta = cv2.absdiff(gray, previous_gray)
            motion = float(np.count_nonzero(delta > 22) / delta.size)
            scene_change = motion > .65
            motions.append({"timestamp_sec": frame.timestamp, "source_index": frame.index,
                            "changed_fraction": motion, "scene_change_suspected": scene_change})
            if scene_change:
                gaps.append({"start_sec": previous_frame.timestamp, "end_sec": frame.timestamp,
                             "source_indices": [previous_frame.index, frame.index], "reason": "scene_change_or_exposure"})
                tracker.reset()
                previous_tracks = {}
                zones_invalidated = True
            if motion >= config.motion_threshold:
                proposals.append(_proposal(previous_frame, frame, "scene_change" if scene_change else "motion_change", info.duration))
        boxes = detector.detect(image)
        tracked, missing = tracker.update(boxes, frame)
        gaps.extend(missing)
        frame_samples.append({"source_index": frame.index, "timestamp_sec": frame.timestamp,
                              "person_detection_available": bool(detector.provenance.get("person_detection_available")),
                              "scene_change_suspected": scene_change})
        current_tracks = {}
        frame_persons = []
        for index, detection in enumerate(tracked):
            if not _valid_box(detection["bbox"]):
                raise AnalysisError("person_output_invalid", "Person detection has an invalid normalized box")
            record = {"id": f"perception-{frame.index}-{index}", "label": "Visible person", "kind": "person",
                      "confidence": detection["confidence"], "bbox": detection["bbox"],
                      "timestamp_sec": frame.timestamp, "source_index": frame.index,
                      "track_id": detection["track_id"], "continuity": detection["continuity"],
                      "backend": detector.name, "role": "unknown",
                      "association_status": detection.get("association_status", "geometric_hypothesis"),
                      "refs": [{"source_index": frame.index, "timestamp_sec": frame.timestamp}]}
            detections.append(record)
            frame_persons.append(record)
            center = [(record["bbox"][0] + record["bbox"][2]) / 2, (record["bbox"][1] + record["bbox"][3]) / 2]
            occupied = [z["name"] for z in scene_map["zones"] if not zones_invalidated
                        and z["bbox"][0] <= center[0] <= z["bbox"][2] and z["bbox"][1] <= center[1] <= z["bbox"][3]]
            current_tracks[record["track_id"]] = {"center": center, "zones": occupied, "frame": frame}
            prior = previous_tracks.get(record["track_id"])
            if prior and not scene_change and record["continuity"] == "adjacent_detection":
                distance = math.dist(center, prior["center"])
                if distance > .08 or occupied != prior["zones"]:
                    proposals.append(_proposal(prior["frame"], frame, "person_zone_transition" if occupied != prior["zones"]
                                               else "person_displacement", info.duration, track_id=record["track_id"]))
            observations.append({"track_id": record["track_id"], "source_index": frame.index,
                                 "timestamp_sec": frame.timestamp, "zones": occupied,
                                 "role": "unknown", "state": "visible_detection"})
        inspect_objects = objects is not None and (frame.timestamp + 1e-6 >= next_objects or scene_change)
        if pose is not None and (frame.timestamp + 1e-6 >= next_pose or scene_change or inspect_objects):
            pose_frames += 1
            for ordinal, estimated in enumerate(pose.estimate(image, frame_persons)):
                poses.append({**estimated, "id": f"pose-{frame.index}-{ordinal}", "source_index": frame.index,
                              "timestamp_sec": frame.timestamp,
                              "refs": [{"source_index": frame.index, "timestamp_sec": frame.timestamp}]})
            next_pose = frame.timestamp + 1. / config.pose_fps
        if inspect_objects:
            object_frames += 1
            object_frame = object_image_refs.get(frame.index)
            if object_frame is None:
                object_frame = cache.get([frame.timestamp], height=min(info.height, 1080))[0]
            object_image = cv2.imread(str(object_frame.path))
            if object_image is None or object_image.size == 0:
                raise AnalysisError("perception_frame_invalid", "An object keyframe could not be read")
            object_samples.append({"source_index": frame.index, "timestamp_sec": frame.timestamp,
                                   "object_detection_available": True, "scene_change_suspected": scene_change,
                                   "input_width": object_image.shape[1], "input_height": object_image.shape[0]})
            if hasattr(objects, "set_timeout"):
                objects.set_timeout(min(config.stage_seconds - (time.monotonic() - started), budget.remaining(2)))
            for ordinal, detected in enumerate(objects.detect(object_image)):
                detections.append({**detected, "id": f"object-{frame.index}-{ordinal}", "source_index": frame.index,
                                   "timestamp_sec": frame.timestamp, "backend": objects.name,
                                   "status": "unconfirmed_object_hypothesis", "confidence_calibrated": False,
                                   "refs": [{"source_index": frame.index, "timestamp_sec": frame.timestamp}]})
            next_objects = frame.timestamp + 1. / config.object_fps
        if not tracked:
            gaps.append({"start_sec": previous_frame.timestamp if previous_frame else frame.timestamp,
                         "end_sec": frame.timestamp, "source_indices": [frame.index], "reason": "person_visibility_unknown"})
        previous_gray, previous_frame, previous_tracks = gray, frame, current_tracks
    # Release optional models before the caller loads or invokes the GPU VLM.
    if pose is not None:
        pose.close()
    if objects is not None:
        objects.close()
    budget.remaining()
    finished = time.monotonic()
    if finished - started > config.stage_seconds:
        raise AnalysisError("perception_timeout", "The complete perception timeline exceeded its stage budget")
    if zones_invalidated:
        scene_map["unknown"] = list(ZONE_NAMES)
        scene_map["invalidated_by_scene_change"] = True
    timestamps = [frame.timestamp for frame in frames]
    max_gap = max([b - a for a, b in zip(timestamps, timestamps[1:])] + [info.duration - timestamps[-1]])
    unknowns = ["Detection misses and occlusion do not establish absence", "Person roles and cash continuity are unknown",
                "Motion and track proposals require focused verification", "Detection scores are uncalibrated"]
    if detector.name == "motion_only":
        unknowns.append("Person detection unavailable: motion-only proposals")
    if scene_map["unknown"]:
        unknowns.append("Scene regions remain unknown or invalidated")
    for gap in gaps:
        gap["refs"] = [{"source_index": index, "timestamp_sec": info.pts[index]}
                       for index in gap["source_indices"]]
    pose_capability.update({"sample_rate": config.pose_fps, "frames_inspected": pose_frames,
                            "missing_keypoints_mean": "not_visible; absence and occlusion are not distinguished"})
    object_capability.update({"sample_rate": config.object_fps, "frames_inspected": object_frames,
                              "temporal_interpolation": False})
    capabilities = {"person": {"status": "enabled" if detector.provenance.get("person_detection_available") else "unavailable",
                                **detector.provenance},
                    "tracking": {"status": "enabled", **tracker.provenance},
                    "pose": pose_capability, "open_vocabulary_objects": object_capability,
                    "scheduling": {"perception_device": "cpu", "gpu_memory_allocated_by_perception": False,
                                   "optional_models_released_before_verification": True}}
    result = {"detections": detections, "poses": poses, "frame_samples": frame_samples, "object_samples": object_samples,
              "stage_budget_sec": config.stage_seconds,
              "capabilities": capabilities, "proposals": merge_proposals(proposals), "visibility_gaps": gaps,
              "track_observations": observations, "motion": motions, "scene": scene_map,
              "coverage": {"sample_rate": config.sample_rate, "frames_analyzed": len(frames),
                           "first_timestamp_sec": timestamps[0], "last_timestamp_sec": timestamps[-1],
                           "ending_included": frames[-1].index == len(info.pts) - 1, "max_gap_sec": max_gap,
                           "complete": True, "unknowns": unknowns}, "backend": detector.name,
              "provenance": {**detector.provenance, "tracker": tracker.provenance},
              "timings": {"model_load_sec": loaded_at - started, "decode_sec": decoded_at - loaded_at,
                          "perception_sec": finished - decoded_at,
                          "total_sec": finished - started}}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "perception.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
