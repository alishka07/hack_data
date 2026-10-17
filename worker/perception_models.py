"""Optional local perception adapters. Network access belongs to setup, never inference.

Only pixels, anonymous boxes and a fixed application vocabulary enter these
models. Filenames, rule labels, dataset annotations and verdicts are not inputs.
All backends use CPU so their weights cannot contend with the GPU VLM.
"""
from __future__ import annotations

import gc
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from worker.video import AnalysisError

ROOT = Path(__file__).resolve().parents[1]
KEYPOINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
OBJECT_QUERIES = (("cash", "a photo of banknotes or paper money"), ("bag", "a photo of a bag"),
                  ("phone", "a photo of a mobile phone"), ("drawer", "a photo of a drawer"),
                  ("safe", "a photo of a metal safe"), ("counter", "a photo of a service counter"),
                  ("window", "a photo of a teller window"))


def dependency_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _manifest(path: Path, expected_format: str):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["format"] != expected_format or manifest["license"] != "Apache-2.0":
            raise ValueError("Unsupported manifest")
        if not manifest["source_url"].startswith("https://"):
            raise ValueError("Missing source")
        return manifest
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AnalysisError("perception_asset_invalid", "The local optional model manifest is missing or invalid") from exc


def _verify_file(path: Path, expected):
    try:
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid SHA256")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Hash mismatch")
    except (OSError, ValueError) as exc:
        raise AnalysisError("perception_asset_invalid", "A local optional perception asset failed its SHA256 check") from exc


class ByteTrackAdapter:
    """Actual two-stage Kalman/Hungarian ByteTrack, using source presentation time.

    Reused IDs are geometric hypotheses. A reappearance after a missed sample
    has continuity=after_unknown_gap; no predicted boxes are emitted as evidence.
    """
    name = "bytetrack"

    def __init__(self, fps=5., max_gap=.8, threshold=.35, low_threshold=.1, iou=.25):
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/perception-mpl"))
        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            import supervision as sv
            from trackers import ByteTrackTracker
        except (ImportError, OSError) as exc:
            raise AnalysisError("tracking_backend_unavailable", "Install requirements.perception.txt for local ByteTrack") from exc
        self.sv = sv
        self.high, self.low = threshold, low_threshold
        self.max_gap, self.segment = max_gap, 0
        # This library specifies lost buffer in 30fps units and supports PTS.
        self.tracker = ByteTrackTracker(frame_rate=fps, lost_track_buffer=max(1, math.ceil(max_gap * 30)),
                                       minimum_consecutive_frames=1, minimum_iou_threshold=iou,
                                       track_activation_threshold=threshold, high_conf_det_threshold=threshold)
        self.previous, self.last_seen = {}, {}
        self.last_frame = None
        self.provenance = {"backend": self.name, "implementation": "trackers.ByteTrackTracker",
                           "version": importlib.metadata.version("trackers"), "license": "Apache-2.0",
                           "source": "https://github.com/roboflow/trackers",
                           "association": "two-stage confidence, Kalman prediction, Hungarian assignment",
                           "reidentification": False, "clock": "source_presentation_timestamp",
                           "high_threshold": threshold, "low_threshold": low_threshold,
                           "max_lost_sec": max_gap, "predicted_boxes_are_evidence": False}

    def reset(self):
        self.tracker.reset()
        self.previous, self.last_seen, self.last_frame = {}, {}, None
        self.segment += 1

    def update(self, boxes, frame):
        if self.last_frame is not None and frame.timestamp - self.last_frame.timestamp > self.max_gap:
            self.reset()
        selected = [box for box in boxes if box["confidence"] >= self.low]
        detections = self.sv.Detections(
            xyxy=np.asarray([box["bbox"] for box in selected], np.float32).reshape(-1, 4) * 1000,
            confidence=np.asarray([box["confidence"] for box in selected], np.float32),
            class_id=np.zeros(len(selected), dtype=int), data={"input_index": np.arange(len(selected))})
        tracked = self.tracker.update(detections, timestamp=frame.timestamp)
        output, current, resumed_gaps = [], {}, []
        for original_index, identity in zip(tracked.data.get("input_index", []), tracked.tracker_id):
            detection = selected[int(original_index)]
            if identity < 0 and detection["confidence"] < self.high:
                continue  # Low confidence boxes only survive when an existing track supports them.
            key = f"bt-{self.segment}-{int(identity)}" if identity >= 0 else f"unassociated-{frame.index}-{int(original_index)}"
            continuity = "adjacent_detection" if key in self.previous else "after_unknown_gap" if key in self.last_seen else "new"
            if continuity == "after_unknown_gap":
                last = self.last_seen[key]
                resumed_gaps.append({"start_sec": last.timestamp, "end_sec": frame.timestamp,
                                     "source_indices": [last.index, frame.index], "reason": "track_detection_gap", "track_id": key})
            row = {**detection, "track_id": key, "continuity": continuity,
                   "association_status": "geometric_hypothesis" if identity >= 0 else "unassociated_detection"}
            output.append(row)
            current[key] = frame
        gaps = [{"start_sec": last.timestamp, "end_sec": frame.timestamp,
                 "source_indices": [last.index, frame.index], "reason": "track_detection_gap", "track_id": key}
                for key, last in self.previous.items() if key not in current]
        gaps.extend(resumed_gaps)
        self.previous = current
        self.last_seen.update(current)
        self.last_frame = frame
        return output, gaps


def pose_affine(image, box, input_size=(192, 256)):
    """Top-down padded bbox transform from the OpenMMLab deployment contract."""
    height, width = image.shape[:2]
    bbox = np.asarray(box, np.float32) * [width, height, width, height]
    center = (bbox[:2] + bbox[2:]) / 2
    scale = (bbox[2:] - bbox[:2]) * 1.25
    w, h = input_size
    scale = np.asarray([max(scale[0], scale[1] * w / h), max(scale[1], scale[0] * h / w)], np.float32)
    matrix = np.asarray([[w / scale[0], 0, w * (.5 - center[0] / scale[0])],
                         [0, h / scale[1], h * (.5 - center[1] / scale[1])]], np.float32)
    crop = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    # Official pure-ONNX example consumes BGR and this mean/std; no channel swap.
    crop = (crop.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32), center, scale


def decode_simcc(outputs, input_size, center, scale, source_size, threshold=.3, split_ratio=2.):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise AnalysisError("pose_output_invalid", "RTMPose must return two raw SimCC tensors")
    x, y = map(np.asarray, outputs)
    w, h = input_size
    if (x.shape != (1, 17, int(w * split_ratio)) or y.shape != (1, 17, int(h * split_ratio))
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise AnalysisError("pose_output_invalid", "RTMPose output does not match the COCO-17 raw SimCC contract")
    points = np.stack([x[0].argmax(axis=1), y[0].argmax(axis=1)], axis=1) / split_ratio
    points = (points / [w, h] * scale + center - np.asarray(scale) / 2) / np.asarray(source_size)
    scores = np.minimum(x[0].max(axis=1), y[0].max(axis=1))
    result = []
    for index, (point, score) in enumerate(zip(points, scores)):
        visible = score >= threshold and bool(np.all(point >= 0) and np.all(point <= 1))
        result.append({"index": index, "name": KEYPOINT_NAMES[index],
                       "x": float(point[0]) if visible else None, "y": float(point[1]) if visible else None,
                       "confidence": float(score), "status": "supported" if visible else "not_visible"})
    return result


class RTMPoseONNX:
    name = "rtmpose_onnx_cpu"

    def __init__(self, model, manifest_path=None, threshold=.3):
        model = Path(model)
        manifest = _manifest(Path(manifest_path) if manifest_path else model.with_suffix(".json"), "rtmpose-simcc-coco17-v1")
        _verify_file(model, manifest.get("sha256"))
        self.size = tuple(manifest.get("input_size", ()))
        if self.size != (192, 256) or manifest.get("color_order") != "BGR" or manifest.get("simcc_split_ratio") != 2.:
            raise AnalysisError("perception_asset_invalid", "Unsupported RTMPose preprocessing contract")
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        except (ImportError, OSError, RuntimeError) as exc:
            raise AnalysisError("pose_backend_unavailable", "Install the CPU ONNX Runtime and a compatible local RTMPose model") from exc
        shape = self.session.get_inputs()[0].shape
        if (len(shape) != 4 or shape[1:] != [3, 256, 192]
                or not (shape[0] == 1 or shape[0] is None or isinstance(shape[0], str))):
            raise AnalysisError("perception_asset_invalid", "RTMPose input shape is incompatible")
        self.threshold = threshold
        self.provenance = {"backend": self.name, "model_sha256": manifest["sha256"], "source": manifest["source_url"],
                           "license": manifest["license"], "device": "cpu", "model_download_during_inference": False,
                           "keypoint_schema": "coco17", "score_semantics": "uncalibrated minimum SimCC peak; not probability"}

    def estimate(self, image, persons):
        height, width = image.shape[:2]
        result = []
        for person in persons:
            tensor, center, scale = pose_affine(image, person["bbox"], self.size)
            try:
                outputs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
            except Exception as exc:
                raise AnalysisError("pose_inference_failed", "The local CPU RTMPose model failed") from exc
            keypoints = decode_simcc(outputs, self.size, center, scale, (width, height), self.threshold)
            result.append({"person_detection_id": person["id"], "track_id": person["track_id"], "bbox": person["bbox"],
                           "keypoints": keypoints, "backend": self.name, "confidence_calibrated": False})
        return result

    def close(self):
        self.session = None


def normalize_owlv2_result(result, source_size, threshold=.15):
    """Only reviewed fixed query IDs can become object classes."""
    height, width = source_size
    def array(value):
        return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
    boxes, scores, labels = (array(result[key]) for key in ("boxes", "scores", "labels"))
    if (boxes.ndim != 2 or boxes.shape[1:] != (4,) or scores.shape != (len(boxes),) or labels.shape != (len(boxes),)
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or not np.isfinite(labels).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise AnalysisError("object_output_invalid", "OWLv2 returned invalid detection tensors")
    output = []
    for box, score, label in zip(boxes, scores, labels):
        if int(label) != label or not 0 <= int(label) < len(OBJECT_QUERIES):
            raise AnalysisError("object_output_invalid", "OWLv2 returned an unknown fixed query ID")
        bbox = np.clip(box / [width, height, width, height], 0., 1.).tolist()
        if score < threshold or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            continue
        kind, query = OBJECT_QUERIES[int(label)]
        # Greedy per-class NMS avoids duplicate token boxes without a TorchVision op.
        from worker.perception import _iou
        record = {"bbox": bbox, "confidence": float(score), "kind": kind, "label": kind, "query": query}
        output.append(record)
    kept = []
    for row in sorted(output, key=lambda r: -r["confidence"]):
        if not any(row["kind"] == old["kind"] and _iou(row["bbox"], old["bbox"]) > .45 for old in kept):
            kept.append(row)
    return kept


class OWLv2Local:
    name = "owlv2_cpu"

    def __init__(self, directory, threshold=.15):
        directory = Path(directory)
        try:
            self.input_size = int(os.getenv("FORTE_OBJECT_INPUT_SIZE", "480"))
            self.threads = int(os.getenv("FORTE_OBJECT_THREADS", "4"))
        except ValueError as exc:
            raise AnalysisError("perception_config_invalid", "Object input size must be an integer") from exc
        if not 192 <= self.input_size <= 1280 or self.input_size % 16:
            raise AnalysisError("perception_config_invalid", "Object input size must be a multiple of 16 from 192 to 1280")
        if not 1 <= self.threads <= 16:
            raise AnalysisError("perception_config_invalid", "Object CPU threads must be from 1 to 16")
        manifest = _manifest(directory / "manifest.json", "owlv2-local-v1")
        if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
            raise AnalysisError("perception_asset_invalid", "OWLv2 manifest has no verified files")
        for filename, digest in manifest["files"].items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise AnalysisError("perception_asset_invalid", "OWLv2 manifest has an invalid file path")
            _verify_file(directory / relative, digest)
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.torch = torch
            torch.set_num_threads(min(self.threads, os.cpu_count() or 1))
            if os.getenv("FORTE_OBJECT_ISOLATED_WORKER") == "1":
                torch.set_num_interop_threads(1)
            # Passing an existing directory + local_files_only prevents a remote fallback.
            self.processor = Owlv2Processor.from_pretrained(str(directory.resolve()), local_files_only=True, use_fast=False)
            self.model = Owlv2ForObjectDetection.from_pretrained(str(directory.resolve()), local_files_only=True,
                                                               use_safetensors=True).to("cpu").eval()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise AnalysisError("object_backend_unavailable", "OWLv2 requires local weights, CPU Torch and Transformers") from exc
        self.threshold = threshold
        self.provenance = {"backend": self.name, "source": manifest["source_url"], "revision": manifest.get("revision"),
                           "license": manifest["license"], "device": "cpu", "model_download_during_inference": False,
                           "queries": [q[0] for q in OBJECT_QUERIES], "confidence_calibrated": False,
                           "weights_verified_sha256": True, "small_object_recall_validated": False,
                           "input_size": [self.input_size, self.input_size], "native_input_size": [960, 960],
                           "position_encoding_interpolated": self.input_size != 960,
                           "cpu_threads": min(self.threads, os.cpu_count() or 1),
                           "resolution_tradeoff": "Lower CPU input size reduces latency and may reduce small-object recall"}

    def detect(self, image):
        from PIL import Image
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=[[query for _, query in OBJECT_QUERIES]], images=rgb, return_tensors="pt",
                                size={"height": self.input_size, "width": self.input_size})
        with self.torch.inference_mode():
            outputs = self.model(**inputs, interpolate_pos_encoding=self.input_size != 960)
        result = self.processor.post_process_object_detection(outputs=outputs, threshold=self.threshold,
                    target_sizes=self.torch.tensor([image.shape[:2]]))[0]
        return normalize_owlv2_result(result, image.shape[:2], self.threshold)

    def close(self):
        self.model, self.processor = None, None
        gc.collect()


class OWLv2Subprocess:
    """Persistent CPU object worker in an isolated Torch environment.

    Pixels cross an anonymous stdin protocol. No video path, filename, training
    label, rule text or GPU tensor crosses this boundary. Close terminates the
    process and frees model memory before VLM verification begins.
    """
    name = "owlv2_cpu_subprocess"

    def __init__(self, directory, python, threshold=.15, timeout=None):
        python = Path(python)
        if not python.is_file():
            raise AnalysisError("object_backend_unavailable", "The configured object worker Python executable is missing")
        self.messages = queue.Queue()
        self.deadline = time.monotonic() + timeout if timeout is not None else None
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                           FORTE_OBJECT_ISOLATED_WORKER="1")
        self.process = subprocess.Popen([str(python.resolve()), "-u", "-m", "worker.perception_models",
                                         "--owl-worker", str(Path(directory).resolve()), str(threshold)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        cwd=str(ROOT), env=environment, text=True, encoding="utf-8",
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        def receive():
            try:
                for line in self.process.stdout:
                    self.messages.put(line)
            except (OSError, ValueError):
                pass
            finally:
                self.messages.put(None)
        threading.Thread(target=receive, daemon=True).start()
        try:
            ready = self._receive(120)
            if ready.get("status") != "ready":
                raise AnalysisError(ready.get("code", "object_backend_unavailable"), "The local CPU object worker could not load its model")
            self.provenance = {**ready["provenance"], "backend": self.name,
                               "isolated_runtime": True, "pixel_only_ipc": True}
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout):
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
        if timeout <= 0:
            raise AnalysisError("object_inference_timeout", "The local object worker has no remaining stage budget")
        try:
            message = self.messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise AnalysisError("object_inference_timeout", "The local CPU object worker exceeded its response budget") from exc
        if message is None:
            raise AnalysisError("object_backend_unavailable", "The local CPU object worker exited before replying")
        try:
            return json.loads(message)
        except (ValueError, TypeError) as exc:
            raise AnalysisError("object_output_invalid", "The local CPU object worker returned invalid JSON") from exc

    def set_timeout(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise AnalysisError("object_inference_timeout", "The local object worker has no remaining stage budget")
        deadline = time.monotonic() + seconds
        self.deadline = deadline if self.deadline is None else min(self.deadline, deadline)

    def detect(self, image):
        success, encoded = cv2.imencode(".png", image)
        if not success or encoded.nbytes > 20 * 1024**2:
            raise AnalysisError("object_input_invalid", "Cannot encode the bounded object analysis image")
        try:
            self.process.stdin.write(json.dumps({"png_base64": base64.b64encode(encoded).decode("ascii")}) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise AnalysisError("object_backend_unavailable", "The local CPU object worker pipe closed") from exc
        result = self._receive(90)
        if result.get("status") != "ok" or not isinstance(result.get("detections"), list):
            raise AnalysisError("object_inference_failed", "The local CPU object worker failed to inspect pixels")
        return result["detections"]

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
            self.process = None


def object_worker_python():
    explicit = os.getenv("FORTE_OBJECT_PYTHON")
    if explicit:
        return Path(explicit)
    candidate = ROOT / (".venv-train/Scripts/python.exe" if os.name == "nt" else ".venv-train/bin/python")
    return candidate if candidate.is_file() else None


def optional_backend(kind, setting, path, threshold, timeout=None):
    """Auto reports absence explicitly; explicit configuration fails loudly."""
    if setting not in ("auto", "disabled", "rtmpose" if kind == "pose" else "owlv2"):
        raise AnalysisError("perception_config_invalid", f"Unknown {kind} backend")
    base = {"status": "unavailable", "enabled": False, "model_download_during_inference": False,
            "reason": "No local model installed; run scripts/setup_architecture_models.py"}
    if setting == "disabled":
        return None, {**base, "status": "disabled", "reason": "Disabled by configuration"}
    exists = Path(path).is_file() if kind == "pose" else (Path(path) / "manifest.json").is_file()
    if not exists and setting == "auto":
        return None, base
    dependencies = ("onnxruntime",) if kind == "pose" else ("torch", "transformers")
    isolated = kind == "objects" and object_worker_python() is not None
    missing_deps = any(not dependency_available(name) for name in dependencies)
    if setting == "auto" and missing_deps and not isolated:
        return None, {**base, "reason": "Local model exists but optional runtime dependencies are not installed"}
    try:
        if kind == "pose":
            backend = RTMPoseONNX(path, threshold=threshold)
        else:
            # A subprocess also bounds inference when Torch happens to be
            # installed in the API environment. The child invokes OWLv2Local
            # directly, so this cannot recursively spawn workers.
            backend = OWLv2Subprocess(path, object_worker_python() or Path(sys.executable), threshold, timeout=timeout)
    except AnalysisError as exc:
        if setting == "auto" and exc.code in ("pose_backend_unavailable", "object_backend_unavailable"):
            return None, {**base, "reason": "Local optional runtime is unavailable; install the documented CPU dependencies"}
        raise
    return backend, {"status": "enabled", "enabled": True, **backend.provenance}


def _owl_worker(directory, threshold):
    try:
        backend = OWLv2Local(directory, threshold)
        print(json.dumps({"status": "ready", "provenance": backend.provenance}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            encoded = request.get("png_base64")
            if not isinstance(encoded, str) or len(encoded) > 28 * 1024**2 or set(request) != {"png_base64"}:
                raise ValueError("Invalid pixel-only request")
            image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[0] * image.shape[1] > 8_000_000:
                raise ValueError("Invalid bounded image")
            print(json.dumps({"status": "ok", "detections": backend.detect(image)}, allow_nan=False), flush=True)
        backend.close()
    except Exception as exc:
        # Failure is explicit without echoing labels, model paths or raw pixels.
        print(json.dumps({"status": "error", "code": getattr(exc, "code", "object_backend_unavailable")}), flush=True)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--owl-worker":
        raise SystemExit("This module only accepts an explicit local object worker command")
    _owl_worker(Path(sys.argv[2]), float(sys.argv[3]))
