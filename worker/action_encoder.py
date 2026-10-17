"""Frozen local R3D-18 motion features with label-independent source sampling.

This encoder is a feature extractor, not a banking violation classifier. Its
full-frame letterboxing and approximately 2 fps inputs differ from the original
Kinetics evaluation crop and frame rate; downstream held-out evaluation matters.
Torch and torchvision are imported only when constructing the CPU encoder.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import cv2
import numpy as np

from worker.video import AnalysisError, Budget, extract_frames, nearest_index, probe_video


FEATURE_DIM = 512
CLIP_FRAMES = 16
WINDOW_SEC = 8.
STRIDE_SEC = 4.
MAX_DURATION_SEC = 600.
PREPROCESS_VERSION = "full_frame_letterbox112_kinetics400_mean_std_v1"
WEIGHTS_FILENAME = "r3d_18-b3b3357e.pth"
_MEAN = np.array([.43216, .394666, .37645], dtype=np.float32)
_STD = np.array([.22803, .22145, .216989], dtype=np.float32)


def preprocess_rgb(image: np.ndarray) -> np.ndarray:
    """Full-frame letterbox112, mean-color padding, official Kinetics norm."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or not image.shape[0] or not image.shape[1]):
        raise ValueError("Action inputs must be nonempty HWC RGB arrays")
    if image.dtype == np.uint8:
        rgb = image.astype(np.float32) / 255
    elif np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
            raise ValueError("Floating RGB inputs must have finite values in [0, 1]")
        rgb = image.astype(np.float32, copy=True)
    else:
        raise ValueError("Action inputs require uint8 RGB or floating RGB in [0, 1]")
    height, width = rgb.shape[:2]
    scale = min(112 / height, 112 / width)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.empty((112, 112, 3), dtype=np.float32)
    canvas[:] = _MEAN
    top, left = (112 - resized_height) // 2, (112 - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return np.ascontiguousarray(((canvas - _MEAN) / _STD).transpose(2, 0, 1))


def window_plan(duration: float) -> list[dict]:
    """Plan fixed 8s/stride4s windows, with one shortened final window."""
    if (type(duration) not in (int, float) or not math.isfinite(duration)
            or not 0 < duration <= MAX_DURATION_SEC):
        raise ValueError("Action video duration must be positive and at most 600 seconds")
    windows, start = [], 0.
    while start < duration:
        end = min(start + WINDOW_SEC, float(duration))
        windows.append({"start_sec": start, "end_sec": end, "center_sec": (start + end) / 2})
        if end >= duration:
            break
        start += STRIDE_SEC
    return windows


def _source_timestamps(timestamps, duration):
    try:
        raw = np.asarray(timestamps)
        if raw.dtype.kind not in "iuf":
            raise ValueError
        values = np.asarray(raw, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Action sampling requires finite increasing source timestamps") from exc
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or values[0] < 0 or values[-1] > duration + 1e-3 or np.any(np.diff(values) <= 0)):
        raise ValueError("Action sampling requires finite increasing source timestamps within the clip")
    return tuple(float(value) for value in values)


def window_metadata(timestamps, duration: float) -> list[dict]:
    """Bind every uniform target to one supplied source PTS; repeats are explicit."""
    windows = window_plan(duration)
    pts = _source_timestamps(timestamps, duration)
    for window in windows:
        targets = np.linspace(window["start_sec"], window["end_sec"], CLIP_FRAMES)
        indices = [nearest_index(pts, float(target)) for target in targets]
        window["input_indices"] = indices
        window["sample_timestamps_sec"] = [pts[index] for index in indices]
    return windows


def sample_video_windows(path: Path) -> tuple[list[np.ndarray], list[float], float, list[int]]:
    """One FFmpeg extraction for the union of all window source-frame indices."""
    budget = Budget(time.time() + 120)
    info = probe_video(Path(path), budget)
    windows = window_metadata(info.pts, info.duration)
    indices = sorted({index for window in windows for index in window["input_indices"]})
    temporary_root = Path(os.environ.get("FORTE_FEATURE_TEMP_DIR", Path(__file__).resolve().parents[1] / "tmp" / "action-features"))
    temporary_root.mkdir(parents=True, exist_ok=True)
    images, timestamps = [], []
    with tempfile.TemporaryDirectory(prefix="sample-", dir=temporary_root) as temporary:
        refs = extract_frames(info, [info.pts[index] for index in indices], Path(temporary), budget, height=112)
        for ref in refs:
            image = cv2.imread(str(ref.path), cv2.IMREAD_COLOR)
            if image is None:
                raise AnalysisError("frame_extraction_failed", "A sampled action source frame could not be read")
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            timestamps.append(float(ref.timestamp))
    return images, timestamps, float(info.duration), indices


class ActionEncoder:
    """Offline, checksum-verified Kinetics R3D-18 pre-classifier CPU features."""

    def __init__(self, assets_dir: Path, threads: int = 4):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("Action encoder thread count must be an integer between 1 and 64")
        assets = Path(assets_dir)
        manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("encoder") != "r3d_18"
                or manifest.get("pretraining") != "KINETICS400_V1"
                or manifest.get("feature_dimension") != FEATURE_DIM
                or manifest.get("preprocessing") != PREPROCESS_VERSION
                or not isinstance(manifest.get("weights_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", manifest["weights_sha256"]) is None):
            raise ValueError("The local action encoder provenance is incompatible")
        weights = assets / WEIGHTS_FILENAME
        digest = hashlib.sha256()
        with weights.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest["weights_sha256"]:
            raise ValueError("The local action encoder weights failed their checksum")
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        if manifest.get("source_code_version") != "torchvision==" + torchvision.__version__:
            raise ValueError("The installed torchvision version does not match the pinned encoder")
        torch.set_num_threads(threads)
        model = torchvision.models.video.r3d_18(weights=None)
        model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True), strict=True)
        # Load all original weights strictly before replacing the Kinetics head.
        model.fc = torch.nn.Identity()
        model.to("cpu")
        model.eval()
        model.requires_grad_(False)
        self._torch, self._model, self.provenance = torch, model, manifest

    def encode_clips(self, clips) -> np.ndarray:
        """Encode ordered 16-frame RGB clips as float32 Nx512, one CPU clip/batch."""
        result = []
        with self._torch.inference_mode():
            for clip in clips:
                clip = list(clip)
                if len(clip) != CLIP_FRAMES:
                    raise ValueError("The action encoder requires exactly 16 ordered frames per clip")
                array = np.stack([preprocess_rgb(image) for image in clip], axis=1)[None]
                output = self._model(self._torch.from_numpy(array)).detach().cpu().numpy()
                if (output.shape != (1, FEATURE_DIM) or not np.isfinite(output).all()
                        or np.any(np.abs(output) > np.finfo(np.float32).max)):
                    raise ValueError("The action encoder returned invalid 512-dimensional features")
                result.append(output[0].astype(np.float32))
        return np.stack(result) if result else np.empty((0, FEATURE_DIM), dtype=np.float32)

    def encode_windows(self, images, timestamps, duration: float) -> tuple[np.ndarray, list[dict]]:
        """Create the same deterministic windows from already sampled RGB frames."""
        images = list(images)
        windows = window_metadata(timestamps, duration)
        if len(images) != len(timestamps):
            raise ValueError("Every supplied action source frame needs its source timestamp")
        clips = ([images[index] for index in window["input_indices"]] for window in windows)
        return self.encode_clips(clips), windows

    def clip_features(self, video: Path) -> tuple[np.ndarray, list[dict]]:
        images, timestamps, duration, source_indices = sample_video_windows(video)
        features, windows = self.encode_windows(images, timestamps, duration)
        for window in windows:
            window["source_indices"] = [source_indices[index] for index in window["input_indices"]]
        return features, windows
