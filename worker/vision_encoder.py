"""Offline frozen DINOv2 features for supervised video case classifiers.

Only the encoder constructor imports Torch. Training and inference share the
same full-frame preprocessing, temporal sampling, and pooling; labels and file
identities never enter the feature vector.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from worker.video import AnalysisError, Budget, extract_frames, nearest_index, probe_video


FEATURE_DIM = 384
CLIP_FEATURE_DIM = FEATURE_DIM * 3
PREPROCESS_VERSION = "dinov2-vits14-letterbox224-imagenet-mean-std-delta-v1"
_MEAN = np.array([.485, .456, .406], dtype=np.float32)
_STD = np.array([.229, .224, .225], dtype=np.float32)


def preprocess_rgb(image: np.ndarray) -> np.ndarray:
    """Return contiguous float32 CHW: RGB, full-frame letterbox, ImageNet norm.

    Accept uint8 RGB or finite float RGB in [0, 1]. Padding uses the ImageNet
    mean, giving a zero normalized border without cropping visible content.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or not image.shape[0] or not image.shape[1]):
        raise ValueError("Encoder images must be nonempty HWC RGB arrays")
    if image.dtype == np.uint8:
        rgb = image.astype(np.float32) / 255
    elif np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
            raise ValueError("Floating RGB images must have finite values in [0, 1]")
        rgb = image.astype(np.float32, copy=True)
    else:
        raise ValueError("RGB images must use uint8 or floating values in [0, 1]")
    height, width = rgb.shape[:2]
    scale = min(224 / width, 224 / height)
    resized_width = min(224, max(1, round(width * scale)))
    resized_height = min(224, max(1, round(height * scale)))
    resized = cv2.resize(rgb, (resized_width, resized_height),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.empty((224, 224, 3), dtype=np.float32)
    canvas[:] = _MEAN
    left, top = (224 - resized_width) // 2, (224 - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return np.ascontiguousarray(((canvas - _MEAN) / _STD).transpose(2, 0, 1))


def pool_features(frame_features: np.ndarray) -> np.ndarray:
    """Pool chronological Nx384 features as mean, population std, last-first."""
    values = np.asarray(frame_features)
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != FEATURE_DIM
            or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all()):
        raise ValueError("Temporal pooling requires finite nonempty Nx384 features")
    values = values.astype(np.float64)
    pooled = np.concatenate((values.mean(axis=0), values.std(axis=0), values[-1] - values[0]))
    if not np.isfinite(pooled).all() or np.any(np.abs(pooled) > np.finfo(np.float32).max):
        raise ValueError("Pooled visual features are outside the finite float32 range")
    return pooled.astype(np.float32)


def sample_video(path: Path, max_frames: int = 24) -> tuple[list[np.ndarray], list[float], float]:
    """Uniformly sample actual source PTS, including first and last source frames.

    FFmpeg selects decoded source indices; no approximate OpenCV seeking is
    used. Temporary images contain only the selected source frames and are
    removed before return. Short videos may yield fewer than max_frames.
    """
    if type(max_frames) is not int or not 2 <= max_frames <= 512:
        raise ValueError("max_frames must be an integer between 2 and 512")
    budget = Budget(time.time() + 120)
    info = probe_video(Path(path), budget)
    pts = np.asarray(info.pts, dtype=np.float64)
    if (pts.ndim != 1 or not len(pts) or not np.isfinite(pts).all()
            or pts[0] < 0 or np.any(np.diff(pts) <= 0)):
        raise AnalysisError("invalid_timestamps", "Source sampling needs finite increasing frame timestamps")
    count = min(max_frames, len(pts))
    targets = np.linspace(float(pts[0]), float(pts[-1]), count)
    indices = sorted({nearest_index(info.pts, float(target)) for target in targets})
    images, timestamps = [], []
    temporary_root = Path(os.environ.get("FORTE_FEATURE_TEMP_DIR", Path(__file__).resolve().parents[1] / "tmp" / "visual-features"))
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sample-", dir=temporary_root) as temporary:
        refs = extract_frames(info, [info.pts[index] for index in indices], Path(temporary), budget, height=224)
        for ref in refs:
            image = cv2.imread(str(ref.path), cv2.IMREAD_COLOR)
            if image is None:
                raise AnalysisError("frame_extraction_failed", "A sampled source frame could not be read")
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            timestamps.append(float(ref.timestamp))
    return images, timestamps, float(info.duration)


class DinoEncoder:
    """Load a local frozen ViT-S/14 on CPU; never download code or weights."""

    def __init__(self, assets_dir: Path, threads: int = 4):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("Encoder thread count must be an integer between 1 and 64")
        assets = Path(assets_dir)
        source, weights = assets / "source", assets / "dinov2_vits14_pretrain.pth"
        if not (source / "hubconf.py").is_file() or not weights.is_file():
            raise FileNotFoundError("The pinned local DINOv2 source and ViT-S/14 weights are required")
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ImportError("Frozen visual encoding requires the separate Torch training environment") from exc
        torch.set_num_threads(threads)
        model = torch.hub.load(str(source), "dinov2_vits14", source="local", pretrained=False)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        model.eval()
        model.requires_grad_(False)
        self._torch, self._model = torch, model

    def encode_rgb(self, images) -> np.ndarray:
        """Return float32 Nx384 features in input order, using CPU batches of 4."""
        images = [images] if isinstance(images, np.ndarray) and images.ndim == 3 else list(images)
        if not images:
            return np.empty((0, FEATURE_DIM), dtype=np.float32)
        result = []
        with self._torch.inference_mode():
            for start in range(0, len(images), 4):
                batch = np.stack([preprocess_rgb(image) for image in images[start:start + 4]])
                output = self._model(self._torch.from_numpy(batch)).detach().cpu().numpy()
                if (output.shape != (len(batch), FEATURE_DIM) or not np.isfinite(output).all()
                        or np.any(np.abs(output) > np.finfo(np.float32).max)):
                    raise ValueError("The local visual encoder returned invalid Nx384 features")
                result.append(output.astype(np.float32))
        return np.concatenate(result, axis=0)

    def clip_features(self, path: Path, max_frames: int = 24) -> tuple[np.ndarray, list[float], np.ndarray]:
        images, timestamps, _ = sample_video(path, max_frames)
        frame_features = self.encode_rgb(images)
        return pool_features(frame_features), timestamps, frame_features
