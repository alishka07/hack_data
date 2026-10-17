"""Offline v2 dataset predictions from video pixels and frozen event weights.

No workbook, training manifest, private feature cache or known-video lookup is
read during inference. The returned window is an input-window estimate, not a
supervised event boundary or a legal conclusion.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from worker.event_classifier import predict_event_windows
from worker.event_features import FEATURE_VERSION, contextualize_windows


_FEATURE_DIM = 1024
_CONFIG = {"window_seconds": 8, "stride_seconds": 4, "frames_per_window": 16,
           "context_features": FEATURE_VERSION}


def _validate_model(model, *, experimental=False):
    if (not isinstance(model, Mapping) or type(model.get("format_version")) is not int
            or model["format_version"] != 2 or model.get("model_type") != "r3d18_weak_anchor_event_video"
            or model.get("task") != "dataset_violation_prediction"):
        raise ValueError("Unsupported event model format or task")
    if type(model.get("approved_for_inference")) is not bool:
        raise ValueError("The event model needs an explicit inference approval decision")
    if not model["approved_for_inference"] and not experimental:
        raise ValueError("This event model did not pass its approval gate; --experimental explicitly enables research predictions")
    threshold = model.get("threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The event model threshold must be finite and between zero and one")
    config = model.get("feature_config")
    if not isinstance(config, Mapping) or any(config.get(key) != value for key, value in _CONFIG.items()):
        raise ValueError("The event model feature configuration is incompatible")
    provenance = config.get("encoder")
    if (not isinstance(provenance, Mapping) or provenance.get("encoder") != "r3d_18"
            or provenance.get("feature_dimension") != 512 or provenance.get("pretraining") != "KINETICS400_V1"
            or provenance.get("preprocessing") != "full_frame_letterbox112_kinetics400_mean_std_v1"
            or not isinstance(provenance.get("source_code_version"), str)
            or not provenance["source_code_version"].strip()
            or not isinstance(provenance.get("weights_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", provenance["weights_sha256"]) is None):
        raise ValueError("The event model needs compatible action-encoder provenance")
    try:
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("The event feature configuration is not valid JSON") from exc
    if model.get("feature_fingerprint") != fingerprint:
        raise ValueError("The event feature configuration fingerprint is invalid")
    classifier = model.get("classifier")
    if (not isinstance(classifier, Mapping) or classifier.get("format") != "forte_weak_anchor_event_v1"
            or type(classifier.get("feature_dim")) is not int or classifier["feature_dim"] != _FEATURE_DIM):
        raise ValueError("The event head must accept 1024 contextual action features")
    predict_event_windows(classifier, np.zeros((1, _FEATURE_DIM)))
    return model


def _encoder(assets_dir, threads):
    # The action encoder imports Torch only when actually constructed.
    from worker.action_encoder import ActionEncoder
    return ActionEncoder(assets_dir, threads=threads)


def _verify_assets(assets_dir, expected):
    assets = Path(assets_dir).resolve()
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or any(key not in manifest or manifest[key] != value for key, value in expected.items()):
        raise ValueError("The local action-encoder manifest does not match the trained artifact")
    filename = manifest.get("weights_filename")
    if not isinstance(filename, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.pth", filename) is None:
        raise ValueError("The action encoder weight filename is invalid")
    weights = (assets / filename).resolve()
    if weights.parent != assets:
        raise ValueError("Action encoder weights must remain inside the local assets directory")
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected["weights_sha256"]:
        raise ValueError("The local action encoder weights do not match the trained checksum")


def _window_bounds(windows, count):
    if not isinstance(windows, (list, tuple)) or len(windows) != count:
        raise ValueError("Action features and input-window metadata must have matching lengths")
    result = []
    for window in windows:
        if not isinstance(window, Mapping):
            raise ValueError("Input-window metadata must contain source time bounds")
        values = [window.get(key) for key in ("start_sec", "end_sec", "center_sec")]
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values)
                or not values[0] <= values[2] <= values[1]):
            raise ValueError("Input-window times must be finite, nonnegative and ordered")
        result.append(dict(zip(("start_sec", "end_sec", "center_sec"), map(float, values))))
    return result


class EventDetector:
    """Load an approved model and verified local action encoder once."""

    def __init__(self, model_path, encoder_dir, threads=4, experimental=False):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("Encoder thread count must be an integer between one and 64")
        self.model = _validate_model(json.loads(Path(model_path).read_text(encoding="utf-8")),
                                     experimental=experimental)
        _verify_assets(encoder_dir, self.model["feature_config"]["encoder"])
        self.encoder = _encoder(Path(encoder_dir), threads)

    def predict(self, video):
        video = Path(video)
        if not video.is_file():
            raise FileNotFoundError("The supplied video file does not exist")
        features, windows = self.encoder.clip_features(video)
        contextual = contextualize_windows(features)
        bounds = _window_bounds(windows, len(contextual))
        probabilities = predict_event_windows(self.model["classifier"], contextual)
        index = int(np.argmax(probabilities))
        score, threshold = float(probabilities[index]), float(self.model["threshold"])
        return {"dataset_prediction": "violation" if score >= threshold else "no_violation",
                "score": score, "threshold": threshold,
                "approved_for_inference": self.model["approved_for_inference"],
                "score_calibrated": False, "best_window": bounds[index],
                "timing_scope": "highest_scoring_input_window_not_event_boundaries"}
