"""Offline binary dataset predictions from a frozen encoder and fitted weights.

Inference uses the model artifact and video pixels. Training workbooks, split
manifests, cached labels, and video-identity lookups are never opened here.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from worker.linear_classifier import predict_probabilities
from worker.vision_encoder import CLIP_FEATURE_DIM, DinoEncoder


def _validate_model(model, *, experimental=False):
    if (not isinstance(model, Mapping) or type(model.get("format_version")) is not int
            or model["format_version"] != 1 or model.get("model_type") != "dinov2_linear_binary_video"
            or model.get("task") != "dataset_violation_prediction"):
        raise ValueError("Unsupported trained case model format or task")
    approved = model.get("approved_for_inference")
    if type(approved) is not bool:
        raise ValueError("The model needs an explicit inference approval decision")
    if not approved and not experimental:
        raise ValueError("This model did not pass its inference approval gate; use --experimental only for an explicit research prediction")
    threshold = model.get("threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The model decision threshold must be finite and between zero and one")
    config = model.get("feature_config")
    if (not isinstance(config, Mapping) or type(config.get("max_frames")) is not int
            or not 2 <= config["max_frames"] <= 512
            or config.get("preprocessing") != "full_frame_letterbox_224_imagenet_v1"
            or config.get("pooling") != "mean_std_last_minus_first_v1"):
        raise ValueError("The model feature configuration does not match this encoder pipeline")
    provenance = config.get("encoder")
    if (not isinstance(provenance, Mapping) or provenance.get("encoder") != "dinov2_vits14"
            or not isinstance(provenance.get("weights_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", provenance["weights_sha256"]) is None):
        raise ValueError("The model needs the expected local encoder weight checksum")
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if model.get("feature_fingerprint") != fingerprint:
        raise ValueError("The model feature configuration fingerprint is invalid")
    classifier = model.get("classifier")
    if (not isinstance(classifier, Mapping) or classifier.get("format") != "forte_linear_classifier_v1"
            or classifier.get("feature_dim") != CLIP_FEATURE_DIM):
        raise ValueError("The model classifier must accept the expected 1152 visual features")
    # Validate fitted coefficients and normalization before loading an encoder.
    predict_probabilities(classifier, np.empty((0, CLIP_FEATURE_DIM)))
    return model


def load_model_artifact(path: Path, *, experimental=False) -> dict:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_model(model, experimental=experimental)


def predict_features(model: Mapping, features: np.ndarray, *, experimental=False) -> dict:
    """Classify one pooled feature vector; the score is explicitly uncalibrated."""
    _validate_model(model, experimental=experimental)
    vector = np.asarray(features)
    if vector.shape != (CLIP_FEATURE_DIM,):
        raise ValueError("A dataset prediction requires one pooled 1152-feature vector")
    score = float(predict_probabilities(model["classifier"], vector)[0])
    threshold = float(model["threshold"])
    return {"dataset_prediction": "violation" if score >= threshold else "no_violation",
            "score": score, "threshold": threshold,
            "approved_for_inference": model["approved_for_inference"], "score_calibrated": False}


def predict_video(video: Path, model_path: Path, encoder_dir: Path, *, threads=4, experimental=False) -> dict:
    """Load only the artifact, local encoder assets, and the supplied video."""
    model = load_model_artifact(model_path, experimental=experimental)
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ValueError("Encoder thread count must be an integer between 1 and 64")
    if not Path(video).is_file():
        raise FileNotFoundError("The supplied video file does not exist")
    weights = Path(encoder_dir) / "dinov2_vits14_pretrain.pth"
    digest = hashlib.sha256()
    with weights.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != model["feature_config"]["encoder"]["weights_sha256"]:
        raise ValueError("The local encoder weights do not match the trained model checksum")
    encoder = DinoEncoder(Path(encoder_dir), threads=threads)
    vector, _, _ = encoder.clip_features(Path(video), max_frames=model["feature_config"]["max_frames"])
    return predict_features(model, vector, experimental=experimental)
