"""Offline v3 predictions from physical video observations and fitted weights.

Only the saved model, local pixel-extraction recipe and supplied video are used.
Training labels, feature caches and known-video identity lookups are not read.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

import numpy as np

from worker.linear_classifier import predict_probabilities
from worker.semantic_assets import encoder_recipe, recipe_fingerprint
from worker.semantic_encoder import SemanticEncoder
from worker.semantic_features import (
    SEMANTIC_FEATURE_DIM, SEMANTIC_INTERACTION_DIM, SEMANTIC_VERSION, semantic_vector,
)


def _validate_model(model, *, experimental=False):
    if (not isinstance(model, Mapping) or type(model.get("format_version")) is not int
            or model["format_version"] != 3 or model.get("model_type") != "qwen3vl_semantic_binary_video"
            or model.get("task") != "dataset_violation_prediction"
            or model.get("semantic_version") != SEMANTIC_VERSION
            or model.get("evaluation_scope") != "exploratory_nested_group_cv"
            or model.get("category_heads") != []):
        raise ValueError("Unsupported semantic model format, task or evaluation scope")
    if type(model.get("approved_for_inference")) is not bool:
        raise ValueError("The semantic model needs an explicit inference approval decision")
    if not model["approved_for_inference"] and not experimental:
        raise ValueError("This semantic model did not pass its approval gate; --experimental explicitly enables research predictions")
    threshold = model.get("threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The semantic model threshold must be finite and between zero and one")
    recipe = model.get("semantic_recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("The semantic model needs its complete pixel-extraction recipe")
    fingerprint = recipe_fingerprint(recipe)
    if model.get("recipe_fingerprint") != fingerprint or model.get("encoder_fingerprint") != fingerprint:
        raise ValueError("The semantic model recipe fingerprints are invalid")
    family = model.get("feature_family")
    if family not in ("main", "pairs"):
        raise ValueError("The semantic feature family must be main or pairs")
    dimension = SEMANTIC_INTERACTION_DIM if family == "pairs" else SEMANTIC_FEATURE_DIM
    classifier = model.get("classifier")
    if (not isinstance(classifier, Mapping) or classifier.get("format") != "forte_linear_classifier_v1"
            or type(classifier.get("feature_dim")) is not int or classifier["feature_dim"] != dimension):
        raise ValueError("The semantic head dimension does not match its selected feature family")
    predict_probabilities(classifier, np.empty((0, dimension)))
    return model


class SemanticDetector:
    """Validate once, then reuse a caller-owned runtime for video predictions."""

    def __init__(self, root, model_path, runtime, experimental=False):
        self.root = Path(root).resolve()
        self.model = _validate_model(json.loads(Path(model_path).read_text(encoding="utf-8")),
                                     experimental=experimental)
        actual_recipe = encoder_recipe(self.root)
        if self.model["semantic_recipe"] != actual_recipe:
            raise ValueError("The local semantic encoder recipe or assets do not match the trained artifact")
        if recipe_fingerprint(actual_recipe) != self.model["encoder_fingerprint"]:
            raise ValueError("The local semantic encoder fingerprint does not match the trained artifact")
        self.encoder = SemanticEncoder(self.root, self.root / "data", runtime)

    def predict(self, video):
        video = Path(video)
        if not video.is_file():
            raise FileNotFoundError("The supplied video file does not exist")
        observations, _ = self.encoder.clip_features(video)
        vector = semantic_vector(observations, interactions=self.model["feature_family"] == "pairs")
        score = float(predict_probabilities(self.model["classifier"], vector)[0])
        threshold = float(self.model["threshold"])
        return {"dataset_prediction": "violation" if score >= threshold else "no_violation",
                "score": score, "threshold": threshold,
                "approved_for_inference": self.model["approved_for_inference"],
                "score_calibrated": False}


# Explicit name for callers emphasizing that this adapter owns no runtime lifecycle.
PureSemanticDetector = SemanticDetector
