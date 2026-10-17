"""Deterministic binary classifiers for frozen numeric video features.

The returned artifact stores preprocessing, coefficients and aggregate optimizer
diagnostics. Prediction never needs training labels or source-video hashes.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


def _matrix(values, *, allow_vector=False, allow_empty=False):
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "biuf":
            raise ValueError
        matrix = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Features must be a finite numeric matrix") from exc
    if allow_vector and matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if (matrix.ndim != 2 or matrix.shape[1] == 0
            or not allow_empty and matrix.shape[0] == 0
            or not np.isfinite(matrix).all()):
        raise ValueError("Features must be a nonempty finite numeric matrix")
    return matrix


def _targets(values, expected_count=None):
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "biuf":
            raise ValueError
        targets = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Targets must be a one-dimensional binary array") from exc
    if (targets.ndim != 1 or expected_count is not None and len(targets) != expected_count
            or not np.isfinite(targets).all()
            or not np.isin(targets, [0, 1]).all()):
        raise ValueError("Targets must be a one-dimensional binary array matching the samples")
    return targets


def _sigmoid(logits):
    """Evaluate sigmoid without overflowing exp for either sign."""
    result = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def _standardize(features, mean, scale, constant):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (features - mean) / scale
        if not np.isfinite(normalized[:, ~constant]).all():
            alternative = features / scale - mean / scale
            normalized = np.where(np.isfinite(normalized), normalized, alternative)
    normalized[:, constant] = 0
    if not np.isfinite(normalized).all():
        raise ValueError("Features cannot be standardized to finite values")
    return normalized


def fit_linear_classifier(X, y, regularization=1.0, max_iter=1000):
    """Fit balanced logistic regression using train-only standardization.

    Objective: balanced mean logistic loss plus regularization times squared
    weight norm divided by twice the training sample count. The intercept is
    unpenalized. Full-batch gradient descent uses an Armijo line search and
    allocates no feature-by-feature Hessian. Both binary classes are required.
    Inputs are copied and never modified.
    """
    features = _matrix(X)
    targets = _targets(y, len(features))
    if len(np.unique(targets)) != 2:
        raise ValueError("Training requires both binary classes")
    if (type(regularization) not in (int, float) or not math.isfinite(regularization)
            or regularization < 0):
        raise ValueError("Regularization must be finite and nonnegative")
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    # Scaling before computing moments avoids overflowing a finite column's sum
    # or squared values. Only these training rows determine the transform.
    magnitude = np.max(np.abs(features), axis=0)
    safe_magnitude = np.where(magnitude > 0, magnitude, 1)
    relative = features / safe_magnitude
    relative_mean = relative.mean(axis=0)
    relative_scale = np.sqrt(np.mean((relative - relative_mean) ** 2, axis=0))
    mean = relative_mean * safe_magnitude
    scale = relative_scale * safe_magnitude
    constant = (relative_scale == 0) | (scale == 0)
    scale[constant] = 1
    normalized = _standardize(features, mean, scale, constant)

    sample_count, feature_dim = features.shape
    positive_count = int(targets.sum())
    negative_count = sample_count - positive_count
    sample_weights = np.where(targets == 1, .5 / positive_count, .5 / negative_count)
    penalty = float(regularization) / sample_count
    coefficients = np.zeros(feature_dim + 1, dtype=np.float64)

    def objective(value):
        weights, bias = value[:-1], value[-1]
        logits = normalized @ weights + bias
        signed_logits = (2 * targets - 1) * logits
        loss = float(sample_weights @ np.logaddexp(0, -signed_logits)
                     + .5 * penalty * (weights @ weights))
        residual = sample_weights * (_sigmoid(logits) - targets)
        gradient = np.empty_like(value)
        gradient[:-1] = normalized.T @ residual + penalty * weights
        gradient[-1] = residual.sum()
        return loss, gradient

    loss, gradient = objective(coefficients)
    history = [loss]
    converged = False
    step = 1 / max(1, penalty)
    iterations = 0
    for _ in range(max_iter):
        if float(np.max(np.abs(gradient))) <= 1e-7:
            converged = True
            break
        squared_norm = float(gradient @ gradient)
        candidate_step = step
        accepted = False
        for _ in range(60):
            proposed = coefficients - candidate_step * gradient
            proposed_loss, proposed_gradient = objective(proposed)
            if math.isfinite(proposed_loss) and proposed_loss <= loss - 1e-4 * candidate_step * squared_norm:
                accepted = True
                break
            candidate_step *= .5
        if not accepted:
            converged = float(np.max(np.abs(gradient))) <= 1e-6
            break
        previous_loss = loss
        coefficients, loss, gradient = proposed, proposed_loss, proposed_gradient
        history.append(loss)
        iterations += 1
        step = min(candidate_step * 1.5, 1e6)
        if previous_loss - loss <= 1e-12 * max(1, abs(previous_loss)):
            converged = float(np.max(np.abs(gradient))) <= 1e-6
            if converged:
                break

    return {
        "format": "forte_linear_classifier_v1",
        "feature_dim": feature_dim,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "constant_features": constant.tolist(),
        "weights": coefficients[:-1].tolist(),
        "bias": float(coefficients[-1]),
        "regularization": float(regularization),
        "class_weight": "balanced",
        "objective": "balanced_mean_log_loss_plus_l2_over_sample_count",
        "optimizer": "gradient_descent_armijo",
        "n_samples": sample_count,
        "n_iter": iterations,
        "converged": converged,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "loss_history": history,
    }


def predict_probabilities(model, X):
    """Return P(label=1) using only the stored training transform.

    A one-dimensional feature vector represents one sample. A matrix returns
    one probability per row, including an empty result for an empty matrix.
    Neither the model nor the supplied feature values are modified.
    """
    if not isinstance(model, Mapping):
        raise ValueError("A fitted classifier artifact is required")
    dimension = model.get("feature_dim")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Classifier feature dimension is invalid")
    features = _matrix(X, allow_vector=True, allow_empty=True)
    if features.shape[1] != dimension:
        raise ValueError("Feature dimension does not match the classifier")
    try:
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        bias = model["bias"]
        constant = np.asarray(model.get("constant_features", [False] * dimension))
        if (any(value.shape != (dimension,) or not np.isfinite(value).all() for value in (mean, scale, weights))
                or np.any(scale <= 0) or type(bias) not in (int, float) or not math.isfinite(bias)
                or constant.shape != (dimension,) or constant.dtype.kind != "b"):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Classifier preprocessing or coefficients are invalid") from exc
    normalized = _standardize(features, mean, scale, constant)
    with np.errstate(over="ignore"):
        logits = normalized @ weights + bias
    if np.isnan(logits).any():
        raise ValueError("Classifier produced invalid logits")
    return _sigmoid(logits)


def choose_threshold(y_train_oof, probabilities):
    """Calibrate a threshold on training out-of-fold or validation predictions.

    Recall breaks balanced-accuracy ties. Remaining ties prefer a threshold
    nearest 0.5, then the higher threshold. With fewer than two observed classes,
    return 0.5. A dedicated validation partition may calibrate the threshold;
    its resulting metrics are selection metrics rather than an independent
    evaluation. Never use final held-out test labels for threshold calibration.
    """
    targets = _targets(y_train_oof)
    try:
        raw = np.asarray(probabilities)
        if raw.dtype.kind not in "biuf":
            raise ValueError
        scores = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Probabilities must be finite values between zero and one") from exc
    if (scores.ndim != 1 or len(scores) != len(targets) or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Probabilities must match the binary targets and lie between zero and one")
    if len(np.unique(targets)) != 2:
        return .5
    unique = np.unique(scores)
    candidates = np.unique(np.concatenate(([0, .5, 1], unique, (unique[:-1] + unique[1:]) / 2)))
    positive = targets == 1
    positive_count = int(positive.sum())
    negative_count = len(targets) - positive_count

    def quality(threshold):
        predictions = scores >= threshold
        true_positive = int(np.count_nonzero(predictions & positive))
        true_negative = int(np.count_nonzero(~predictions & ~positive))
        # An integer numerator avoids floating-point differences in exact ties.
        return (true_positive * negative_count + true_negative * positive_count,
                true_positive, -abs(float(threshold) - .5), float(threshold))

    return float(max(candidates, key=quality))
