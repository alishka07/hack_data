"""Weak-anchor event learning on frozen window features, using only NumPy.

Each input bag represents one independent video. Positive training rows are
restricted to its weak-anchor mask; unselected positive windows are unknown.
Every negative window is eligible, with equal total weight per video.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


def _bag(values):
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "biuf":
            raise ValueError
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Window features must be a finite numeric matrix") from exc
    if result.ndim != 2 or not all(result.shape) or not np.isfinite(result).all():
        raise ValueError("Window features must be a nonempty finite numeric matrix")
    return result


def _sigmoid(values):
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def _standardize(features, mean, scale, constant):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (features - mean) / scale
        alternative = features / scale - mean / scale
        result = np.where(np.isfinite(result), result, alternative)
    result[:, constant] = 0
    if not np.isfinite(result).all():
        raise ValueError("Window features cannot be standardized to finite values")
    return result


def _weighted_pca(features, weights, requested):
    magnitude = np.max(np.abs(features), axis=0)
    magnitude[magnitude == 0] = 1
    relative = features / magnitude
    relative_mean = weights @ relative
    relative_scale = np.sqrt(weights @ ((relative - relative_mean) ** 2))
    mean, scale = relative_mean * magnitude, relative_scale * magnitude
    constant = np.all(features == features[0], axis=0) | (relative_scale == 0) | (scale == 0)
    mean[constant] = features[0, constant]
    scale[constant] = 1
    normalized = _standardize(features, mean, scale, constant)
    _, singular, components = np.linalg.svd(normalized * np.sqrt(weights[:, None]), full_matrices=False)
    tolerance = (singular[0] if len(singular) else 0) * max(normalized.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    count = min(requested, rank, len(features) - 1, features.shape[1])
    components, output_scale = components[:count].copy(), singular[:count].copy()
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    transformed = (normalized @ components.T) / output_scale
    return {"mean": mean.tolist(), "scale": scale.tolist(), "constant_features": constant.tolist(),
            "components": components.tolist(), "output_scale": output_scale.tolist(),
            "n_components": count, "requested_components": requested}, transformed


def _fit_weighted_head(features, targets, weights, penalty, max_iter):
    design = np.column_stack((features, np.ones(len(features))))
    coefficients = np.zeros(design.shape[1])
    positive_mass = float(weights @ targets)
    coefficients[-1] = math.log(positive_mass / (1 - positive_mass))
    diagonal = np.ones(design.shape[1]) * penalty
    diagonal[-1] = 0

    def objective(value, hessian=False):
        logits = design @ value
        probabilities = _sigmoid(logits)
        loss = float(weights @ np.logaddexp(0, -(2 * targets - 1) * logits)
                     + .5 * (diagonal * value) @ value)
        gradient = design.T @ (weights * (probabilities - targets)) + diagonal * value
        if hessian:
            curvature = weights * probabilities * (1 - probabilities)
            matrix = design.T @ (curvature[:, None] * design) + np.diag(diagonal)
            return loss, gradient, matrix
        return loss, gradient

    loss, gradient, hessian = objective(coefficients, True)
    history, converged = [loss], False
    for _ in range(max_iter):
        if float(np.max(np.abs(gradient))) < 1e-7:
            converged = True
            break
        damped = hessian + np.eye(len(coefficients)) * 1e-10
        try:
            direction = np.linalg.solve(damped, gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(damped, gradient, rcond=None)[0]
        descent = float(gradient @ direction)
        if not math.isfinite(descent) or descent <= 0:
            direction, descent = gradient, float(gradient @ gradient)
        step, accepted = 1., False
        for _ in range(50):
            proposed = coefficients - step * direction
            proposed_loss, _ = objective(proposed)
            if math.isfinite(proposed_loss) and proposed_loss <= loss - 1e-4 * step * descent:
                accepted = True
                break
            step *= .5
        if not accepted:
            converged = float(np.max(np.abs(gradient))) < 1e-6
            break
        coefficients = proposed
        loss, gradient, hessian = objective(coefficients, True)
        history.append(loss)
    return {"weights": coefficients[:-1].tolist(), "bias": float(coefficients[-1]),
            "initial_loss": history[0], "final_loss": history[-1], "loss_history": history,
            "n_iter": len(history) - 1, "converged": converged,
            "optimizer": "damped_newton_armijo", "intercept_penalized": False}


def fit_event_classifier(feature_bags, labels, anchor_masks, regularization=1.0,
                         pca_components=8, max_iter=100):
    """Fit weighted PCA and a logistic head from eligible training windows.

    Positive videos require a nonempty boolean anchor mask; there is no fallback
    that labels the rest of a positive video. Negative masks may be None and are
    ignored after validation. Each class receives half the weight, divided
    equally among its videos and then their eligible windows. These weights
    determine preprocessing, PCA and logistic loss. The L2 penalty is
    regularization divided by video_count; the bias remains unpenalized.

    Select hyperparameters and threshold on video-level inner out-of-fold
    predictions, fitting this entire transform anew within each training fold.
    """
    if (not isinstance(feature_bags, (list, tuple)) or len(feature_bags) < 2
            or not isinstance(anchor_masks, (list, tuple)) or len(anchor_masks) != len(feature_bags)):
        raise ValueError("Provide one feature bag and anchor mask per video")
    targets = np.asarray(labels)
    if (targets.ndim != 1 or len(targets) != len(feature_bags) or targets.dtype.kind not in "biuf"
            or not np.isfinite(targets).all() or not np.isin(targets, [0, 1]).all()
            or len(np.unique(targets)) != 2):
        raise ValueError("Provide both binary classes with one label per video")
    if (type(regularization) not in (int, float) or not math.isfinite(regularization)
            or regularization < 0):
        raise ValueError("Regularization must be finite and nonnegative")
    if type(pca_components) is not int or pca_components < 1:
        raise ValueError("PCA components must be a positive integer")
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    rows, row_targets, row_weights = [], [], []
    class_counts = {label: int(np.count_nonzero(targets == label)) for label in (0, 1)}
    dimension = None
    for bag_values, target, supplied_mask in zip(feature_bags, targets, anchor_masks):
        bag = _bag(bag_values)
        if dimension is not None and bag.shape[1] != dimension:
            raise ValueError("Every video must use the same feature dimension")
        dimension = bag.shape[1]
        mask = None if supplied_mask is None else np.asarray(supplied_mask)
        if mask is not None and (mask.dtype.kind != "b" or mask.shape != (len(bag),)):
            raise ValueError("Anchor masks must contain one boolean per window")
        if target == 1:
            if mask is None or not mask.any():
                raise ValueError("Positive videos require at least one weak-anchor window")
            eligible = bag[mask]
        else:
            eligible = bag
        rows.append(eligible)
        row_targets.extend([float(target)] * len(eligible))
        row_weights.extend([.5 / (class_counts[int(target)] * len(eligible))] * len(eligible))
    features = np.concatenate(rows)
    weights = np.asarray(row_weights)
    weights /= weights.sum()
    pca, transformed = _weighted_pca(features, weights, pca_components)
    head = _fit_weighted_head(transformed, np.asarray(row_targets), weights,
                              float(regularization) / len(feature_bags), max_iter)
    return {"format": "forte_weak_anchor_event_v1", "feature_dim": dimension,
            "pca": pca, "head": head, "regularization": float(regularization),
            "weighting": "balanced_classes_equal_total_per_video_within_class", "aggregation": "max_window_probability",
            "positive_supervision": "weak_anchor_windows_only",
            "video_count": len(feature_bags), "eligible_window_count": len(features)}


def predict_event_windows(model, bag):
    """Predict each window without labels, anchor masks or refitting PCA."""
    if not isinstance(model, Mapping) or type(model.get("feature_dim")) is not int:
        raise ValueError("A fitted event classifier artifact is required")
    features = _bag(bag)
    dimension = model["feature_dim"]
    if dimension < 1 or features.shape[1] != dimension:
        raise ValueError("Window feature dimension does not match the model")
    try:
        pca, head = model["pca"], model["head"]
        mean, scale = np.asarray(pca["mean"], dtype=float), np.asarray(pca["scale"], dtype=float)
        constant = np.asarray(pca["constant_features"])
        count = pca["n_components"]
        components = np.asarray(pca["components"], dtype=float)
        if type(count) is not int or not 0 <= count <= dimension:
            raise ValueError
        if count == 0 and components.size == 0:
            components = components.reshape(0, dimension)
        output_scale = np.asarray(pca["output_scale"], dtype=float)
        coefficients, bias = np.asarray(head["weights"], dtype=float), head["bias"]
        if (mean.shape != (dimension,) or scale.shape != (dimension,)
                or constant.shape != (dimension,) or constant.dtype.kind != "b"
                or components.shape != (count, dimension) or output_scale.shape != (count,)
                or coefficients.shape != (count,) or np.any(scale <= 0) or np.any(output_scale <= 0)
                or not all(np.isfinite(value).all() for value in (mean, scale, components, output_scale, coefficients))
                or type(bias) not in (int, float) or not math.isfinite(bias)):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Event classifier preprocessing or coefficients are invalid") from exc
    normalized = _standardize(features, mean, scale, constant)
    with np.errstate(over="ignore", invalid="ignore"):
        projected = (normalized @ components.T) / output_scale
        logits = projected @ coefficients + bias
    if np.isnan(logits).any():
        raise ValueError("Event classifier produced invalid logits")
    return _sigmoid(logits)


def predict_event_score(model, bag):
    """Aggregate a video's window predictions by their maximum."""
    return float(np.max(predict_event_windows(model, bag)))
