"""Fixed grouped validation and aggregate metrics for exploratory case training.

These helpers never fit a transform, encoder, or classifier. Every source hash
represents one video, and each outer video must be predicted by its held-out
model with that model's inner-selected threshold. The inspected corpus does
not become an untouched external test set through a different partition.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
import random
import re


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _group_labels(groups) -> dict[str, int]:
    labels = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("Each video group must be a record")
        digest, label = group.get("video_sha256"), group.get("binary_label")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError("Each video group needs a valid SHA-256")
        if digest in labels:
            raise ValueError("Validation requires exactly one record per video hash group")
        if not (isinstance(label, bool) or _number(label)) or label not in (0, 1):
            raise ValueError("Each video group needs an explicit binary label")
        labels[digest] = int(label)
    return labels


def stratified_group_folds(groups, n_splits: int = 5, seed: int = 17) -> list[dict]:
    """Return disjoint train/validation hash IDs, independent of input order.

    Both classes must occur in every validation fold. Class remainders rotate
    across folds so total fold sizes also differ by at most one. With the
    current 21 positive and 19 negative groups, five folds each have 8 videos.
    """
    if type(n_splits) is not int or n_splits < 2 or type(seed) is not int:
        raise ValueError("Fold count must be at least two and seed must be an integer")
    labels = _group_labels(groups)
    by_label = {label: sorted(digest for digest in labels if labels[digest] == label) for label in (0, 1)}
    if any(len(ids) < n_splits for ids in by_label.values()):
        raise ValueError("Each binary class needs at least one video per validation fold")
    rng = random.Random(seed)
    buckets = [[] for _ in range(n_splits)]
    offset = 0
    for label in (0, 1):
        ids = by_label[label]
        rng.shuffle(ids)
        for index, digest in enumerate(ids):
            buckets[(offset + index) % n_splits].append(digest)
        offset = (offset + len(ids)) % n_splits
    all_ids = set(labels)
    return [{"fold": index, "train_ids": sorted(all_ids - set(ids)), "validation_ids": sorted(ids)}
            for index, ids in enumerate(buckets)]


def nested_group_folds(groups, outer_splits: int = 5, inner_splits: int = 4, seed: int = 17) -> list[dict]:
    """Reserve outer test videos before forming any inner training selections.

    Inner seeds are ``seed + 1009 * (outer_fold + 1)``. All learned transforms,
    hyperparameters and thresholds belong inside the corresponding outer
    training IDs. Baseline and proposed models should share this exact plan.
    """
    groups = list(groups)
    outer = stratified_group_folds(groups, outer_splits, seed)
    by_id = {group["video_sha256"]: group for group in groups}
    result = []
    for fold in outer:
        training = [by_id[digest] for digest in fold["train_ids"]]
        inner = stratified_group_folds(training, inner_splits, seed + 1009 * (fold["fold"] + 1))
        result.append({"fold": fold["fold"], "train_ids": fold["train_ids"],
                       "test_ids": fold["validation_ids"], "inner_folds": inner})
    return result


def _labels_scores(labels, probabilities) -> tuple[list[int], list[float]]:
    labels, probabilities = list(labels), list(probabilities)
    if len(labels) != len(probabilities):
        raise ValueError("Every binary label needs exactly one probability")
    if any(not (isinstance(label, bool) or _number(label)) or label not in (0, 1) for label in labels):
        raise ValueError("Metrics require explicit binary labels")
    if any(not _number(score) or not 0 <= score <= 1 for score in probabilities):
        raise ValueError("Probabilities must be finite values between zero and one")
    return [int(label) for label in labels], [float(score) for score in probabilities]


def _threshold(value) -> float:
    if not _number(value) or not 0 <= value <= 1:
        raise ValueError("A decision threshold must be finite and between zero and one")
    return float(value)


def _metrics(labels: list[int], decisions: list[bool]) -> dict:
    tp = sum(label == 1 and decision for label, decision in zip(labels, decisions))
    tn = sum(label == 0 and not decision for label, decision in zip(labels, decisions))
    fp = sum(label == 0 and decision for label, decision in zip(labels, decisions))
    fn = sum(label == 1 and not decision for label, decision in zip(labels, decisions))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"samples": len(labels), "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn,
            "accuracy": (tp + tn) / len(labels) if labels else None,
            "recall": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def safe_binary_metrics(labels, probabilities, threshold: float = .5) -> dict:
    """Compute aggregate metrics; undefined empty/single-class rates stay None."""
    labels, probabilities = _labels_scores(labels, probabilities)
    threshold = _threshold(threshold)
    return _metrics(labels, [score >= threshold for score in probabilities])


def choose_sensitivity_balanced_threshold(labels, probabilities) -> float:
    """Select a threshold using training OOF probabilities only.

    Predeclared tie order: maximum balanced accuracy, maximum minimum of recall
    and specificity, nearest to .5, then higher threshold. Never supply outer
    held-out labels. A final full-corpus training threshold does not replace
    the inner-selected thresholds used to report nested outer performance.
    """
    labels, probabilities = _labels_scores(labels, probabilities)
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("Threshold selection requires both binary classes")
    scores = sorted(set(probabilities))
    candidates = {0., .5, 1., *scores, *((first + second) / 2 for first, second in zip(scores, scores[1:]))}

    def quality(threshold):
        tp = sum(label == 1 and score >= threshold for label, score in zip(labels, probabilities))
        tn = sum(label == 0 and score < threshold for label, score in zip(labels, probabilities))
        # Both integer numerators share a positive denominator for exact ties.
        return (tp * negative_count + tn * positive_count,
                min(tp * negative_count, tn * positive_count), -abs(threshold - .5), threshold)

    return float(max(candidates, key=quality))


def evaluate_oof_predictions(groups, predictions) -> dict:
    """Aggregate exactly one held-out prediction per hash at its own threshold.

    Prediction records contain ``video_sha256``, ``probability``, ``threshold``.
    Thresholds must already have been fixed without the video's outer label.
    This validates coverage and arithmetic, not how a caller fitted its model.
    Returned values contain no video IDs, source paths, or per-video targets.
    """
    labels = _group_labels(groups)
    if not labels:
        raise ValueError("OOF evaluation requires labeled video groups")
    seen, truth, decisions = set(), [], []
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ValueError("Each OOF prediction must be a record")
        digest = prediction.get("video_sha256")
        if not isinstance(digest, str) or digest not in labels or digest in seen:
            raise ValueError("OOF predictions must cover each known video hash exactly once")
        probability = prediction.get("probability")
        _, checked = _labels_scores([labels[digest]], [probability])
        threshold = _threshold(prediction.get("threshold"))
        seen.add(digest)
        truth.append(labels[digest])
        decisions.append(checked[0] >= threshold)
    if seen != set(labels):
        raise ValueError("OOF predictions are missing one or more video hash groups")
    return _metrics(truth, decisions)
