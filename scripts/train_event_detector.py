"""Compare timestamp-supervised action windows with the original clip baseline.

Nested grouped evaluation is exploratory on an already inspected corpus. All
hyperparameters, thresholds and transforms are selected inside training folds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.event_classifier import fit_event_classifier, predict_event_score
from worker.event_features import FEATURE_VERSION, contextualize_windows
from worker.grouped_validation import (choose_sensitivity_balanced_threshold, evaluate_oof_predictions,
                                       nested_group_folds, safe_binary_metrics, stratified_group_folds)
from worker.linear_classifier import choose_threshold, fit_linear_classifier, predict_probabilities
from worker.training_data import read_simplified_labels


def weak_anchor_mask(windows, anchors):
    """Training-only weak points select nearby window centers, never boundaries."""
    centers = np.asarray([row["center_sec"] for row in windows], dtype=float)
    points = np.asarray(anchors, dtype=float)
    if (not len(centers) or points.ndim != 1 or not len(points)
            or not np.isfinite(centers).all() or not np.isfinite(points).all()
            or np.any(points < 0)):
        raise ValueError("Positive event training requires finite weak point anchors")
    mask = np.zeros(len(centers), dtype=bool)
    for point in points:
        distance = np.abs(centers - point)
        eligible = distance <= 2
        if not eligible.any():
            eligible[np.argmin(distance)] = True
        mask |= eligible
    return mask


def select_configuration(kind, ids, folds, records, bags, windows, baseline, protocol):
    """Select settings using only the supplied training IDs and inner folds."""
    candidates = protocol["stronger_model" if kind == "event" else "baseline"]["regularization_candidates"]
    scores_by_candidate = []
    for regularization in candidates:
        scores = {}
        for fold in folds:
            train_ids, valid_ids = fold["train_ids"], fold["validation_ids"]
            if not set(train_ids + valid_ids).issubset(ids) or set(train_ids) & set(valid_ids):
                raise ValueError("Inner fold escapes its training partition")
            fitted = fit_model(kind, train_ids, records, bags, windows, baseline, regularization, protocol)
            scores.update({digest: score_model(kind, fitted, digest, bags, baseline) for digest in valid_ids})
        if set(scores) != set(ids):
            raise ValueError("Inner OOF predictions must cover training groups exactly once")
        truth = [records[digest]["binary_label"] for digest in ids]
        values = [scores[digest] for digest in ids]
        threshold = (choose_sensitivity_balanced_threshold if kind == "event" else choose_threshold)(truth, values)
        metrics = safe_binary_metrics(truth, values, threshold)
        tie = min(metrics["recall"], metrics["specificity"]) if kind == "event" else metrics["recall"]
        scores_by_candidate.append((metrics["balanced_accuracy"], tie, regularization,
                                    {"regularization": regularization, "threshold": threshold,
                                     "training_selection_metrics": metrics}))
    return max(scores_by_candidate, key=lambda row: row[:3])[3]


def fit_model(kind, ids, records, bags, windows, baseline, regularization, protocol):
    truth = [records[digest]["binary_label"] for digest in ids]
    if kind == "baseline":
        return fit_linear_classifier(np.stack([baseline[digest] for digest in ids]), truth,
                                     regularization=regularization)
    # Only the current training groups' labels and anchors enter fitting.
    masks = [weak_anchor_mask(windows[digest], records[digest]["anchors_sec"])
             if records[digest]["binary_label"] else None for digest in ids]
    return fit_event_classifier([bags[digest] for digest in ids], truth, masks,
                                regularization=regularization,
                                pca_components=protocol["stronger_model"]["pca_components"])


def score_model(kind, fitted, digest, bags, baseline):
    # Inference uses numeric pixel features only; no labels or anchor positions.
    return (predict_event_score(fitted, bags[digest]) if kind == "event"
            else float(predict_probabilities(fitted, baseline[digest])[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "open/open_labels.simplified.xlsx")
    parser.add_argument("--videos", type=Path, default=ROOT / "open/videos")
    parser.add_argument("--encoder", type=Path, default=ROOT / "models/action-encoder")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "models/case-detector-v2/model.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/event-model-evaluation.json")
    args = parser.parse_args(argv)
    from worker.action_encoder import ActionEncoder

    started = time.monotonic()
    protocol_path = ROOT / "docs/event-model-protocol.json"
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    manifest = read_simplified_labels(args.labels, args.videos)
    groups = manifest["groups"]
    records = {row["video_sha256"]: row for row in groups}
    splits = nested_group_folds(groups, outer_splits=5, inner_splits=4, seed=17)
    provenance = json.loads((args.encoder / "manifest.json").read_text(encoding="utf-8"))
    feature_config = {"encoder": provenance, "window_seconds": 8, "stride_seconds": 4,
                      "frames_per_window": 16, "context_features": FEATURE_VERSION}
    fingerprint = hashlib.sha256(json.dumps(feature_config, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/training/action_features" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    private = ROOT / "data/training" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-event")
    private.mkdir(parents=True, exist_ok=True)
    (private / "manifest.json").write_text(json.dumps({**manifest, "folds": splits}, indent=2), encoding="utf-8")
    baseline_report = json.loads((ROOT / "docs/trained-case-evaluation.json").read_text())
    if baseline_report["dataset_sha256"] != manifest["source_sha256"]:
        raise ValueError("Baseline feature snapshot differs from the current labels")
    baseline_cache = ROOT / "data/training/features" / baseline_report["feature_fingerprint"]
    bags, windows, baseline = {}, {}, {}
    encoder = None
    for index, group in enumerate(groups):
        digest = group["video_sha256"]
        location, metadata = cache / (digest + ".npz"), cache / (digest + ".json")
        if location.is_file() and metadata.is_file():
            with np.load(location, allow_pickle=False) as stored:
                frozen = stored["features"]
            window_rows = json.loads(metadata.read_text())
        else:
            if encoder is None:
                encoder = ActionEncoder(args.encoder, threads=args.threads)
            frozen, window_rows = encoder.clip_features(Path(group["video_path"]))
            np.savez_compressed(location, features=frozen)
            metadata.write_text(json.dumps(window_rows), encoding="utf-8")
        if len(frozen) != len(window_rows):
            raise ValueError("Window features and source-time metadata disagree")
        bags[digest], windows[digest] = contextualize_windows(frozen), window_rows
        with np.load(baseline_cache / (digest + ".npz"), allow_pickle=False) as stored:
            baseline[digest] = stored["features"]
        print(json.dumps({"stage": "action_features", "completed": index + 1, "total": len(groups),
                          "elapsed_sec": round(time.monotonic() - started, 1)}), flush=True)

    predictions = {"baseline": [], "event": []}
    fold_reports = []
    for outer in splits:
        fold_report = {"fold": outer["fold"]}
        for kind in ("baseline", "event"):
            selected = select_configuration(kind, outer["train_ids"], outer["inner_folds"],
                                            records, bags, windows, baseline, protocol)
            fitted = fit_model(kind, outer["train_ids"], records, bags, windows, baseline,
                               selected["regularization"], protocol)
            rows = [{"video_sha256": digest, "probability": score_model(kind, fitted, digest, bags, baseline),
                     "threshold": selected["threshold"], "fold": outer["fold"]} for digest in outer["test_ids"]]
            predictions[kind].extend(rows)
            metrics = evaluate_oof_predictions([records[digest] for digest in outer["test_ids"]], rows)
            fold_report[kind] = {"selection": selected, "outer_metrics": metrics}
        fold_reports.append(fold_report)
        print(json.dumps({"stage": "nested_evaluation", "completed": len(fold_reports), "total": len(splits),
                          "elapsed_sec": round(time.monotonic() - started, 1)}), flush=True)

    metrics = {kind: evaluate_oof_predictions(groups, rows) for kind, rows in predictions.items()}
    gain = metrics["event"]["balanced_accuracy"] - metrics["baseline"]["balanced_accuracy"]
    gate = protocol["inference_gate"]
    approved = (metrics["event"]["balanced_accuracy"] >= gate["minimum_nested_balanced_accuracy"]
                and metrics["event"]["recall"] >= gate["minimum_nested_recall"]
                and metrics["event"]["specificity"] >= gate["minimum_nested_specificity"]
                and gain >= gate["minimum_balanced_accuracy_gain_over_baseline"])
    # Evaluation is frozen above. The full-data model is a separate final fit,
    # with settings selected by training OOF, never an independently tested fit.
    ids = sorted(records)
    selected = select_configuration("event", ids, stratified_group_folds(groups, n_splits=4, seed=17),
                                    records, bags, windows, baseline, protocol)
    fitted = fit_model("event", ids, records, bags, windows, baseline, selected["regularization"], protocol)
    model = {"format_version": 2, "model_type": "r3d18_weak_anchor_event_video",
             "task": "dataset_violation_prediction", "trained_at_utc": datetime.now(timezone.utc).isoformat(),
             "dataset_sha256": manifest["source_sha256"], "feature_config": feature_config,
             "feature_fingerprint": fingerprint, "classifier": fitted, "threshold": selected["threshold"],
             "approved_for_inference": approved, "evaluation_scope": "exploratory_nested_group_cv",
             "category_heads": [], "event_time_semantics": "highest_scoring_input_window_not_validated_boundary"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = {"purpose": "Nested grouped comparison on previously inspected local videos",
              "dataset_sha256": manifest["source_sha256"], "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
              "video_groups": len(groups), "outer_folds": 5, "inner_folds": 4, "metrics": metrics,
              "balanced_accuracy_gain": gain, "folds": fold_reports,
              "approved_for_inference": approved, "inference_gate": gate,
              "final_model_training_selection": selected, "model_training_groups": len(groups),
              "final_model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "feature_fingerprint": fingerprint, "encoder": provenance,
              "category_training": "No validated category heads; positive labels per category remain sparse",
              "event_localization": "Weak training anchors; sliding-window positions are not reviewed event boundaries",
              "test_scope": "Exploratory nested video-hash-group CV; previous test clips are reused and cameras/sessions may overlap; no untouched external test",
              "elapsed_sec": round(time.monotonic() - started, 3)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (private / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "complete", "metrics": metrics, "balanced_accuracy_gain": gain,
                      "approved_for_inference": approved, "elapsed_sec": report["elapsed_sec"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
