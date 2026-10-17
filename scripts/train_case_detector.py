"""Train a local video classifier from explicit simplified-workbook labels.

Only aggregate metrics are printed. Private paths, splits and numeric features
stay under data/training; the input workbook is never modified.
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

from worker.linear_classifier import choose_threshold, fit_linear_classifier, predict_probabilities
from worker.training_data import deterministic_split, read_simplified_labels
from worker.vision_encoder import DinoEncoder


def binary_metrics(labels, probabilities, threshold):
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(probabilities) >= threshold
    tp = int(np.sum((truth == 1) & predicted))
    tn = int(np.sum((truth == 0) & ~predicted))
    fp = int(np.sum((truth == 0) & predicted))
    fn = int(np.sum((truth == 1) & ~predicted))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"samples": len(truth), "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn,
            "accuracy": (tp + tn) / len(truth), "recall": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "open/open_labels.simplified.xlsx")
    parser.add_argument("--videos", type=Path, default=ROOT / "open/videos")
    parser.add_argument("--encoder", type=Path, default=ROOT / "models/dinov2")
    parser.add_argument("--output", type=Path, default=ROOT / "models/case-detector/model.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/trained-case-evaluation.json")
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    manifest = read_simplified_labels(args.labels, args.videos)
    splits = deterministic_split(manifest["groups"], seed=17)
    provenance = json.loads((args.encoder / "manifest.json").read_text())
    feature_config = {"encoder": provenance, "max_frames": args.max_frames,
                      "preprocessing": "full_frame_letterbox_224_imagenet_v1",
                      "pooling": "mean_std_last_minus_first_v1"}
    fingerprint = hashlib.sha256(json.dumps(feature_config, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/training/features" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    private = ROOT / "data/training" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    private.mkdir(parents=True, exist_ok=True)
    (private / "manifest.json").write_text(json.dumps({**manifest, "splits": splits}, indent=2), encoding="utf-8")
    encoder = DinoEncoder(args.encoder, threads=args.threads)
    vectors = {}
    for index, group in enumerate(manifest["groups"]):
        destination = cache / (group["video_sha256"] + ".npz")
        if destination.is_file():
            with np.load(destination, allow_pickle=False) as stored:
                vector = stored["features"]
        else:
            vector, timestamps, per_frame = encoder.clip_features(group["video_path"], args.max_frames)
            np.savez_compressed(destination, features=vector, timestamps=np.asarray(timestamps), frames=per_frame)
        if vector.shape != (1152,) or not np.isfinite(vector).all():
            raise ValueError("Invalid frozen visual features")
        vectors[group["video_sha256"]] = vector
        print(json.dumps({"stage": "visual_features", "completed": index + 1,
                          "total": len(manifest["groups"]), "elapsed_sec": round(time.monotonic() - started, 1)}), flush=True)

    arrays = {name: (np.stack([vectors[row["video_sha256"]] for row in groups]),
                     np.asarray([row["binary_label"] for row in groups])) for name, groups in splits.items()}
    X, y = arrays["train"]
    # Hyperparameters are chosen inside the training partition; test labels are
    # read only once the encoder, features, regularization and threshold are fixed.
    rng = np.random.default_rng(17)
    folds = np.zeros(len(y), dtype=int)
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        folds[indices] = np.arange(len(indices)) % 4
    candidates = []
    for regularization in (.03, .1, .3, 1., 3., 10.):
        predictions = np.zeros(len(y))
        for fold in range(4):
            keep = folds != fold
            fitted = fit_linear_classifier(X[keep], y[keep], regularization=regularization)
            predictions[~keep] = predict_probabilities(fitted, X[~keep])
        threshold = choose_threshold(y, predictions)
        metrics = binary_metrics(y, predictions, threshold)
        candidates.append((metrics["balanced_accuracy"], metrics["recall"], regularization, metrics))
    _, _, regularization, cross_validation = max(candidates, key=lambda row: (row[0], row[1], row[2]))
    classifier = fit_linear_classifier(X, y, regularization=regularization)
    validation_probabilities = predict_probabilities(classifier, arrays["validation"][0])
    threshold = choose_threshold(arrays["validation"][1], validation_probabilities)
    metrics = {name: binary_metrics(target, predict_probabilities(classifier, features), threshold)
               for name, (features, target) in arrays.items()}
    test = metrics["test"]
    approved = test["balanced_accuracy"] >= .75 and test["recall"] >= .75 and test["specificity"] >= .5
    model = {"format_version": 1, "model_type": "dinov2_linear_binary_video",
             "trained_at_utc": datetime.now(timezone.utc).isoformat(),
             "dataset_sha256": manifest["source_sha256"], "feature_fingerprint": fingerprint,
             "feature_config": feature_config, "classifier": classifier,
             "threshold": threshold, "regularization": regularization,
             "approved_for_inference": approved, "task": "dataset_violation_prediction",
             "category_heads": [], "category_status": "Insufficient independent examples for validated category heads"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = {"purpose": "Exploratory group-held-out binary training evaluation on previously inspected clips",
              "dataset_sha256": manifest["source_sha256"], "encoder": provenance,
              "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "feature_fingerprint": fingerprint, "max_frames": args.max_frames,
              "splits": metrics, "training_selection_metrics": cross_validation,
              "regularization": regularization, "decision_threshold": threshold,
              "approved_for_inference": approved,
              "category_training": "Not validated: only 1-3 positive video groups per named category; missing category labels are unknown",
              "bounding_box_training": "Not trained: workbook contains no box annotations",
              "test_scope": "8 video groups excluded from fitting, preprocessing, threshold selection and hyperparameter tuning; not an untouched external dataset",
              "elapsed_sec": round(time.monotonic() - started, 3)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
