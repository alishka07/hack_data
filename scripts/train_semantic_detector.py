"""Fit an exploratory semantic video model without feeding labels to the encoder.

Only counts and aggregate metrics are printed. Pixel-only answers are cached
under their complete encoder recipe; labels and per-video predictions remain
in private training manifests. No existing evaluation report is overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.grouped_validation import (choose_sensitivity_balanced_threshold, evaluate_oof_predictions,
                                       nested_group_folds, safe_binary_metrics, stratified_group_folds)
from worker.linear_classifier import fit_linear_classifier, predict_probabilities
from worker.semantic_runtime import LocalRuntime, _validate_json_schema
from worker.semantic_assets import encoder_recipe, recipe_fingerprint
from worker.semantic_encoder import SemanticEncoder, semantic_recipe
from worker.semantic_features import SEMANTIC_KEYS, SEMANTIC_VERSION, observable_schema, semantic_vector
from worker.training_data import read_simplified_labels
from worker.video import AnalysisError


def write_json(path: Path, value) -> None:
    """Commit a complete cache/artifact atomically, including interrupted runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_feature_record(payload, fingerprint: str, digest: str) -> dict:
    """Reject recipe mismatches, extra label fields, or foreign source references."""
    if (not isinstance(payload, dict) or set(payload) != {"recipe_sha256", "video_sha256", "answer", "metadata"}
            or payload["recipe_sha256"] != fingerprint or payload["video_sha256"] != digest):
        raise ValueError("Semantic cache identity or fields are invalid")
    _validate_json_schema(payload["answer"], observable_schema())
    semantic_vector(payload["answer"])
    metadata = payload["metadata"]
    if (not isinstance(metadata, dict)
            or set(metadata) != {"timestamps", "source_indices", "duration_sec", "frame_dimensions", "recipe"}
            or metadata["recipe"] != semantic_recipe()):
        raise ValueError("Semantic cache source metadata is incompatible")
    timestamps, indices, dimensions = (metadata[name] for name in ("timestamps", "source_indices", "frame_dimensions"))
    duration = metadata["duration_sec"]
    if (not isinstance(timestamps, list) or not 1 <= len(timestamps) <= 8
            or not isinstance(indices, list) or not isinstance(dimensions, list)
            or len(indices) != len(timestamps) or len(dimensions) != len(timestamps)
            or type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in timestamps)
            or any(second <= first for first, second in zip(timestamps, timestamps[1:]))
            or timestamps[-1] > duration + .001
            or any(type(index) is not int or index < 0 for index in indices)
            or any(second <= first for first, second in zip(indices, indices[1:]))):
        raise ValueError("Semantic cache source timestamps or indices are invalid")
    for dimension in dimensions:
        if (not isinstance(dimension, dict) or set(dimension) != {"width", "height"}
                or any(type(dimension[name]) is not int or not 0 < dimension[name] <= 960 for name in dimension)
                or dimension["height"] > 448):
            raise ValueError("Semantic cache frame dimensions are incompatible")
    return payload


def collect_features(groups, recipe, data_dir: Path, *, train_only=False) -> tuple[dict, dict]:
    """Audit every cache before starting at most one owned local runtime."""
    fingerprint = recipe_fingerprint(recipe)
    directory = data_dir / "training/semantic_features" / fingerprint
    answers, missing = {}, []
    invalid = 0
    for group in groups:
        digest = group["video_sha256"]
        path = directory / (digest + ".json")
        try:
            payload = validate_feature_record(json.loads(path.read_text(encoding="utf-8")), fingerprint, digest)
        except (OSError, ValueError, TypeError, AnalysisError):
            invalid += path.is_file()
            missing.append(group)
        else:
            answers[digest] = payload["answer"]
    audit = {"groups": len(groups), "cached": len(answers), "missing": len(missing), "invalid": invalid}
    print(json.dumps({"stage": "semantic_cache_audit", **audit}), flush=True)
    if missing and train_only:
        raise ValueError("Train-only mode requires every exact-recipe semantic cache")
    runtime = None
    try:
        if missing:
            runtime = LocalRuntime(ROOT, data_dir)
            if runtime.external:
                raise ValueError("Semantic extraction requires an owned local runtime")
            runtime.start()
            encoder = SemanticEncoder(ROOT, data_dir, runtime)
            for group in missing:
                digest = group["video_sha256"]
                # Only the source path reaches the pixel encoder: never labels,
                # categories, case IDs, source rows, or weak timestamp anchors.
                answer, metadata = encoder.clip_features(Path(group["video_path"]))
                payload = {"recipe_sha256": fingerprint, "video_sha256": digest,
                           "answer": answer, "metadata": metadata}
                validate_feature_record(payload, fingerprint, digest)
                write_json(directory / (digest + ".json"), payload)
                answers[digest] = answer
                print(json.dumps({"stage": "semantic_features", "completed": len(answers), "total": len(groups)}), flush=True)
    finally:
        if runtime is not None:
            runtime.stop()
    return answers, audit


def select_configuration(ids, folds, records, vectors, protocol):
    """Joint family/regularization/threshold selection on inner OOF videos only."""
    settings = protocol["supervised_head"]
    candidates = []
    for family in settings["candidate_families"]:
        if family not in {"main", "pairs"}:
            raise ValueError("Unknown semantic feature family")
        for regularization in settings["regularization_candidates"]:
            scores = {}
            for fold in folds:
                train, valid = fold["train_ids"], fold["validation_ids"]
                if (not set(train + valid).issubset(ids) or set(train) & set(valid)
                        or any(digest in scores for digest in valid)):
                    raise ValueError("Inner folds escape or repeat their training partition")
                fitted = fit_model(train, records, vectors[family], regularization)
                probabilities = predict_probabilities(fitted, np.stack([vectors[family][digest] for digest in valid]))
                scores.update(zip(valid, (float(value) for value in probabilities)))
            if set(scores) != set(ids):
                raise ValueError("Inner OOF predictions must cover each training video exactly once")
            truth, values = [records[digest]["binary_label"] for digest in ids], [scores[digest] for digest in ids]
            threshold = choose_sensitivity_balanced_threshold(truth, values)
            metrics = safe_binary_metrics(truth, values, threshold)
            selected = {"feature_family": family, "regularization": regularization, "threshold": threshold,
                        "training_selection_metrics": metrics}
            candidates.append((metrics["balanced_accuracy"], min(metrics["recall"], metrics["specificity"]),
                               regularization, family == "main", selected))
    if not candidates:
        raise ValueError("The semantic protocol has no model candidates")
    return max(candidates, key=lambda row: row[:4])[4]


def fit_model(ids, records, vectors, regularization):
    return fit_linear_classifier(np.stack([vectors[digest] for digest in ids]),
                                 [records[digest]["binary_label"] for digest in ids],
                                 regularization=regularization)


def verified_baseline(path: Path, manifest: dict, folds: list[dict], data_dir: Path) -> dict:
    """Match the preserved baseline to private predictions and the same fold IDs."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("dataset_sha256") != manifest["source_sha256"]
            or report.get("video_groups") != len(manifest["groups"])
            or report.get("outer_folds") != 5 or report.get("inner_folds") != 4):
        raise ValueError("The preserved baseline uses a different dataset or fold protocol")
    expected_fold = {digest: fold["fold"] for fold in folds for digest in fold["test_ids"]}
    for private_manifest in sorted((data_dir / "training").glob("*-event/manifest.json"), reverse=True):
        prior = json.loads(private_manifest.read_text(encoding="utf-8"))
        if prior.get("source_sha256") != manifest["source_sha256"] or prior.get("folds") != folds:
            continue
        predictions_path = private_manifest.parent / "predictions.json"
        if not predictions_path.is_file():
            continue
        predictions = json.loads(predictions_path.read_text(encoding="utf-8")).get("baseline", [])
        if any(row.get("fold") != expected_fold.get(row.get("video_sha256")) for row in predictions):
            continue
        metrics = evaluate_oof_predictions(manifest["groups"], predictions)
        if metrics == report.get("metrics", {}).get("baseline"):
            return {"metrics": metrics, "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    raise ValueError("No preserved baseline prediction snapshot matches the exact grouped folds")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "open/open_labels.simplified.xlsx")
    parser.add_argument("--videos", type=Path, default=ROOT / "open/videos")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("FORTE_DATA_DIR", ROOT / "data")))
    parser.add_argument("--output", type=Path, default=ROOT / "models/case-detector-v3/model.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/semantic-model-evaluation.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", help="Fit only if every exact-recipe feature cache already exists")
    mode.add_argument("--features-only", action="store_true", help="Extract physical observations without fitting a classifier")
    args = parser.parse_args(argv)
    started = time.monotonic()
    protocol_raw = (ROOT / "docs/semantic-model-protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    manifest = read_simplified_labels(args.labels, args.videos)
    groups = manifest["groups"]
    folds = nested_group_folds(groups, outer_splits=5, inner_splits=4, seed=17)
    fold_fingerprint = recipe_fingerprint({"folds": folds})
    baseline = verified_baseline(ROOT / "docs/event-model-evaluation.json", manifest, folds, args.data_dir)
    recipe = encoder_recipe(ROOT)  # Hash actual weights/projector/runtime before any startup.
    fingerprint = recipe_fingerprint(recipe)
    private = args.data_dir / "training" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-semantic-" + uuid4().hex[:8])
    write_json(private / "manifest.json", {**manifest, "folds": folds, "recipe_sha256": fingerprint})
    answers, cache_audit = collect_features(groups, recipe, args.data_dir, train_only=args.train_only)
    if args.features_only:
        print(json.dumps({"stage": "complete_features", "groups": len(answers), "elapsed_sec": round(time.monotonic() - started, 3)}), flush=True)
        return 0
    records = {group["video_sha256"]: group for group in groups}
    vectors = {family: {digest: semantic_vector(answer, interactions=family == "pairs") for digest, answer in answers.items()}
               for family in ("main", "pairs")}
    predictions, reports = [], []
    for outer in folds:
        selected = select_configuration(outer["train_ids"], outer["inner_folds"], records, vectors, protocol)
        family = selected["feature_family"]
        fitted = fit_model(outer["train_ids"], records, vectors[family], selected["regularization"])
        values = predict_probabilities(fitted, np.stack([vectors[family][digest] for digest in outer["test_ids"]]))
        rows = [{"video_sha256": digest, "probability": float(value), "threshold": selected["threshold"], "fold": outer["fold"]}
                for digest, value in zip(outer["test_ids"], values)]
        predictions.extend(rows)
        reports.append({"fold": outer["fold"], "selection": selected,
                        "outer_metrics": evaluate_oof_predictions([records[digest] for digest in outer["test_ids"]], rows)})
        print(json.dumps({"stage": "nested_evaluation", "completed": len(reports), "total": len(folds)}), flush=True)
    metrics = evaluate_oof_predictions(groups, predictions)
    gain = metrics["balanced_accuracy"] - baseline["metrics"]["balanced_accuracy"]
    gate = protocol["inference_gate"]
    approved = (metrics["balanced_accuracy"] >= gate["minimum_nested_balanced_accuracy"]
                and metrics["recall"] >= gate["minimum_nested_recall"]
                and metrics["specificity"] >= gate["minimum_nested_specificity"]
                and gain >= gate["minimum_balanced_accuracy_gain_over_baseline"])
    # Frozen outer predictions above remain the evaluation. Full-corpus OOF
    # selection below determines the final research fit, not new test metrics.
    ids = sorted(records)
    selected = select_configuration(ids, stratified_group_folds(groups, n_splits=4, seed=17), records, vectors, protocol)
    fitted = fit_model(ids, records, vectors[selected["feature_family"]], selected["regularization"])
    model = {"format_version": 3, "model_type": "qwen3vl_semantic_binary_video", "task": "dataset_violation_prediction",
             "trained_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": manifest["source_sha256"],
             "semantic_recipe": recipe, "recipe_fingerprint": fingerprint, "encoder_fingerprint": fingerprint,
             "semantic_version": SEMANTIC_VERSION, "feature_family": selected["feature_family"],
             "classifier": fitted, "threshold": selected["threshold"], "approved_for_inference": approved,
             "evaluation_scope": "exploratory_nested_group_cv", "category_heads": [],
             "event_time_semantics": "clip_prediction_not_temporally_localized"}
    write_json(args.output, model)
    report = {"purpose": "Exploratory nested semantic-observable comparison on repeatedly inspected local videos",
              "dataset_sha256": manifest["source_sha256"], "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
              "fold_plan_sha256": fold_fingerprint, "video_groups": len(groups), "outer_folds": 5, "inner_folds": 4,
              "metrics": {"baseline": baseline["metrics"], "semantic": metrics}, "baseline_report_sha256": baseline["report_sha256"],
              "balanced_accuracy_gain": gain, "folds": reports, "approved_for_inference": approved, "inference_gate": gate,
              "final_model_training_selection": selected, "model_training_groups": len(groups),
              "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "recipe_fingerprint": fingerprint,
              "semantic_version": SEMANTIC_VERSION, "observables": len(SEMANTIC_KEYS), "cache_audit": cache_audit,
              "category_training": "No trained category heads or supervised bounding boxes",
              "event_localization": "Clip prediction only; source samples do not establish an event time",
              "test_scope": "Model-iterated, previously inspected corpus; same grouped folds, possible camera/session overlap, no untouched external test",
              "elapsed_sec": round(time.monotonic() - started, 3)}
    write_json(private / "predictions.json", predictions)
    write_json(args.report, report)
    print(json.dumps({"stage": "complete", "metrics": report["metrics"], "balanced_accuracy_gain": gain,
                      "approved_for_inference": approved, "elapsed_sec": report["elapsed_sec"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(json.dumps({"stage": "interrupted"}), flush=True)
        raise SystemExit(130) from None
    except Exception as exc:
        # Exceptions may contain private source paths or annotation values.
        print(json.dumps({"stage": "failed", "error_type": type(exc).__name__,
                          "error_code": exc.code if isinstance(exc, AnalysisError) else "semantic_training_failed"}), flush=True)
        raise SystemExit(1) from None
