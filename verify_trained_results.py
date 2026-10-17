"""CPU-only integrity audit of saved custom RT-DETR results.

No neural-network inference, DINO regression fixtures, semantic accuracy test,
or automatic banking-violation decision is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from calibrate_threshold import sha256_file
import verify_results as base

ROOT = Path(__file__).resolve().parent
CLASSES = {"safe", "banknote_counter", "laptop", "bag"}


def validate_source_positions(verifier, reference, records):
    """Check sampled PTS by sequential grab; random frame seeking is inexact on VFR MOV."""
    name, path = reference["file"], Path(reference["path"])
    source_hash = sha256_file(path)
    verifier.check(source_hash == reference.get("sha256"), "source_sha256_matches_inventory", name)
    capture = cv2.VideoCapture(str(path))
    if not verifier.check(capture.isOpened(), "source_video_opens_for_pts_check", name):
        capture.release()
        return {"checked_sample_positions": 0, "source_sha256": source_hash}
    checked = 0
    next_frame = 0
    try:
        for index, record in enumerate(records):
            frame_id = record["source_frame"]
            ok = True
            while next_frame <= frame_id:
                ok = capture.grab()
                if not ok:
                    break
                next_frame += 1
            if not verifier.check(ok, "sampled_source_frame_readable", name, index):
                break
            decoded_index = round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            actual_pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if frame_id == 0 and actual_pts < 0:
                actual_pts = 0.0
            verifier.check(decoded_index == frame_id, "sampled_source_frame_index_exact", name,
                           {"sample": index, "requested": frame_id, "decoded": decoded_index})
            verifier.check(base.finite_number(actual_pts) and abs(actual_pts - record["timestamp_sec"]) < 1e-3,
                           "sampled_source_pts_matches_saved_record", name,
                           {"sample": index, "actual": actual_pts, "stored": record["timestamp_sec"]})
            checked += 1
    finally:
        capture.release()
    return {"checked_sample_positions": checked, "source_sha256": source_hash, "source_frames_grabbed": next_frame}


def trained_rows(verifier, folder, summary, records, thresholds, expected_provenance):
    name = Path(summary["video"]).name
    signature = summary["signature"]
    verifier.check(summary.get("model_provenance") == expected_provenance, "trained_provenance_exact", name)
    verifier.check(signature.get("groups") == ["trained"] and signature.get("threshold") is None,
                   "trained_group_with_per_class_thresholds", name)
    verifier.check(signature.get("interval") == 5.0 and signature.get("max_samples") == 0,
                   "five_second_sampling_without_sample_limit", name)
    verifier.check(signature.get("video_preview") is True, "preview_requested", name)
    label_counts = Counter()
    raw_count = 0
    for index, record in enumerate(records):
        for raw in record["raw_detections"]:
            raw_count += 1
            label, score, box = raw.get("label"), raw.get("score"), raw.get("box")
            verifier.check(label in CLASSES, "raw_label_is_one_of_four_trained_classes", name, {"sample": index, "label": label})
            verifier.check(raw.get("query") == "trained", "raw_query_is_trained", name, index)
            valid_score = base.finite_number(score) and 0 <= score <= 1
            verifier.check(valid_score, "raw_score_finite_and_bounded", name, index)
            if label in thresholds and valid_score:
                verifier.check(score >= thresholds[label], "raw_score_meets_fitted_class_threshold", name,
                               {"sample": index, "label": label, "score": score, "threshold": thresholds[label]})
            verifier.check(isinstance(box, list) and len(box) == 4 and all(base.finite_number(v) for v in box)
                           and box[2] > box[0] and box[3] > box[1], "raw_box_finite_positive_area", name, index)
            label_counts[label] += 1
        for group in base.CATEGORIES:
            for row in record[group]:
                verifier.check(row.get("label") in CLASSES, "filtered_label_is_one_of_four_trained_classes", name,
                               {"sample": index, "group": group, "label": row.get("label")})
    thumbnails = summary.get("thumbnails", [])
    verifier.check(len(thumbnails) == min(6, len(records)), "thumbnail_count_matches_first_six_samples", name)
    for index, thumbnail in enumerate(thumbnails):
        if index >= len(records):
            break
        path = (folder / thumbnail["file"]).resolve()
        if not verifier.check(path.is_relative_to(folder.resolve()) and path.is_file(), "thumbnail_path_local_and_present", name, index):
            continue
        # imread(str(path)) cannot reliably open Cyrillic Windows paths.
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        size = (int(frame.shape[1]), int(frame.shape[0])) if frame is not None else None
        expected_size = (summary["image_size"][0], summary["image_size"][1] + base.RENDER_FOOTER)
        verifier.check(size == expected_size, "thumbnail_decodes_with_expected_dimensions", name, index)
        timestamp = thumbnail.get("timestamp_sec")
        verifier.check(base.finite_number(timestamp) and abs(timestamp - records[index]["timestamp_sec"]) < 1e-3,
                       "thumbnail_timestamp_matches_record", name, index)
    return {"raw_detection_count": raw_count, "raw_label_counts": dict(label_counts), "thumbnails_checked": len(thumbnails)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/trained_all_5s")
    parser.add_argument("--inventory", type=Path, default=ROOT / "results/inventory/inventory.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "training/rtdetr_custom_final/best.pth")
    parser.add_argument("--calibration", type=Path, default=ROOT / "training/rtdetr_custom_final/calibration.json")
    parser.add_argument("--annotations", type=Path, default=ROOT / "training/dataset/val.json")
    parser.add_argument("--validation-predictions", type=Path, default=ROOT / "training/rtdetr_custom_final/best_val_predictions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "training/trained_verification.json")
    args = parser.parse_args()
    verifier = base.Verifier()
    inventory = base.read_json(args.inventory)
    calibration = base.read_json(args.calibration)
    batch = base.read_json(args.results / "batch_summary.json")
    if len(batch) != 42 or any(item.get("status") != "completed" for item in batch):
        print("Batch is not yet complete: this verifier requires all 42 completed results.", flush=True)
        return 2
    checkpoint_hash = sha256_file(args.weights)
    filter_hash = sha256_file(ROOT / "candidate_filters.py")
    calibration_hash = sha256_file(args.calibration)
    verifier.check(calibration.get("checkpoint_sha256") == checkpoint_hash, "calibration_checkpoint_hash_matches_actual_weights")
    verifier.check(calibration.get("validation_annotation_sha256") == sha256_file(args.annotations), "calibration_validation_annotation_hash_matches_actual")
    verifier.check(calibration.get("validation_predictions_sha256") == sha256_file(args.validation_predictions), "calibration_validation_prediction_hash_matches_actual")
    verifier.check(calibration.get("fit_type") == "validation_threshold_fitting" and calibration.get("independent_test") is False,
                   "threshold_fit_not_claimed_as_independent_test")
    thresholds = calibration.get("thresholds", {})
    verifier.check(set(thresholds) == CLASSES and all(base.finite_number(v) and 0 < v < 1 for v in thresholds.values()),
                   "exactly_four_valid_class_thresholds")
    expected_provenance = {"engine": "fine_tuned_rtdetrv2", "checkpoint_sha256": checkpoint_hash,
                           "checkpoint": str(args.weights.resolve()), "per_class_thresholds": thresholds,
                           "global_threshold_override": None}
    # Inject the independently constructed expected signature. No DINO fixture
    # code is called: only the reusable video/preview integrity methods below.
    base.PROMPTS["trained"] = json.dumps(expected_provenance, sort_keys=True)
    references = {Path(item["file"]).stem: item for item in inventory["videos"]}
    folders = {path.parent.name: path.parent for path in args.results.glob("*/summary.json")}
    verifier.check(len(references) == 42, "inventory_has_42_videos")
    verifier.check(set(folders) == set(references), "result_directory_set_matches_inventory",
                   detail={"missing": sorted(set(references) - set(folders)), "extra": sorted(set(folders) - set(references))})
    batch_by_stem = {Path(item["video"]).stem: item for item in batch}
    verifier.check(len(batch_by_stem) == len(batch) and set(batch_by_stem) == set(references), "batch_names_match_inventory_once")
    video_results = []
    for stem, reference in references.items():
        if stem not in folders:
            continue
        errors_before = verifier.error_count
        try:
            folder = folders[stem]
            summary = base.read_json(folder / "summary.json")
            verifier.check(summary == batch_by_stem.get(stem), "batch_and_per_video_summary_identical", reference["file"])
            result, records = verifier.video(folder, reference, filter_hash)
            result.update(trained_rows(verifier, folder, summary, records, thresholds, expected_provenance))
            result.update(validate_source_positions(verifier, reference, records))
            result["errors"] = verifier.error_count - errors_before
            video_results.append(result)
            print(f"CHECKED {reference['file']}: {result['samples']} samples; {result['errors']} integrity errors", flush=True)
        except Exception as exc:
            verifier.check(False, "trained_video_verification_exception", reference["file"], f"{type(exc).__name__}: {exc}")
    verifier.check(sha256_file(ROOT / "candidate_filters.py") == filter_hash, "filter_unchanged_during_verification")
    verifier.check(sha256_file(args.weights) == checkpoint_hash, "checkpoint_unchanged_during_verification")
    verifier.check(sha256_file(args.calibration) == calibration_hash, "calibration_unchanged_during_verification")
    verifier.check(len(video_results) == 42, "all_42_video_results_inspected")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if not verifier.error_count else "failed",
        "scope": "CPU-only saved-result integrity audit; no neural inference, semantic accuracy test, or DINO-specific regressions",
        "results": str(args.results.resolve()), "inventory": str(args.inventory.resolve()),
        "model_provenance": expected_provenance, "calibration_sha256": calibration_hash, "filter_sha256": filter_hash,
        "checks": verifier.check_count, "error_count": verifier.error_count, "errors": verifier.errors,
        "errors_truncated": verifier.error_count > len(verifier.errors), "verified_videos": len(video_results),
        "decoded_source_frames_from_matching_inventory": sum(row["decoded_source_frames"] for row in video_results),
        "analyzed_samples": sum(row["samples"] for row in video_results),
        "independently_checked_source_positions": sum(row["checked_sample_positions"] for row in video_results),
        "decoded_preview_frames": sum((row["preview"] or {}).get("decoded_frames", 0) for row in video_results),
        "raw_detection_count": sum(row["raw_detection_count"] for row in video_results),
        "videos": video_results,
        "limitations": ["This checks file integrity, provenance, thresholds and timestamps, not whether object labels are correct.",
                        "All previews use sampled frames; the model did not analyze every source frame.",
                        "Source SHA256 is compared with the earlier full-decode inventory; sampled PTS are checked by sequential grab through intervening frames because VFR random seeking is inexact.",
                        "All saved verdicts must remain not_evaluated and violations empty.",
                        "Threshold fitting used validation data, not an independent test set."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "verified_videos", "analyzed_samples", "independently_checked_source_positions", "decoded_preview_frames", "error_count"]}, ensure_ascii=False, indent=2))
    print(f"Report: {args.output.resolve()}")
    return int(verifier.error_count > 0)


if __name__ == "__main__":
    raise SystemExit(main())
