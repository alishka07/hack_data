"""CPU-only integrity checks for saved batch predictions and H.264 previews.

Run after inference and refiltering have both finished:
    python verify_results.py --results results/all_videos_5s

This reads saved predictions, never runs a neural network, and does not measure
accuracy or establish that any bank rule was violated. Frame-specific regions
below are regression fixtures, never production scene calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2

from run import PROMPTS, RENDER_FOOTER

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("candidates", "review", "suppressed")
INTERPRETATIONS = {
    "candidates": "unverified_object_candidate",
    "review": "ambiguous_object_requires_review",
    "suppressed": "suppressed_candidate",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mp4_sample_entries(path):
    """Read codec sample entries from MP4 stsd boxes, independent of OpenCV tags."""
    data = Path(path).read_bytes()
    found = []

    def boxes(start, end):
        while start + 8 <= end:
            size = int.from_bytes(data[start:start + 4], "big")
            kind = data[start + 4:start + 8]
            header = 8
            if size == 1:
                if start + 16 > end:
                    raise ValueError("Truncated extended MP4 box")
                size = int.from_bytes(data[start + 8:start + 16], "big")
                header = 16
            elif size == 0:
                size = end - start
            if size < header or start + size > end:
                raise ValueError("Invalid MP4 box size")
            payload, finish = start + header, start + size
            if kind in {b"moov", b"trak", b"mdia", b"minf", b"stbl"}:
                boxes(payload, finish)
            elif kind == b"stsd":
                if payload + 8 > finish:
                    raise ValueError("Truncated MP4 stsd box")
                entry_count = int.from_bytes(data[payload + 4:payload + 8], "big")
                cursor = payload + 8
                for _ in range(entry_count):
                    if cursor + 8 > finish:
                        raise ValueError("Truncated MP4 sample entry")
                    length = int.from_bytes(data[cursor:cursor + 4], "big")
                    if length < 8 or cursor + length > finish:
                        raise ValueError("Invalid MP4 sample entry size")
                    found.append(data[cursor + 4:cursor + 8].decode("ascii", errors="replace"))
                    cursor += length
            start = finish

    boxes(0, len(data))
    return found


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def at_region(rows, label, region, center_also=False):
    matches = []
    for row in rows:
        box = row.get("box")
        if row.get("label") != label or not isinstance(box, list) or len(box) != 4:
            continue
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        center_inside = region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]
        if iou(box, region) >= 0.5 or (center_also and center_inside):
            matches.append({key: row.get(key) for key in ("label", "box", "score", "reasons")})
    return matches


class Verifier:
    def __init__(self):
        self.error_count = 0
        self.check_count = 0
        self.errors = []

    def check(self, passed, name, video=None, detail=None):
        self.check_count += 1
        if not passed:
            self.error_count += 1
            if len(self.errors) < 100:
                self.errors.append({"video": video, "check": name, "detail": detail})
        return bool(passed)

    def preview(self, path, source_size, expected_count, video):
        cap = cv2.VideoCapture(str(path))
        if not self.check(cap.isOpened(), "preview_opens", video, str(path)):
            cap.release()
            return {"decoded_frames": 0, "codec": None}
        try:
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
            entries = mp4_sample_entries(path)
            self.check(codec.lower() in {"avc1", "h264"} and "avc1" in entries,
                       "preview_codec_h264_avc1", video,
                       {"opencv_fourcc": codec, "mp4_sample_entries": entries})
            metadata_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.check(metadata_frames == expected_count, "preview_metadata_count", video,
                       {"actual": metadata_frames, "expected": expected_count})
            expected_size = (source_size[0], source_size[1] + RENDER_FOOTER)
            decoded = 0
            sizes = set()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded += 1
                sizes.add((int(frame.shape[1]), int(frame.shape[0])))
            self.check(decoded == expected_count, "preview_decoded_count", video,
                       {"actual": decoded, "expected": expected_count})
            self.check(sizes == {expected_size}, "preview_size_with_footer", video,
                       {"actual": sorted(sizes), "expected": expected_size})
            return {"decoded_frames": decoded, "codec": codec, "mp4_sample_entries": entries,
                    "dimensions": sorted(sizes)}
        finally:
            cap.release()

    def video(self, folder, reference, filter_hash):
        name = reference["file"]
        summary = read_json(folder / "summary.json")
        signature = summary.get("signature", {})
        self.check(summary.get("status") == "completed", "summary_complete", name, summary.get("status"))
        self.check(Path(summary.get("video", "")).name == name, "summary_source_name", name)
        actual_frames, metadata_frames = summary.get("decoded_frames"), summary.get("metadata_frames")
        self.check(actual_frames == reference["decoded_frames"], "decoded_count_matches_inventory", name,
                   {"actual": actual_frames, "expected": reference["decoded_frames"]})
        self.check(metadata_frames == reference["reported_frames"] == actual_frames,
                   "metadata_count_matches_inventory_and_decode", name)
        self.check(summary.get("frame_count_matches") is True, "frame_count_flag", name)
        self.check(summary.get("image_size") in reference["decoded_dimensions"], "source_dimensions", name)
        last_timestamp = summary.get("last_timestamp_sec")
        expected_last = reference.get("last_timestamp_seconds")
        self.check(finite_number(last_timestamp) and finite_number(expected_last)
                   and abs(last_timestamp - expected_last) < 1e-3, "last_source_pts_matches_inventory", name)
        # Some MOV files genuinely begin at -1/30s. The pipeline documents its
        # normalization of frame zero to zero; every subsequent sample must use PTS.
        inventory_first = reference.get("first_timestamp_seconds")
        initial_normalization = int(finite_number(inventory_first) and inventory_first < 0)
        self.check(summary.get("timestamp_problems") == initial_normalization
                   and summary.get("fallback_timestamp_count") == initial_normalization,
                   "only_inventory_supported_initial_pts_normalization", name,
                   {"inventory_first_pts": inventory_first, "expected_normalizations": initial_normalization,
                    "actual_problems": summary.get("timestamp_problems"),
                    "actual_normalizations": summary.get("fallback_timestamp_count")})
        self.check(summary.get("verdict") == "not_evaluated" and summary.get("violations") == [],
                   "no_automatic_violation_verdict", name)
        self.check(signature.get("filter_hash") == filter_hash, "current_filter_hash", name,
                   {"actual": signature.get("filter_hash"), "expected": filter_hash})
        self.check(signature.get("video_size") == reference["bytes"], "source_bytes_match_inventory", name)
        groups = signature.get("groups", [])
        expected_prompts = {group: PROMPTS.get(group) for group in groups}
        self.check(bool(groups) and all(group in PROMPTS for group in groups)
                   and signature.get("prompts") == expected_prompts, "current_prompt_signature", name)
        records = []
        with (folder / "frames.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("Record must be a JSON object")
                        records.append(record)
                    except (ValueError, TypeError) as exc:
                        self.check(False, "valid_jsonl_record", name, {"line": line_number, "error": str(exc)})
        self.check(len(records) == summary.get("analyzed_samples") and len(records) > 0,
                   "jsonl_count_matches_analyzed_samples", name,
                   {"actual": len(records), "expected": summary.get("analyzed_samples")})
        times, frame_ids = [], []
        counts = Counter()
        for ordinal, record in enumerate(records):
            location = f"sample {ordinal}"
            pts, frame_id = record.get("timestamp_sec"), record.get("source_frame")
            valid_pts = finite_number(pts) and 0 <= pts <= (last_timestamp if finite_number(last_timestamp) else -1) + 1e-6
            valid_frame = isinstance(frame_id, int) and not isinstance(frame_id, bool) and 0 <= frame_id < reference["decoded_frames"]
            self.check(valid_pts and valid_frame, "sample_position_valid", name, location)
            if valid_pts:
                times.append(pts)
            if valid_frame:
                frame_ids.append(frame_id)
            expected_source = "frame_index_fps_estimate" if initial_normalization and frame_id == 0 else "video_pts"
            self.check(record.get("timestamp_source") == expected_source
                       and (expected_source == "video_pts" or pts == 0),
                       "sample_uses_source_pts_or_documented_first_frame_normalization", name, location)
            self.check(not record.get("violations") and record.get("verdict", "not_evaluated") == "not_evaluated",
                       "sample_has_no_automatic_violation", name, location)
            fields_valid = all(isinstance(record.get(key), list) for key in ("raw_detections", *CATEGORIES))
            if not self.check(fields_valid, "raw_and_filtered_fields_present", name, location):
                continue
            indices = []
            for category in CATEGORIES:
                rows = record[category]
                counts[category] += len(rows)
                for row in rows:
                    if not self.check(isinstance(row, dict), "filtered_row_is_object", name, location):
                        continue
                    self.check(row.get("interpretation") == INTERPRETATIONS[category],
                               "candidate_interpretation_not_confirmed", name,
                               {"sample": ordinal, "category": category, "value": row.get("interpretation")})
                    indices.append(row.get("detection_index"))
            self.check(all(isinstance(index, int) for index in indices)
                       and sorted(indices) == list(range(len(record["raw_detections"]))),
                       "raw_rows_partitioned_once", name, location)
        self.check(len(times) == len(records) and all(a < b for a, b in zip(times, times[1:])),
                   "unique_strictly_increasing_sample_pts", name)
        self.check(len(frame_ids) == len(records) and all(a < b for a, b in zip(frame_ids, frame_ids[1:])),
                   "unique_strictly_increasing_source_frames", name)
        self.check(all(summary.get("counts", {}).get(key, 0) == counts[key] for key in CATEGORIES),
                   "summary_category_counts_match_jsonl", name)
        preview_name = summary.get("video_preview")
        if self.check(preview_name == "review.mp4", "preview_present_in_summary", name, preview_name):
            preview = self.preview(folder / preview_name, summary["image_size"], len(records), name)
        else:
            preview = None
        return {"video": name, "decoded_source_frames": actual_frames, "samples": len(records),
                "counts": dict(counts), "preview": preview,
                "source_first_pts": inventory_first,
                "initial_negative_pts_normalizations": initial_normalization}, records

    def regions(self, saved_records):
        checks = []
        fixtures = (("X-1.mov", 10.0), ("Х-15.mov", 10.0))
        chosen = {}
        for name, timestamp in fixtures:
            rows = [row for row in saved_records.get(name, [])
                    if finite_number(row.get("timestamp_sec")) and abs(row["timestamp_sec"] - timestamp) <= 0.04]
            if self.check(len(rows) == 1, "regression_fixture_available", name,
                          {"timestamp_sec": timestamp, "matches": len(rows)}):
                chosen[name] = rows[0]

        def check(name, passed, video, roi, evidence, detail):
            self.check(passed, name, video, detail)
            checks.append({"check": name, "passed": bool(passed), "video": video,
                           "timestamp_sec": 10.0, "region_xyxy": roi, "evidence": evidence, "detail": detail})

        if "X-1.mov" in chosen:
            candidates = chosen["X-1.mov"].get("candidates", [])
            counter_roi, outlet_roi, safe_roi = [33, 398, 115, 490], [28, 494, 75, 545], [296, 353, 428, 592]
            found = at_region(candidates, "banknote_counter", counter_roi)
            check("latest_x1_real_counter_remains_candidate", found, "X-1.mov", counter_roi, found,
                  "At least one banknote-counter candidate has IoU >= 0.5 with the manually inspected counter.")
            wrong_bag = at_region(candidates, "plastic_bag", counter_roi, center_also=True)
            check("latest_x1_counter_not_unambiguous_plastic_bag", not wrong_bag, "X-1.mov", counter_roi, wrong_bag,
                  "No plastic-bag candidate overlaps this counter (IoU >= 0.5 or box center within region). Review rows are allowed.")
            for name, roi in (("outlet", outlet_roi), ("safe", safe_roi)):
                wrong_counter = at_region(candidates, "banknote_counter", roi, center_also=True)
                check(f"latest_x1_{name}_not_unambiguous_counter", not wrong_counter, "X-1.mov", roi, wrong_counter,
                      "No banknote-counter candidate overlaps this manually inspected background region. Review rows are allowed.")
        if "Х-15.mov" in chosen:
            roi = [132, 370, 208, 406]
            found = at_region(chosen["Х-15.mov"].get("candidates", []), "plastic_bag", roi)
            check("latest_x15_visible_bag_survives", found, "Х-15.mov", roi, found,
                  "The visible green bag remains a candidate at this specific frame. Its contents are not inferred.")
        return checks


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "all_videos_5s")
    parser.add_argument("--inventory", type=Path, default=ROOT / "results" / "inventory" / "inventory.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "verification.json")
    parser.add_argument("--expected-videos", type=int, default=42)
    args = parser.parse_args()
    verifier = Verifier()
    filter_hash = hashlib.sha256((ROOT / "candidate_filters.py").read_bytes()).hexdigest()
    inventory = read_json(args.inventory)
    references = inventory["videos"]
    expected = {Path(item["file"]).stem: item for item in references}
    actual = {path.parent.name: path for path in args.results.glob("*/summary.json")}
    verifier.check(len(expected) == args.expected_videos, "inventory_expected_video_count",
                   detail={"actual": len(expected), "expected": args.expected_videos})
    verifier.check(set(actual) == set(expected), "complete_summary_directory_set",
                   detail={"missing": sorted(set(expected) - set(actual)), "extra": sorted(set(actual) - set(expected))})
    video_results, saved_records = [], {}
    for stem, reference in expected.items():
        if stem not in actual:
            continue
        before = verifier.error_count
        try:
            result, records = verifier.video(actual[stem].parent, reference, filter_hash)
            result["errors"] = verifier.error_count - before
            video_results.append(result)
            saved_records[reference["file"]] = records
            print(f"CHECKED {reference['file']}: {result['samples']} samples, {result['errors']} errors", flush=True)
        except Exception as exc:
            verifier.check(False, "video_verification_exception", reference["file"], f"{type(exc).__name__}: {exc}")
    region_checks = verifier.regions(saved_records)
    verifier.check(hashlib.sha256((ROOT / "candidate_filters.py").read_bytes()).hexdigest() == filter_hash,
                   "filter_not_changed_during_verification")
    preroll_counts = Counter(row["source_first_pts"] for row in video_results
                            if row["initial_negative_pts_normalizations"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if verifier.error_count == 0 else "failed",
              "results": str(args.results.resolve()), "inventory": str(args.inventory.resolve()),
              "filter_hash": filter_hash, "checks": verifier.check_count,
              "error_count": verifier.error_count, "errors": verifier.errors,
              "errors_truncated": verifier.error_count > len(verifier.errors),
              "verified_videos": len(video_results),
              "decoded_source_frames": sum(row["decoded_source_frames"] for row in video_results
                                           if isinstance(row["decoded_source_frames"], int)),
              "analyzed_samples": sum(row["samples"] for row in video_results),
              "decoded_preview_frames": sum((row["preview"] or {}).get("decoded_frames", 0) for row in video_results),
              "initial_negative_pts_normalizations": sum(row["initial_negative_pts_normalizations"] for row in video_results),
              "timestamp_normalization_notes": {
                  "scope": "Only source frame index 0 is normalized to 0 seconds when its original MOV PTS is negative. All other analyzed timestamps remain source PTS; source frame indices and saved raw prediction counts remain exact.",
                  "first_frame_normalization_groups": [
                      {"original_pts_seconds": pts, "normalized_pts_seconds": 0,
                       "adjustment_seconds": -pts, "video_count": count}
                      for pts, count in sorted(preroll_counts.items())]},
              "region_regressions": region_checks, "videos": video_results,
              "limitations": ["Integrity and frame-specific regression checks are not detection-accuracy measurements.",
                              "Candidates and review rows are not confirmed objects or violations.",
                              "Region fixtures apply only to the named frames, not to all videos or camera angles.",
                              "Preview videos contain sampled frames, not every source frame.",
                              "OpenCV decoding does not detect every possible codec/container problem."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "verified_videos", "decoded_source_frames", "analyzed_samples", "decoded_preview_frames", "error_count")}, ensure_ascii=False, indent=2))
    print(f"Report: {args.output.resolve()}")
    return 1 if verifier.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
