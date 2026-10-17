"""Inventory local videos and export leakage-aware training manifests.

All row-level paths, hashes, weak labels and source candidates stay in ignored
data/training. Only aggregate, content-free statistics go into docs/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from openpyxl import load_workbook

from worker.augmentation_data import (SCHEMA_VERSION, VIDEO_SUFFIXES, assign_supervision, augmentation_family,
                                      export_supervised_jsonl, inspect_video, load_reviewed_provenance,
                                      rank_source_candidates, read_explicit_workbook, sha256_file)
from worker.training_data import read_simplified_labels


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _workbook_summary(path: Path) -> dict:
    book = load_workbook(path, read_only=True, data_only=False)
    sheets = []
    try:
        for sheet in book:
            rows = list(sheet.values)
            if not rows:
                continue
            headers = rows[0]
            review_columns = [index for index, header in enumerate(headers) if header and "проверка человеком" in str(header).casefold()]
            review_values = Counter(str(row[index]).casefold().strip() for row in rows[1:] for index in review_columns if index < len(row))
            sheets.append({"sheet": sheet.title, "data_rows": len(rows) - 1,
                           "human_review_pending_rows": review_values.get("pending", 0),
                           "has_human_review_field": bool(review_columns),
                           "formula_cells": sum(isinstance(cell, str) and cell.startswith("=") for row in rows for cell in row)})
    finally:
        book.close()
    return {"sha256": sha256_file(path), "sheets": sheets}


def run_audit(args) -> dict:
    cv2.setNumThreads(1)
    originals_dir, clips_dir = args.originals.resolve(), args.clips.resolve()
    original_paths = sorted(path for path in originals_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    clip_paths = sorted(path for path in clips_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not original_paths:
        raise ValueError("No original videos found")
    all_paths = original_paths + clip_paths
    records, signatures = [], {}
    print(json.dumps({"stage": "hash_and_temporal_samples", "originals": len(original_paths), "clips": len(clip_paths)}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (record, signature) in enumerate(pool.map(inspect_video, all_paths)):
            records.append(record)
            if signature is not None:
                signatures[record["sha256"]] = signature
            if (index + 1) % 40 == 0 or index + 1 == len(all_paths):
                print(json.dumps({"inspected": index + 1, "total": len(all_paths)}), flush=True)
    originals, clips = records[:len(original_paths)], records[len(original_paths):]
    original_unique = list({record["sha256"]: record for record in originals}.values())
    labels = read_explicit_workbook(args.labels, originals_dir)
    comparisons = []
    possible_categories = None
    reference_rows = {row["video_path"]: row["binary_label"] for row in labels["rows"]}
    for path in sorted(args.labels.parent.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        comparison = {"workbook": path.name, **_workbook_summary(path)}
        try:
            candidate = read_explicit_workbook(path, originals_dir)
            pairs = [(row["binary_label"], reference_rows[row["video_path"]]) for row in candidate["rows"] if row["video_path"] in reference_rows]
            comparison.update({"matched_rows": len(pairs), "column_c_disagreements": sum(left != right for left, right in pairs),
                               "invalid_or_unmatched_rows": sum(bool(row["issues"]) for row in candidate["rows"])})
        except ValueError:
            comparison["compatible_label_schema"] = False
        comparisons.append(comparison)
        if path.name == "open_labels.simplified.xlsx":
            try:
                simplified = read_simplified_labels(path, originals_dir)
                category_counts = defaultdict(Counter)
                for group in simplified["groups"]:
                    for category in group["observed_categories"]:
                        category_counts[category]["positive_video_groups" if group["binary_label"] else "negative_video_groups"] += 1
                possible_categories = {"groups_with_possible_category_hints": sum(group["category_mask"] for group in simplified["groups"]),
                                       "groups_without_possible_category_hints": sum(not group["category_mask"] for group in simplified["groups"]),
                                       "distinct_possible_category_hints": len(category_counts),
                                       "counts": {category: dict(counts) for category, counts in sorted(category_counts.items())},
                                       "usable_as_reviewed_category_supervision": False}
            except ValueError:
                possible_categories = {"compatible_simplified_schema": False, "usable_as_reviewed_category_supervision": False}
    for clip in clips:
        family, transform = augmentation_family(Path(clip["path"]))
        clip.update({"family": family, "transform_name_hint": transform,
                     "source_candidates": rank_source_candidates(clip, signatures.get(clip["sha256"]), original_unique, signatures)})
    reviewed = load_reviewed_provenance(args.reviewed_provenance, clips, originals)
    assigned = assign_supervision(originals, clips, labels, reviewed, args.seed)
    manifest = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
                "labels": labels, "original_files": originals, **assigned,
                "reviewed_provenance_sha256": sha256_file(args.reviewed_provenance) if args.reviewed_provenance else None,
                "category_supervision": "disabled: workbook possible-category hints and cloud-assisted events are not human-reviewed targets",
                "sampling": {"frames_per_video": 5, "purpose": "source-candidate_review_only", "source_pts": False}}
    output = args.private_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    export_counts = export_supervised_jsonl(manifest, output / "binary")
    _json(output / "manifest.json", manifest)
    # Review queue contains candidates and immutable source identities, without
    # guessed answers or a pre-filled human attestation.
    queue = [{"clip_path": clip["path"], "clip_sha256": clip["sha256"], "family": clip["family"],
              "transform_name_hint": clip["transform_name_hint"], "source_candidates": clip["source_candidates"],
              "human_review_status": "pending", "binary_label": clip["binary_label"],
              "needs": ["verify_original_link", "verify_full_timeline", "verify_label_preservation_and_visibility"]}
             for clip in clips if clip["provenance_status"] == "unverified"]
    (output / "augmentation-review.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in queue), encoding="utf-8")
    _json(output / "reviewed-provenance.template.json", {"schema_version": "forte-reviewed-augmentation-provenance/1",
          "human_review_attested": False, "reviewer": None, "reviewed_at": None, "links": [],
          "link_fields": {"clip_sha256": "exact reviewed derivative SHA-256", "source_sha256": "exact original SHA-256",
                          "label_preserving": True, "full_video": True, "time_offset_sec": 0,
                          "time_scale": 1, "horizontal_flip": "reviewed boolean"}})
    families = defaultdict(list)
    for clip in clips:
        families[clip["family"]].append(clip)
    family_matches = []
    for members in families.values():
        hints = {member["source_candidates"][0]["source_sha256"] for member in members if member["source_candidates"] and member["source_candidates"][0]["strong_candidate"]}
        strong_count = sum(bool(member["source_candidates"] and member["source_candidates"][0]["strong_candidate"]) for member in members)
        family_matches.append({"variant_count": len(members), "strong_candidate_variants": strong_count,
                               "candidate_source_count": len(hints), "all_variants_consistent": len(hints) == 1 and strong_count == len(members)})
    labels_count = Counter(group["binary_label"] for group in manifest["source_groups"] if group["valid"])
    originals_hashes, clips_hashes = Counter(record["sha256"] for record in originals), Counter(record["sha256"] for record in clips)
    durations = [source["duration_sec_estimate"] for source in original_unique if source["decode_status"] == "sampled"]
    candidate_scores = [clip["source_candidates"][0]["mean_frame_correlation"] for clip in clips if clip["source_candidates"]]
    candidate_sources = {clip["source_candidates"][0]["source_sha256"] for clip in clips if clip["source_candidates"] and clip["source_candidates"][0]["strong_candidate"]}
    originals_by_hash = {source["sha256"]: source for source in originals}
    exposure_deltas = defaultdict(list)
    for clip in clips:
        if clip["source_candidates"] and clip.get("photometric_samples"):
            source = originals_by_hash[clip["source_candidates"][0]["source_sha256"]]
            exposure_deltas[clip["transform_name_hint"]].append(statistics.mean(
                candidate["mean_luma_0_255"] - original["mean_luma_0_255"]
                for candidate, original in zip(clip["photometric_samples"], source["photometric_samples"])))
    report = {"schema_version": SCHEMA_VERSION, "generated_at": manifest["generated_at"],
              "run": {"all_video_bytes_hashed": True, "temporal_samples_per_video": 5,
                      "perceptual_method": "five fraction-aligned grayscale correlation and DCT pHash samples; both horizontal orientations; duration gate",
                      "candidate_is_verified_provenance": False, "inference_or_remote_upload": False},
              "originals": {"files": len(originals), "bytes": sum(record["size_bytes"] for record in originals),
                            "unique_sha256_groups": len(originals_hashes), "exact_duplicate_groups": sum(count > 1 for count in originals_hashes.values()),
                            "positive_groups": labels_count[1], "negative_groups": labels_count[0],
                            "invalid_label_groups": sum(not group["valid"] for group in manifest["source_groups"]),
                            "weak_positive_anchor_points": sum(len(group["weak_anchors_sec"]) for group in manifest["source_groups"] if group["binary_label"] == 1),
                            "positive_groups_with_weak_anchors": sum(bool(group["weak_anchors_sec"]) for group in manifest["source_groups"] if group["binary_label"] == 1),
                            "resolution_histogram": dict(Counter(f"{record['width']}x{record['height']}" for record in original_unique if record["decode_status"] == "sampled")),
                            "unique_duration_total_sec_estimate": round(sum(durations), 3),
                            "duration_range_sec_estimate": [round(min(durations), 3), round(max(durations), 3)] if durations else []},
              "augmentations": {"files": len(clips), "bytes": sum(record["size_bytes"] for record in clips),
                                "unique_sha256_groups": len(clips_hashes), "exact_duplicate_groups": sum(count > 1 for count in clips_hashes.values()),
                                "byte_identical_to_original_groups": len(set(originals_hashes) & set(clips_hashes)),
                                "filename_families": len(families), "family_size_histogram": dict(Counter(len(members) for members in families.values())),
                                "transform_name_hints": dict(Counter(clip["transform_name_hint"] for clip in clips)),
                                "strong_perceptual_candidate_files": sum(bool(clip["source_candidates"] and clip["source_candidates"][0]["strong_candidate"]) for clip in clips),
                                "consistent_candidate_families": sum(match["all_variants_consistent"] for match in family_matches),
                                "distinct_original_candidate_groups": len(candidate_sources),
                                "top_candidate_correlation_range": [min(candidate_scores), max(candidate_scores)] if candidate_scores else [],
                                "median_mean_luma_delta_0_255_by_transform": {name: round(statistics.median(values), 4) for name, values in sorted(exposure_deltas.items())},
                                "human_reviewed_provenance_files": len(reviewed),
                                "quarantined_files": sum(clip["split"] == "quarantine" for clip in clips),
                                "quarantine_reasons": dict(Counter(clip["quarantine_reason"] for clip in clips if clip["quarantine_reason"]))},
              "decode": {"sampled_files": sum(record["decode_status"] == "sampled" for record in records),
                         "failed_files": sum(record["decode_status"] != "sampled" for record in records)},
              "workbooks": comparisons, "possible_category_hints": possible_categories, "supervised_binary_export": export_counts,
              "source_split_groups": dict(Counter(group["split"] for group in manifest["source_groups"])),
              "split_scope": "exploratory inspected-source hash split; camera/session independence unknown; no external generalization claim",
              "not_supervised": ["boxes", "keypoints", "track_ids", "zone_polygons", "human_reviewed_event_intervals", "human_reviewed_rule_predicates", "human_reviewed_vlm_evidence_answers"],
              "privacy": "Aggregate report only. Paths, row labels, hashes of individual media, review queues and manifests remain in ignored local data/training."}
    _json(args.output, report)
    print(json.dumps({"stage": "complete", "original_groups": len(originals_hashes), "clips": len(clips),
                      "strong_candidates": report["augmentations"]["strong_perceptual_candidate_files"],
                      "quarantined": report["augmentations"]["quarantined_files"], "binary_export": export_counts}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", type=Path, default=ROOT / "open" / "videos")
    parser.add_argument("--clips", type=Path, default=ROOT / "clips")
    parser.add_argument("--labels", type=Path, default=ROOT / "open" / "open_labels.xlsx")
    parser.add_argument("--reviewed-provenance", type=Path)
    parser.add_argument("--private-dir", type=Path, default=ROOT / "data" / "training" / "augmentation-audit")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "finetuning-data.json")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--seed", type=int, default=17)
    run_audit(parser.parse_args())


if __name__ == "__main__":
    main()
