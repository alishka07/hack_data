"""Check saved, real model outputs for the known counter/bag/outlet mistakes.

This does not run a neural network and does not measure detection accuracy.
The source prompt strings are taken from probe_results.json, independently of
the current prompts in run.py. Manually inspected boxes are for these exact
frames only; they are never written into a shared scene configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from candidate_filters import canonical_label, filter_candidates


ROOT = Path(__file__).resolve().parent
BASELINE_NAMES = ("X-1_10.00_equipment.json", "X-1_10.00_plastic_bag.json")
# These two prompts are the actual saved probe experiment, not a moving import
# from run.PROMPTS. An alternative third probe is intentionally not substituted.
PROBE_PROMPTS = (
    "a banknote counting machine. a power strip. an electrical outlet. a laptop. a computer monitor. a safe.",
    "a plastic bag. a banknote counting machine. a power strip.",
)
IMAGE_SIZE = (464, 848)
COUNTER_X1 = [33, 398, 115, 490]
OUTLET_X1 = [28, 494, 75, 545]
GREEN_BAG_X15 = [132, 370, 208, 406]
SAFE_X1 = [296, 353, 428, 592]
CHAIR_X20 = [0, 716, 204, 848]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def at_box(rows, reference, labels, minimum_iou=0.5):
    return [row for row in rows if canonical_label(row.get("label")) in labels
            and row.get("box") and iou(row["box"], reference) >= minimum_iou]


def evidence(rows):
    return [{"label": row["label"], "box": [round(x, 2) for x in row["box"]],
             "score": round(row["score"], 4), "reasons": row.get("reasons", []),
             "source_prompt": row.get("query")} for row in rows]


def file_reference(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(probe_path, baseline_dir, threshold=0.28):
    probe = read_json(probe_path)
    baseline_sources = [baseline_dir / name for name in BASELINE_NAMES]
    baseline = [read_json(path) for path in baseline_sources]
    old_raw = [dict(row, query=source["prompt"]) for source in baseline for row in source["detections"]]
    old_filtered = filter_candidates(old_raw, IMAGE_SIZE)

    def from_probe(video, timestamp):
        sources = [row for row in probe if row["video"] == video and row["time_sec"] == timestamp
                   and row["prompt"] in PROBE_PROMPTS]
        if len(sources) != len(PROBE_PROMPTS) or {x["prompt"] for x in sources} != set(PROBE_PROMPTS):
            raise ValueError(f"Expected exactly the two recorded probe prompts for {video} at {timestamp}s")
        raw = [dict(row, query=source["prompt"]) for source in sources
               for row in source["detections"] if row["score"] >= threshold]
        return raw, filter_candidates(raw, IMAGE_SIZE)

    x1_raw, x1_filtered = from_probe("X-1.mov", 10.0)
    x15_raw, x15_filtered = from_probe("Х-15.mov", 10.0)
    x20_raw, x20_filtered = from_probe("Х-20.mov", 5.0)
    checks = []

    def check(name, passed, rows, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail, "evidence": evidence(rows)})

    old_conflict = at_box(old_filtered["review"], COUNTER_X1, {"banknote_counter", "plastic_bag"})
    old_unambiguous = at_box(old_filtered["candidates"], COUNTER_X1, {"banknote_counter", "plastic_bag"})
    check("original_x1_counter_bag_conflict_requires_review",
          {row["label"] for row in old_conflict} == {"banknote_counter", "plastic_bag"} and not old_unambiguous,
          old_conflict, "The original 0.60 bag and 0.42 counter at the same object are both ambiguous; no score winner.")

    new_conflict = at_box(x1_filtered["review"], COUNTER_X1, {"banknote_counter", "plastic_bag"})
    new_unambiguous = at_box(x1_filtered["candidates"], COUNTER_X1, {"banknote_counter", "plastic_bag"})
    check("probe_x1_counter_bag_conflict_requires_review",
          {row["label"] for row in new_conflict} == {"banknote_counter", "plastic_bag"} and not new_unambiguous,
          new_conflict, "The saved joint prompts still conflict on the real counter; filtering sends both to review.")

    old_outlet_wrong = at_box(old_raw, OUTLET_X1, {"banknote_counter"})
    new_outlet_background = at_box(x1_filtered["candidates"], OUTLET_X1, {"power_strip", "electrical_outlet"})
    new_outlet_wrong = at_box(x1_raw, OUTLET_X1, {"banknote_counter"})
    check("same_x1_outlet_box_now_power_strip_candidate",
          old_outlet_wrong and new_outlet_background and not new_outlet_wrong,
          old_outlet_wrong + new_outlet_background,
          "At this exact lower-left outlet/power-strip box, original output said counter; saved new probe says power strip.")

    green_bag = at_box(x15_filtered["candidates"], GREEN_BAG_X15, {"plastic_bag"})
    check("x15_green_bag_candidate_survives", green_bag,
          green_bag, "The visually inspected green bag at 10s remains a bag candidate; contents are not inferred.")

    # This manually inspected exclusion is only a regression fixture. No scenes/
    # file is created and no coordinates are reused for another camera/video.
    regions = [{"kind": "exclude", "label": "banknote_counter", "source": "manual",
                "box": [25 / 464, 491 / 848, 78 / 464, 548 / 848]}]
    manual_filtered = filter_candidates(old_raw, IMAGE_SIZE, regions)
    manual_excluded = [row for row in at_box(manual_filtered["suppressed"], OUTLET_X1, {"banknote_counter"})
                       if "manual_exclusion_region" in row["reasons"]]
    check("optional_manual_x1_outlet_region_excludes_old_false_counter", manual_excluded,
          manual_excluded, "A manually reviewed, frame-specific ROI also suppresses the original outlet error.")

    remaining = []
    safe_errors = at_box(x1_filtered["candidates"], SAFE_X1, {"banknote_counter"})
    if safe_errors:
        remaining.append({"video": "X-1.mov", "timestamp_sec": 10.0,
                          "description": "Safe/cabinet region still has false banknote-counter candidates in these saved probe prompts.",
                          "evidence": evidence(safe_errors)})
    chair_errors = at_box(x20_filtered["candidates"], CHAIR_X20, {"plastic_bag"})
    if chair_errors:
        remaining.append({"video": "Х-20.mov", "timestamp_sec": 5.0,
                          "description": "Foreground chair still has a false plastic-bag candidate in these saved probe prompts.",
                          "evidence": evidence(chair_errors)})

    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "scope": "Saved-frame regression only; not a dataset accuracy benchmark or a test of later prompt edits.",
        "probe_source": file_reference(probe_path),
        "baseline_sources": [file_reference(path) for path in baseline_sources],
        "probe_prompts_used": list(PROBE_PROMPTS), "applied_probe_threshold": threshold,
        "source_probe_generation_threshold": "not recorded in probe_results.json",
        "image_size": list(IMAGE_SIZE),
        "reference_boxes": {"counter_x1_10s": COUNTER_X1, "outlet_x1_10s": OUTLET_X1,
                            "green_bag_x15_10s": GREEN_BAG_X15},
        "reference_box_source": "Manual visual inspection of those exact saved frames; not global camera calibration.",
        "checks": checks, "remaining_known_errors_in_saved_probe": remaining,
        "limitations": ["Passed checks do not establish overall precision, recall, or successful violation detection.",
                        "Object candidates and ambiguity labels still require review.",
                        "No model inference, GPU processing, or scene configuration changes occur in this script."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=ROOT / "probe_results.json")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "regression_inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "regression.json")
    args = parser.parse_args()
    result = run(args.probe, args.baseline_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f'{result["status"]}: {len(result["checks"])} saved-frame checks; {len(result["remaining_known_errors_in_saved_probe"])} known residual error locations.')
    print(f'Saved: {args.output}')
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
