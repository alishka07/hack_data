"""Offline tracker comparison on existing local YOLOX perception JSON only.

Run in a separate evaluation environment with trackers==2.6.1 and
supervision==0.26.1. This script does not read video/images, load a detector,
contact the network, or alter production configuration. No tracking ground truth
is inferred from temporary IDs, so counts are diagnostics rather than accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/tracking-mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")


def load_timeline(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    timeline = {0: float(report["coverage"]["first_timestamp_sec"])}
    for row in report["motion"]:
        index, timestamp = row["source_index"], row["timestamp_sec"]
        if type(index) is not int or index < 0 or type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError("Invalid source clock in the cached perception report")
        if index in timeline and timeline[index] != timestamp:
            raise ValueError("Conflicting source clock in the cached perception report")
        timeline[index] = float(timestamp)
    if len(timeline) != report["coverage"]["frames_analyzed"]:
        raise ValueError("The cached report must retain the complete sampled motion timeline")
    grouped = defaultdict(list)
    for row in report["detections"]:
        if row["kind"] != "person":
            continue
        index, timestamp = row["source_index"], row["timestamp_sec"]
        if index not in timeline or timeline[index] != timestamp:
            raise ValueError("Detection is not bound to the cached sampled source timeline")
        box, score = row["bbox"], row["confidence"]
        if (len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1
                or type(score) not in (int, float) or not 0 <= score <= 1):
            raise ValueError("Invalid cached person detection")
        grouped[index].append({"bbox": list(box), "confidence": float(score)})
    frames = [(index, timestamp, grouped[index]) for index, timestamp in sorted(timeline.items(), key=lambda pair: pair[1])]
    cuts = {row["source_index"] for row in report["motion"] if row.get("scene_change_suspected")}
    return report, frames, cuts


def summarize_assignments(assignments, frames):
    counts, last, previous = Counter(), {}, set()
    interrupted = bridged = missing_samples = segments = 0
    max_bridge = 0.
    unassigned = 0
    for ordinal, ((source_index, timestamp, inputs), ids) in enumerate(zip(frames, assignments)):
        if len(ids) > len(inputs) or len(ids) != len(set(ids)):
            raise ValueError("A tracker returned duplicated IDs or more assignments than supplied detections")
        current = set(ids)
        unassigned += len(inputs) - len(ids)
        interrupted += len(previous - current)
        for identity in ids:
            counts[identity] += 1
            prior = last.get(identity)
            if prior is None or prior[0] != ordinal - 1:
                segments += 1
            if prior is not None and prior[0] < ordinal - 1:
                bridged += 1
                missing_samples += ordinal - prior[0] - 1
                max_bridge = max(max_bridge, timestamp - prior[1])
            last[identity] = (ordinal, timestamp)
        previous = current
    return {"unique_track_ids": len(counts), "contiguous_observed_track_segments": segments,
            "assigned_detections": sum(counts.values()), "unassigned_detections": unassigned,
            "track_interruptions": interrupted, "one_sample_track_ids": sum(count == 1 for count in counts.values()),
            "resumed_ids_across_unobserved_samples": bridged, "unobserved_samples_inside_resumed_ids": missing_samples,
            "max_resumption_span_sec": max_bridge,
            "gap_semantics": "Every interrupted interval is unknown visibility, including when a tracker later reuses its ID"}


def run_iou(frames, cuts):
    from worker.perception import IoUTracker
    from worker.video import FrameRef
    tracker = IoUTracker(threshold=.25, max_gap=.8)
    assignments, segment = [], 0
    for index, timestamp, boxes in frames:
        if index in cuts:
            tracker.reset()
            segment += 1
        tracked, _ = tracker.update(boxes, FrameRef(index, timestamp, Path("unused")))
        assignments.append([f"segment-{segment}-{row['track_id']}" for row in tracked])
    return assignments


def run_bytetrack(frames, cuts, fps, lost_buffer):
    import numpy as np
    import supervision as sv
    from trackers import ByteTrackTracker
    settings = {"lost_track_buffer": lost_buffer, "frame_rate": fps, "minimum_consecutive_frames": 1,
                "minimum_iou_threshold": .25, "track_activation_threshold": .35, "high_conf_det_threshold": .5}
    parameters = inspect.signature(ByteTrackTracker).parameters
    if any(name not in parameters for name in settings):
        raise RuntimeError("ByteTrack API differs from the reviewed trackers 2.6.1 API")
    tracker = ByteTrackTracker(**settings)
    assignments, segment = [], 0
    for index, timestamp, boxes in frames:
        if index in cuts:
            tracker = ByteTrackTracker(**settings)
            segment += 1
        xyxy = np.asarray([box["bbox"] for box in boxes], dtype=np.float32).reshape((-1, 4)) * 1000
        detections = sv.Detections(xyxy=xyxy, confidence=np.asarray([box["confidence"] for box in boxes], dtype=np.float32),
                                   class_id=np.zeros(len(boxes), dtype=int))
        tracked = tracker.update(detections, timestamp=timestamp)
        if tracked.tracker_id is None:
            if len(tracked):
                raise RuntimeError("ByteTrack returned nonempty detections without explicit track IDs")
            identities = []
        else:
            identities = tracked.tracker_id
        assignments.append([f"segment-{segment}-track-{int(identity)}" for identity in identities if identity >= 0])
    return assignments


def evaluate(paths: list[Path], repeats=5):
    imported = time.perf_counter()
    import cv2
    import numpy
    import scipy
    import supervision
    import trackers
    import_seconds = time.perf_counter() - imported
    clips = []
    for number, path in enumerate(paths, 1):
        report, frames, cuts = load_timeline(path)
        fps = report["coverage"]["sample_rate"]
        methods = []
        runners = [("adjacent_iou", lambda: run_iou(frames, cuts), {"iou_threshold": .25, "max_adjacent_source_gap_sec": .8,
                                                                "missed_sample_bridging": False}),
                   ("bytetrack_no_lost_buffer", lambda: run_bytetrack(frames, cuts, fps, 0), {"lost_track_buffer_30fps_units": 0}),
                   ("bytetrack_0_2s_lost_buffer", lambda: run_bytetrack(frames, cuts, fps, 6), {"lost_track_buffer_30fps_units": 6}),
                   ("bytetrack_0_8s_lost_buffer", lambda: run_bytetrack(frames, cuts, fps, 24), {"lost_track_buffer_30fps_units": 24})]
        for name, runner, settings in runners:
            runner()  # Exclude imports/initialization warm-up from repeated timings.
            durations = []
            for _ in range(repeats):
                before = time.perf_counter()
                assigned = runner()
                durations.append(time.perf_counter() - before)
            methods.append({"name": name, "settings": settings,
                            "runtime_median_sec": statistics.median(durations), "runtime_repeats": repeats,
                            **summarize_assignments(assigned, frames)})
        clips.append({"clip": f"local-clip-{number}", "source_span_sec": frames[-1][1] - frames[0][1],
                      "sample_rate": fps, "sampled_frames": len(frames), "input_person_detections": sum(len(row[2]) for row in frames),
                      "scene_change_resets": len(cuts), "methods": methods})
    return {"status": "completed", "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Local replay of cached YOLOX person boxes; diagnostic comparison without tracking ground truth",
            "network_during_evaluation": False, "video_or_image_reads": False, "gpu_used": False,
            "production_tracker_changed": False, "dependency_import_sec": import_seconds,
            "environment": {"python": sys.version.split()[0], "trackers": importlib.metadata.version("trackers"),
                            "supervision": supervision.__version__, "numpy": numpy.__version__,
                            "scipy": scipy.__version__, "opencv": cv2.__version__},
            "implementation": {"class": "trackers.ByteTrackTracker", "license": "Apache-2.0",
                               "source": "https://github.com/roboflow/trackers", "supervision_license": "MIT",
                               "api_reference": "https://trackers.roboflow.com/develop/trackers/bytetrack/"},
            "common_bytetrack_settings": {"minimum_consecutive_frames": 1, "minimum_iou_threshold": .25,
                                          "track_activation_threshold": .35, "high_conf_det_threshold": .5,
                                          "frame_rate": "cached sample rate", "clock": "authoritative sampled source PTS",
                                          "coordinate_basis": "normalized xyxy scaled to 1000x1000"},
            "clips": clips,
            "limitations": ["No IDF1, HOTA, MOTA, identity-switch count, or tracking accuracy can be calculated without reviewed identities",
                            "Different ID counts can reflect false merges, suppressed tracks or fragmentation; fewer IDs is not proof of better tracking",
                            "Cached detections were already filtered at confidence .35; ByteTrack cannot recover omitted .1-.35 detections in this replay",
                            "Reused IDs across missing detections remain unknown visibility, not evidence of presence, absence, cash lineage or identity",
                            "Timings exclude detection, video decoding, VLM inference and dependency imports",
                            "Coordinates are normalized for this diagnostic replay, not calibrated real-world positions"],
            "configuration_caveats": ["The zero-lost-buffer setting produced no assigned tracks in trackers2.6.1 timestamp mode on these clips; it is a degenerate observed configuration, not an accuracy result"],
            "decision": "Retain conservative adjacent-IoU production default until reviewed association labels establish benefit; ByteTrack is a feasible CPU option"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/tracking-comparison.json")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 30:
        parser.error("repeats must be from 1 to 30")
    result = evaluate(args.input, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "clips": result["clips"], "output": str(args.output)}, indent=2))
