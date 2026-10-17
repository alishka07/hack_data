"""Local, human-reviewed event annotations and evaluation-only metrics.

This module is deliberately independent of inference. A source hash joins local
evaluation records; it must never decide a production prediction.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from worker.video import Budget, VideoInfo, probe_video


SCHEMA_VERSION = "forte-reviewed-events/1"
STATES = {"supported", "contradicted", "unknown"}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_MAX_JSON_BYTES = 16 * 1024 * 1024


class AnnotationError(ValueError):
    """Errors intentionally omit private paths, text, and scene information."""


def _fail(message: str) -> None:
    raise AnnotationError(message)


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _keys(value, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"Invalid {label} fields")


def _text(value, *, nullable: bool = False) -> bool:
    return (value is None and nullable) or (
        isinstance(value, str) and 0 < len(value.strip()) <= 4000
        and not value.lstrip().startswith(("=", "+", "@"))
    )


def local_file(path: str | Path) -> Path:
    """Reject URLs/UNC paths before probing a regular, existing local file."""
    raw = str(path)
    if re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.I) or raw.startswith(("\\\\", "//")):
        _fail("An existing local file is required")
    try:
        # Windows restricted runtimes can deny GetFinalPathNameByHandle for a
        # freshly renamed file despite allowing stat/read. Check existence
        # explicitly rather than requiring pathlib's strict handle lookup.
        resolved = Path(path).resolve()
        if str(resolved).startswith(("\\\\", "//")) or not resolved.is_file():
            _fail("An existing local file is required")
    except (OSError, ValueError):
        _fail("An existing local file is required")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_source(path: str | Path) -> tuple[VideoInfo, str]:
    path = local_file(path)
    before = path.stat()
    digest = sha256_file(path)
    info = probe_video(path, Budget(time.time() + 90))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        _fail("The source changed during inspection")
    return info, digest


def source_identity(info: VideoInfo, digest: str) -> dict:
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        _fail("Invalid source digest")
    if not info.pts or any(not _number(t) for t in info.pts):
        _fail("Invalid source timestamps")
    if info.pts[0] != 0 or any(b <= a for a, b in zip(info.pts, info.pts[1:])):
        _fail("Invalid source timestamps")
    if not _number(info.duration) or info.duration <= 0 or info.pts[-1] > info.duration:
        _fail("Invalid source duration")
    if (type(info.width) is not int or info.width <= 0 or type(info.height) is not int
            or info.height <= 0 or not _number(info.first_pts)):
        _fail("Invalid source geometry")
    pts_bytes = json.dumps(list(info.pts), separators=(",", ":"), allow_nan=False).encode("ascii")
    return {"sha256": digest, "duration_sec": info.duration, "width": info.width,
            "height": info.height, "frame_count": len(info.pts), "first_pts_sec": info.first_pts,
            "pts_sha256": hashlib.sha256(pts_bytes).hexdigest()}


def annotation_template(info: VideoInfo, digest: str) -> dict:
    """Create an empty draft. No label, event, or human review is inferred."""
    return {"schema_version": SCHEMA_VERSION, "source": source_identity(info, digest),
            "groups": {"camera_id": None, "session_id": None, "related_group_id": None},
            "review": {"status": "draft", "reviewer": None, "reviewed_at": None,
                       "human_review_attested": False, "coverage": "partial",
                       "coverage_intervals": [], "video_verdict": "unknown", "unknowns": []},
            "events": []}


def load_annotation(path: str | Path) -> dict:
    path = local_file(path)
    if path.stat().st_size > _MAX_JSON_BYTES:
        _fail("Annotation file exceeds the size limit")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                _fail("Duplicate JSON fields are not accepted")
            value[key] = item
        return value

    def invalid_constant(_value):
        _fail("Nonfinite JSON numbers are not accepted")

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        _fail("Annotation must be valid UTF-8 JSON")
    if not isinstance(value, dict):
        _fail("Annotation must be a JSON object")
    return value


def _interval(value, duration: float) -> tuple[float, float]:
    _keys(value, {"start_sec", "end_sec"}, "coverage interval")
    start, end = value["start_sec"], value["end_sec"]
    if not _number(start) or not _number(end) or not 0 <= start <= end <= duration:
        _fail("Invalid annotation interval")
    return start, end


def _refs(refs, info: VideoInfo, *, relevance: bool = False) -> list[int]:
    if not isinstance(refs, list):
        _fail("Frame references must be a list")
    indices = []
    for ref in refs:
        _keys(ref, {"source_index", "timestamp_sec"} | ({"relevance"} if relevance else set()),
              "frame reference")
        index, timestamp = ref["source_index"], ref["timestamp_sec"]
        if type(index) is not int or not 0 <= index < len(info.pts) or not _number(timestamp):
            _fail("Invalid source frame reference")
        # JSON round trips preserve these float values. Rounded or guessed times
        # must be replaced by the exact supplied source PTS, never silently snapped.
        if timestamp != info.pts[index]:
            _fail("Frame timestamp does not equal its source PTS")
        if relevance and ref["relevance"] not in {"relevant", "irrelevant", "unknown"}:
            _fail("Invalid evidence relevance label")
        indices.append(index)
    if len(indices) != len(set(indices)):
        _fail("Duplicate source frame references are not accepted")
    return indices


def _facts(facts, info: VideoInfo) -> None:
    if not isinstance(facts, list):
        _fail("Facts must be a list")
    names = set()
    for fact in facts:
        _keys(fact, {"name", "state", "description", "frame_refs"}, "fact")
        name = fact["name"]
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) or name in names:
            _fail("Fact names must be unique identifiers within their list")
        names.add(name)
        if fact["state"] not in STATES or not _text(fact["description"]):
            _fail("Invalid fact state or description")
        refs = _refs(fact["frame_refs"], info)
        if fact["state"] != "unknown" and not refs:
            _fail("Supported and contradicted facts need source frames")


def validate_annotation(annotation: dict, info: VideoInfo, source_sha256: str,
                        *, require_reviewed: bool = False,
                        allowed_types: set[str] | None = None) -> dict:
    """Validate structure and bind every source reference to the actual video.

    A human attestation records a review assertion; software cannot establish
    that the asserted visual interpretation is correct.
    """
    _keys(annotation, {"schema_version", "source", "groups", "review", "events"}, "annotation")
    if annotation["schema_version"] != SCHEMA_VERSION:
        _fail("Unsupported annotation schema version")
    expected_source = source_identity(info, source_sha256)
    if (not isinstance(annotation["source"], dict)
            or set(annotation["source"]) != set(expected_source)
            or any(type(annotation["source"][key]) is not type(value)
                   for key, value in expected_source.items())
            or annotation["source"] != expected_source):
        _fail("Annotation source identity does not match the local video")
    groups = annotation["groups"]
    _keys(groups, {"camera_id", "session_id", "related_group_id"}, "group")
    if any(not _text(value, nullable=True) for value in groups.values()):
        _fail("Invalid annotation grouping identifier")
    review = annotation["review"]
    _keys(review, {"status", "reviewer", "reviewed_at", "human_review_attested", "coverage",
                   "coverage_intervals", "video_verdict", "unknowns"}, "review")
    if review["status"] not in {"draft", "reviewed"} or type(review["human_review_attested"]) is not bool:
        _fail("Invalid human review status")
    if review["coverage"] not in {"partial", "complete"} or review["video_verdict"] not in {
            "violation", "no_violation", "unknown"}:
        _fail("Invalid review coverage or verdict")
    if not isinstance(review["unknowns"], list) or any(not _text(t) for t in review["unknowns"]):
        _fail("Invalid review unknowns")
    if not isinstance(review["coverage_intervals"], list):
        _fail("Coverage intervals must be a list")
    intervals = sorted(_interval(value, info.duration) for value in review["coverage_intervals"])
    end = 0.0
    merged = []
    for start, stop in intervals:
        if review["coverage"] == "complete" and start > end:
            _fail("Complete review coverage contains a gap")
        end = max(end, stop)
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    if review["coverage"] == "complete" and (not intervals or end < info.duration):
        _fail("Complete review must cover the entire source duration")
    reviewed = review["status"] == "reviewed"
    if reviewed:
        if not review["human_review_attested"] or not _text(review["reviewer"]):
            _fail("Reviewed annotations need an explicit named human attestation")
        try:
            when = datetime.fromisoformat(review["reviewed_at"])
            if when.utcoffset() is None:
                _fail("Review timestamp requires a timezone")
        except (TypeError, ValueError):
            _fail("Review timestamp requires an ISO timestamp and timezone")
        if groups["camera_id"] is None or groups["session_id"] is None:
            _fail("Reviewed annotations require camera and session grouping")
    elif review["human_review_attested"] or review["reviewer"] is not None or review["reviewed_at"] is not None:
        _fail("Draft review metadata cannot assert completed human review")
    if require_reviewed and not reviewed:
        _fail("Only explicitly human-reviewed annotations can be imported")
    if not isinstance(annotation["events"], list):
        _fail("Events must be a list")
    ids, supported = set(), 0
    for event in annotation["events"]:
        _keys(event, {"event_id", "type", "start_sec", "end_sec", "status", "visible_facts",
                      "prerequisites", "exceptions", "uncertainty", "evidence_refs"}, "event")
        if not _text(event["event_id"]) or event["event_id"] in ids:
            _fail("Event identifiers must be unique")
        ids.add(event["event_id"])
        kind = event["type"]
        if not isinstance(kind, str) or not _IDENTIFIER.fullmatch(kind):
            _fail("Event type must be an identifier")
        if allowed_types is not None and kind not in allowed_types:
            _fail("Event type is outside the local catalog")
        start, stop = _interval({"start_sec": event["start_sec"], "end_sec": event["end_sec"]}, info.duration)
        if event["status"] not in STATES:
            _fail("Invalid event status")
        for name in ("visible_facts", "prerequisites", "exceptions"):
            _facts(event[name], info)
        if not isinstance(event["uncertainty"], list) or any(not _text(t) for t in event["uncertainty"]):
            _fail("Event uncertainty must be a list of descriptions")
        refs = _refs(event["evidence_refs"], info, relevance=True)
        if any(not start <= info.pts[index] <= stop for index in refs):
            _fail("Event evidence must lie inside the annotated event interval")
        if event["status"] == "supported":
            supported += 1
            if kind == "unknown":
                _fail("A supported event requires an identified event type")
            if not refs or not any(fact["state"] == "supported" for fact in event["visible_facts"]):
                _fail("Supported events need visible facts and source evidence")
            if reviewed and not any(start >= a and stop <= b for a, b in merged):
                _fail("Reviewed supported events must lie in reviewed coverage")
    if review["video_verdict"] == "no_violation" and (
            review["coverage"] != "complete" or supported or review["unknowns"]
            or any(event["status"] == "unknown" for event in annotation["events"])):
        _fail("A negative video verdict requires complete, resolved review")
    if review["video_verdict"] == "violation" and not supported:
        _fail("A positive video verdict needs a supported reviewed event")
    if not reviewed and review["video_verdict"] != "unknown":
        _fail("Draft annotations cannot assert a video verdict")
    return {"schema_version": SCHEMA_VERSION, "reviewed": reviewed, "events": len(ids),
            "supported_events": supported, "coverage": review["coverage"]}


def write_exclusive_json(path: str | Path, value: dict) -> Path:
    """Publish a complete JSON file atomically, without replacing any file."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".annotation-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Windows rename is atomic and refuses an existing destination.
            # POSIX rename replaces files, so use an exclusive hard link there.
            if os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                _fail("The destination already contains a different annotation; nothing was replaced")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def import_annotation(annotation: dict, info: VideoInfo, source_sha256: str,
                      directory: Path, *, attesting_reviewer: str,
                      allowed_types: set[str] | None = None) -> Path:
    validate_annotation(annotation, info, source_sha256, require_reviewed=True, allowed_types=allowed_types)
    if not _text(attesting_reviewer) or attesting_reviewer != annotation["review"]["reviewer"]:
        _fail("Import requires the matching reviewer's explicit attestation")
    return write_exclusive_json(Path(directory) / (source_sha256 + ".json"), deepcopy(annotation))


def split_group_ids(annotations: list[dict]) -> dict[str, str]:
    """Connected groups keep duplicates, sessions and related clips together.

    Camera IDs are reported separately: use a camera holdout to measure view
    generalization. Unknown session IDs cannot establish independent samples.
    """
    parent = {a["source"]["sha256"]: a["source"]["sha256"] for a in annotations}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    seen = {}
    for annotation in annotations:
        digest = annotation["source"]["sha256"]
        groups = annotation["groups"]
        for field in ("session_id", "related_group_id"):
            value = groups[field]
            if value is None:
                continue
            key = (field, value)
            if key in seen:
                roots = sorted((find(digest), find(seen[key])))
                parent[roots[1]] = roots[0]
            else:
                seen[key] = digest
    return {digest: find(digest) for digest in sorted(parent)}


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def proposal_metrics(annotations: list[dict], predictions: list[dict], *,
                     min_iou: float = .1, tolerance_sec: float = 3.0) -> dict:
    """Aggregate already validated human annotations, never weak anchors.

    predictions: video_sha256, status, optional verdict/proposals/violations,
    review_required, latency_sec and peak_gpu_memory_mib. Predictions contain no
    reference labels. Evidence temporal overlap is distinct from a human label.
    """
    if not _number(min_iou) or not 0 < min_iou <= 1 or not _number(tolerance_sec) or tolerance_sec < 0:
        _fail("Invalid evaluation thresholds")
    references = {}
    for annotation in annotations:
        if (annotation.get("schema_version") != SCHEMA_VERSION
                or annotation.get("review", {}).get("status") != "reviewed"
                or annotation["review"].get("human_review_attested") is not True):
            _fail("Metrics require validated human-reviewed annotations")
        digest = annotation["source"]["sha256"]
        if digest in references:
            _fail("Metrics require exactly one annotation per source hash")
        references[digest] = annotation
    by_hash = {}
    for prediction in predictions:
        digest = prediction.get("video_sha256")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or digest in by_hash:
            _fail("Metrics require unique, valid prediction source hashes")
        if prediction.get("status") not in {"done", "failed", "processing"}:
            _fail("Invalid prediction status")
        if prediction["status"] == "done" and prediction.get("verdict") not in {"violation", "no_violation"}:
            _fail("Completed predictions need a binary verdict")
        by_hash[digest] = prediction
    counts = dict.fromkeys(("tp", "fp", "tn", "fn", "failed", "processing", "done", "review_required",
                            "proposal_hits", "supported_events", "localized_events", "emitted_events",
                            "assessable_emitted_events", "unassessed_emitted_events",
                            "evidence_frames", "temporally_relevant_frames", "human_labeled_frames",
                            "human_relevant_frames", "unknown_video_labels", "known_video_labels"), 0)
    latencies, peaks = [], []
    matched = sorted(set(references) & set(by_hash))
    for digest in matched:
        annotation, prediction = references[digest], by_hash[digest]
        status = prediction["status"]
        counts[status] += 1
        counts["review_required"] += bool(prediction.get("review_required", False))
        label = annotation["review"]["video_verdict"]
        counts["unknown_video_labels" if label == "unknown" else "known_video_labels"] += 1
        events = [event for event in annotation["events"] if event["status"] == "supported"]
        counts["supported_events"] += len(events)
        for name, storage in (("latency_sec", latencies), ("peak_gpu_memory_mib", peaks)):
            value = prediction.get(name)
            if value is not None:
                if not _number(value) or value < 0:
                    _fail("Invalid measured runtime or memory")
                storage.append(value)
        if status != "done":
            continue
        if label != "unknown":
            positive, emitted = label == "violation", prediction["verdict"] == "violation"
            counts["tp" if positive and emitted else "fn" if positive else "fp" if emitted else "tn"] += 1
        proposals = prediction.get("proposals", [])
        checked = []
        for proposal in proposals:
            start, stop = proposal.get("start_sec"), proposal.get("end_sec")
            if not _number(start) or not _number(stop) or not 0 <= start <= stop <= annotation["source"]["duration_sec"]:
                _fail("Invalid proposed evaluation interval")
            checked.append((start, stop))
        for event in events:
            start, stop = event["start_sec"], event["end_sec"]
            def matches(interval):
                left, right = interval
                if start == stop:
                    return left <= start <= right
                overlap = max(0., min(stop, right) - max(start, left))
                union = max(stop, right) - min(start, left)
                return (overlap / union >= min_iou) if union else start == left
            counts["proposal_hits"] += any(matches(interval) for interval in checked)
        findings = prediction.get("violations", [])
        counts["emitted_events"] += len(findings)
        available = set(range(len(events)))
        relevance_labels = {}
        for event in annotation["events"]:
            for ref in event["evidence_refs"]:
                if ref["relevance"] != "unknown":
                    key = (event["type"], ref["timestamp_sec"])
                    previous = relevance_labels.setdefault(key, ref["relevance"])
                    if previous != ref["relevance"]:
                        _fail("Conflicting human frame relevance labels")
        for finding in findings:
            timestamp = finding.get("timecode_sec")
            if not _number(timestamp) or not 0 <= timestamp <= annotation["source"]["duration_sec"]:
                _fail("Invalid finding timestamp")
            candidates = [i for i in available if events[i]["type"] == finding.get("type")
                          and events[i]["start_sec"] - tolerance_sec <= timestamp <= events[i]["end_sec"] + tolerance_sec]
            assessable = any(event["status"] in {"supported", "contradicted"}
                             and event["type"] == finding.get("type")
                             and event["start_sec"] - tolerance_sec <= timestamp <= event["end_sec"] + tolerance_sec
                             for event in annotation["events"])
            counts["assessable_emitted_events" if assessable else "unassessed_emitted_events"] += 1
            if candidates:
                chosen = min(candidates, key=lambda i: abs(timestamp - (events[i]["start_sec"] + events[i]["end_sec"]) / 2))
                available.remove(chosen)
                counts["localized_events"] += 1
            for frame in finding.get("evidence_frames", []):
                stamp = frame.get("timestamp_sec")
                if not _number(stamp) or not 0 <= stamp <= annotation["source"]["duration_sec"]:
                    _fail("Invalid evidence timestamp")
                counts["evidence_frames"] += 1
                counts["temporally_relevant_frames"] += any(
                    event["type"] == finding.get("type") and event["start_sec"] <= stamp <= event["end_sec"]
                    for event in events)
                key = (finding.get("type"), stamp)
                if key in relevance_labels:
                    counts["human_labeled_frames"] += 1
                    counts["human_relevant_frames"] += relevance_labels[key] == "relevant"
    evaluated = counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"]
    return {"schema_version": SCHEMA_VERSION, "matched_sources": len(matched),
            "missing_predictions": len(set(references) - set(by_hash)),
            "unmatched_predictions": len(set(by_hash) - set(references)), "counts": counts,
            "video_precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "video_recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
            "video_specificity": _ratio(counts["tn"], counts["tn"] + counts["fp"]),
            "video_confusion_denominator": evaluated,
            "proposal_recall": _ratio(counts["proposal_hits"], counts["supported_events"]),
            "event_localization_recall": _ratio(counts["localized_events"], counts["supported_events"]),
            "event_localization_precision": _ratio(counts["localized_events"], counts["assessable_emitted_events"]),
            "evidence_temporal_relevance": _ratio(counts["temporally_relevant_frames"], counts["evidence_frames"]),
            "human_evidence_relevance": _ratio(counts["human_relevant_frames"], counts["human_labeled_frames"]),
            "review_rate": _ratio(counts["review_required"], len(matched)),
            "failure_rate": _ratio(counts["failed"], len(matched)),
            "latency_measured_sources": len(latencies), "max_latency_sec": max(latencies, default=None),
            "peak_gpu_memory_measured_sources": len(peaks), "max_peak_gpu_memory_mib": max(peaks, default=None),
            "min_proposal_iou": min_iou, "localization_tolerance_sec": tolerance_sec,
            "scope": "Human-reviewed observable labels; temporal overlap is not human evidence relevance. "
                     "Failed/pending jobs never enter binary confusion. Missing predictions are reported separately. "
                     "Event precision excludes findings without a resolved same-category reference interval."}
