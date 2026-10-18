"""Source-linked object boxes for visual review, separate from breach decisions."""
from __future__ import annotations

from copy import deepcopy
import math

from worker.video import AnalysisError, FrameRef


KINDS = ("person", "cash", "document", "cash_drawer", "counting_machine", "bag", "phone", "drawer", "safe", "counter", "window", "computer", "other")
_FIELDS = {"frame_index", "label", "kind", "confidence", "bbox"}


def _invalid():
    raise AnalysisError("model_schema_invalid", "A local object detection contained invalid fields or source references")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def detection_schema(frame_count: int) -> dict:
    """Bound the object annotations returned by one screening request."""
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("Object detection requires at least one source frame")
    properties = {
        "frame_index": {"type": "integer", "minimum": 0, "maximum": frame_count - 1},
        "label": {"type": "string", "minLength": 1, "maxLength": 60},
        "kind": {"type": "string", "enum": list(KINDS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                 "items": {"type": "number", "minimum": 0, "maximum": 1}},
    }
    return {"type": "array", "maxItems": 12,
            "items": {"type": "object", "properties": properties,
                      "required": list(properties), "additionalProperties": False}}


def localization_prompt() -> str:
    return (
        "\nAlso return detections: at most 12 clearly visible people or physical items. "
        "Kinds: person, cash, document, cash_drawer, counting_machine, bag, phone, drawer, safe, counter, window, computer, other. "
        "Computer means laptop or desktop monitor hardware; its box does not establish screen power or lock state. "
        "Choose one best-visible supplied frame per object; do not repeat it in multiple frames. "
        "Use frame_index from this request and a concise visible label, without identities, "
        "ownership, hidden contents or assumed roles. Box only the visible object using "
        "bbox=[x1,y1,x2,y2], normalized 0..1 from the displayed image's top-left; "
        "x1<x2 and y1<y2. Give confidence 0..1; omit objects that cannot be localized. "
        "These estimated boxes are review annotations and do not prove a breach."
    )


def _bbox(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(not _number(coordinate) or not 0 <= coordinate <= 1 for coordinate in value)):
        _invalid()
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        _invalid()
    return [float(coordinate) for coordinate in value]


def collect_detections(answer: dict, frames: list[FrameRef], window_id: int) -> list[dict]:
    """Validate annotations atomically and bind them to host-owned frame metadata.

    Older synthetic runtimes may omit ``detections``; the actual request schema
    requires it. Zero-confidence annotations retain no detection record.
    """
    if (not isinstance(answer, dict) or not isinstance(frames, list) or not frames
            or type(window_id) is not int or window_id < 0):
        _invalid()
    source_times = {}
    for frame in frames:
        if (not isinstance(frame, FrameRef) or type(frame.index) is not int or frame.index < 0
                or not _number(frame.timestamp) or frame.timestamp < 0):
            _invalid()
        if frame.index in source_times and source_times[frame.index] != frame.timestamp:
            _invalid()
        source_times[frame.index] = frame.timestamp
    supplied = answer.get("detections", [])
    if not isinstance(supplied, list) or len(supplied) > 12:
        _invalid()
    result = []
    for index, detection in enumerate(supplied):
        if not isinstance(detection, dict) or set(detection) != _FIELDS:
            _invalid()
        frame_index = detection["frame_index"]
        if type(frame_index) is not int or not 0 <= frame_index < len(frames):
            _invalid()
        label, kind, confidence = detection["label"], detection["kind"], detection["confidence"]
        if (not isinstance(label, str) or not label.strip() or len(label) > 60
                or not isinstance(kind, str) or kind not in KINDS
                or not _number(confidence) or not 0 <= confidence <= 1):
            _invalid()
        box = _bbox(detection["bbox"])
        if confidence == 0:
            continue
        frame = frames[frame_index]
        result.append({"id": f"detection-{window_id}-{index}", "label": label.strip(), "kind": kind,
                       "confidence": float(confidence), "timestamp_sec": float(frame.timestamp),
                       "source_index": frame.index, "bbox": box})
    return result


def _overlap(first: list[float], second: list[float]) -> float:
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union else 0


def deduplicate_detections(detections: list[dict]) -> list[dict]:
    """Keep the strongest near-identical box on exactly the same source frame.

    IoU >= .85 removes overlapping duplicate annotations while preserving
    nearby distinct objects. Never infer tracking across different frames.
    Input records are canonical collector output; returned records are copies.
    """
    retained = []
    for index, detection in sorted(enumerate(detections), key=lambda pair: -pair[1]["confidence"]):
        if any(previous["source_index"] == detection["source_index"]
               and previous["timestamp_sec"] == detection["timestamp_sec"]
               and previous["kind"] == detection["kind"]
               and _overlap(previous["bbox"], detection["bbox"]) >= .85
               for _, previous in retained):
            continue
        retained.append((index, detection))
    return [deepcopy(detection) for _, detection in sorted(retained, key=lambda pair: pair[0])]
