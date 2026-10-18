"""Source-linked scene proposals: crop guidance, never proof of legal boundaries."""
from __future__ import annotations

import math

from worker.video import AnalysisError

ZONES = ("workstation", "customer", "counter", "storage", "window", "safe", "cash_zone", "computer")


def scene_schema(frame_count):
    index = {"type": "integer", "minimum": 0, "maximum": frame_count - 1}
    properties = {
        "zone": {"type": "string", "enum": list(ZONES)},
        "status": {"type": "string", "enum": ["supported", "unknown"]},
        "bbox": {"type": "array", "minItems": 0, "maxItems": 4,
                 "items": {"type": "number", "minimum": 0, "maximum": 1}},
        "frame_indices": {"type": "array", "maxItems": 3, "items": index},
    }
    return {"type": "array", "maxItems": len(ZONES), "items": {
        "type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}


def collect_scene(regions, frames, *, strict=True):
    if not isinstance(regions, list) or len(regions) > len(ZONES):
        raise AnalysisError("model_schema_invalid", "Invalid local scene region response")
    zones, seen, normalized = [], set(), 0
    for region in regions:
        if not isinstance(region, dict) or set(region) != {"zone", "status", "bbox", "frame_indices"}:
            raise AnalysisError("model_schema_invalid", "Invalid local scene region fields")
        name, status, box, indices = (region[k] for k in ("zone", "status", "bbox", "frame_indices"))
        if (name not in ZONES or name in seen or status not in {"supported", "unknown"}
                or not isinstance(indices, list) or len(indices) > 3
                or any(type(i) is not int or not 0 <= i < len(frames) for i in indices)
                or not isinstance(box, list)):
            raise AnalysisError("model_schema_invalid", "Invalid local scene region reference")
        if status == "supported":
            if (len(box) != 4 or not indices or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                if strict:
                    raise AnalysisError("model_schema_invalid", "A scene proposal lacks source support or valid coordinates")
                status, box, indices = "unknown", [], []
                normalized += 1
        elif box or indices:
            if strict:
                raise AnalysisError("model_schema_invalid", "Unknown scene regions must not assert geometry or evidence")
            box, indices = [], []
            normalized += 1
        seen.add(name)
        zones.append({"zone": name, "status": status, "bbox": box,
                      "refs": [{"source_index": frames[i].index, "timestamp_sec": frames[i].timestamp} for i in indices]})
    zones.extend({"zone": name, "status": "unknown", "bbox": [], "refs": []} for name in ZONES if name not in seen)
    return {"zones": zones, "method": "local_vlm_scene_proposal", "calibrated": False,
            "unusable_region_estimates": normalized,
            "semantics": "Geometry is crop guidance only. Unusable estimates remain unknown, never positive evidence. No room type, ownership, continuous visibility or legal boundary is established."}


def interaction_box(scene, timestamp=None, *, zone="counter"):
    def segment(value):
        return sum(change <= value for change in scene.get("scene_change_timestamps", []))
    return next((region["bbox"] for region in scene.get("zones", [])
                 if region.get("zone") == zone and region.get("status") == "supported"
                 and (timestamp is None or all(segment(ref["timestamp_sec"]) == segment(timestamp)
                                               for ref in region.get("refs", [])))), None)


def detail_views(cache, frames, scene, anchor, max_frames=None, *, preferred_zone="counter"):
    """Add source crops for one requested region without asserting screen state.

    Computer crops retain full-scene samples to assess workstation occupancy.
    An unknown computer box never substitutes another region as device evidence.
    """
    if preferred_zone not in ZONES:
        raise ValueError("Unknown detail-view scene region")
    box = interaction_box(scene, anchor, zone=preferred_zone)
    if box is None or not frames:
        return frames
    unique = {frame.index: frame for frame in frames}
    selected = sorted((frame for frame in unique.values()
                       if interaction_box(scene, frame.timestamp, zone=preferred_zone) == box),
                      key=lambda frame: abs(frame.timestamp - anchor))[:2]
    crops = cache.crop(selected, box, f"{preferred_zone} detail; full scene supplied separately")
    if max_frames is not None and len(frames) + len(crops) > max_frames:
        if max_frames < 4:
            raise ValueError("Detailed verification requires at least four image slots")
        crops = crops[:max_frames - min(3, len(unique))]
        # Keep full-scene endpoints and the action anchor, then fill temporal
        # gaps. Crop slots are included in the model's actual image budget.
        full_limit = max_frames - len(crops)
        selected_full = []
        for target in (frames[0].timestamp, anchor, frames[-1].timestamp):
            chosen = min(frames, key=lambda frame: abs(frame.timestamp - target))
            if chosen not in selected_full and len(selected_full) < full_limit:
                selected_full.append(chosen)
        while len(selected_full) < full_limit:
            available = [frame for frame in frames if frame not in selected_full]
            if not available:
                break
            selected_full.append(max(available, key=lambda frame: min(abs(frame.timestamp - f.timestamp) for f in selected_full)))
        frames = selected_full
    return sorted([*frames, *crops], key=lambda frame: (frame.timestamp, frame.crop_bbox is not None))
