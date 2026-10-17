"""Conservative filters for object *candidates*, not verified objects or violations.

Detection boxes use image pixels. Optional manually reviewed region boxes use
normalized coordinates in [0, 1]. An ``exclude`` region's label is the class to
exclude there (``*`` means all classes), not the identity of the background.
An ``equipment`` region's label describes equipment verified by a person.
Regions never create a detection and never relabel a conflicting model result.

Scores from different prompts are not treated as comparable probabilities.
Only same-class duplicate removal uses score ordering.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping


_ALIASES = {
    "banknote_counter": (
        "banknote counter", "banknote counting machine", "banking counting machine",
        "cash counting machine", "money counting machine", "currency counter",
        "currency counting machine", "bill counter",
    ),
    "safe": ("safe", "metal safe", "security safe"),
    "plastic_bag": ("plastic bag", "transparent plastic bag", "clear plastic bag", "plastic bags"),
    # Generic 'bag' is intentionally not promoted to 'plastic_bag'.
    "bag": ("bag", "bags"),
    "electrical_outlet": ("electrical outlet", "power outlet", "wall socket", "electrical socket", "wall outlet"),
    "power_strip": ("power strip", "extension socket", "extension cord"),
    "computer_monitor": ("computer monitor", "monitor", "desktop monitor"),
    "laptop": ("laptop", "laptop computer"),
    "desktop_computer": ("desktop computer", "desktop computer tower", "computer tower", "system unit"),
    "cash": ("cash", "banknote", "banknotes", "money", "banknotes money", "money banknotes", "stack of money", "stack of banknotes", "paper money"),
    "food": ("food", "chocolate", "chocolate bar", "wrapped chocolate", "wrapped chocolate bar", "candy", "candy bar", "candy wrapper", "snack", "snack bar", "food wrapper"),
    "drink": ("drink", "cup", "coffee", "coffee cup", "paper cup", "disposable cup", "mug", "tea cup", "cup of coffee"),
    "smartphone": ("smartphone", "mobile phone", "cell phone", "cellphone"),
    "usb_drive": ("usb flash drive", "usb drive", "flash drive", "usb stick"),
    "notebook": ("notebook", "notepad", "paper notebook"),
    "document": ("document", "documents", "paper document", "paper documents", "sheet of paper"),
    "identity_card": ("identity card", "id card", "identification card"),
    "keyboard": ("keyboard", "computer keyboard"),
    "mouse": ("mouse", "computer mouse"),
    "scanner": ("scanner", "document scanner", "identity card scanner", "id card scanner"),
    "printer": ("printer", "office printer"),
    "pos_terminal": ("pos terminal", "payment terminal", "card terminal"),
    "biometric_camera": ("biometric camera",),
    "camera": ("camera", "webcam"),
    "door": ("door", "glass door"),
    "person": ("person", "people", "man", "woman"),
    "chair": ("chair", "office chair"),
    "work_table": ("desk", "office desk", "table", "work table"),
}


def _clean_label(value):
    if not isinstance(value, str):
        return ""
    value = re.sub(r"[_-]+", " ", value.strip().lower())
    value = re.sub(r"\s+", " ", value).strip(" .,:;!?")
    return re.sub(r"^(?:a|an|the)\s+", "", value)


_LABEL_MAP = {_clean_label(alias): canonical
              for canonical, aliases in _ALIASES.items()
              for alias in (canonical, *aliases)}


def canonical_label(value):
    """Return an exact known class, or None for fragment/merged/unknown labels."""
    return _LABEL_MAP.get(_clean_label(value))


def _number(value):
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _box(value):
    try:
        if len(value) != 4:
            return None
        result = [_number(x) for x in value]
    except (TypeError, AttributeError):
        return None
    if any(x is None for x in result):
        return None
    return result if result[2] > result[0] and result[3] > result[1] else None


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _iou(a, b):
    intersection = _intersection(a, b)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _same_object_hypothesis(a, b):
    """Similar boxes or a moderately larger box enclosing the same object.

    The containment branch catches a cabinet's upper half versus its full box.
    Its area-ratio limit prevents a tiny item inside an entire desk/room box from
    being treated as the same object merely because it is contained.
    """
    if _iou(a, b) >= 0.55:
        return True
    smaller, larger = sorted((_area(a), _area(b)))
    return smaller > 0 and larger / smaller <= 2.5 and _intersection(a, b) / smaller >= 0.90


_RIGID = frozenset({"banknote_counter", "safe", "electrical_outlet", "power_strip",
                    "desktop_computer", "computer_monitor", "laptop", "scanner",
                    "printer", "pos_terminal", "biometric_camera", "camera"})
_BAGS = frozenset({"bag", "plastic_bag"})
_PAPER = frozenset({"cash", "notebook", "document", "identity_card"})
_FOOD_DRINK = frozenset({"food", "drink"})


def _incompatible(a, b):
    if a == b or {a, b} == {"bag", "plastic_bag"}:
        return False
    if (a in _FOOD_DRINK and b in _PAPER | _RIGID | {"smartphone", "usb_drive"}) or (b in _FOOD_DRINK and a in _PAPER | _RIGID | {"smartphone", "usb_drive"}):
        return True
    if a in _FOOD_DRINK and b in _FOOD_DRINK:
        return True
    if a in _RIGID and b in _RIGID:
        return True
    if (a in _RIGID and b in _BAGS) or (b in _RIGID and a in _BAGS):
        return True
    if (a == "chair" and b in _BAGS) or (b == "chair" and a in _BAGS):
        return True
    if a in _PAPER and b in _PAPER:
        # A broad 'document' label and a specific identity card are compatible.
        return {a, b} != {"document", "identity_card"}
    if a in {"smartphone", "usb_drive"} and b in _RIGID | _PAPER:
        return True
    if b in {"smartphone", "usb_drive"} and a in _RIGID | _PAPER:
        return True
    return False


def _regions(regions, width, height):
    result = []
    for index, item in enumerate(regions or []):
        if not isinstance(item, Mapping):
            raise ValueError(f"Region {index}: expected an object")
        if item.get("source") != "manual":
            raise ValueError(f"Region {index}: only source='manual' is supported")
        if item.get("kind") not in {"exclude", "equipment"}:
            raise ValueError(f"Region {index}: kind must be exclude or equipment")
        box = _box(item.get("box"))
        if box is None or any(x < 0 or x > 1 for x in box):
            raise ValueError(f"Region {index}: box must be normalized xyxy in [0, 1]")
        label = "*" if item.get("label") == "*" else canonical_label(item.get("label"))
        if label is None or (item["kind"] == "equipment" and label == "*"):
            raise ValueError(f"Region {index}: unsupported label")
        result.append({"index": index, "kind": item["kind"], "label": label,
                       "box": [box[0] * width, box[1] * height, box[2] * width, box[3] * height]})
    return result


def _reason(item, reason):
    if reason not in item["reasons"]:
        item["reasons"].append(reason)


def filter_candidates(detections, image_size, regions=None):
    """Return candidates, review and suppressed rows without asserting violations.

    Thresholding should happen before this function. There is intentionally no
    single confidence threshold that purports to distinguish a counter and bag.
    Same-class NMS uses IoU >= 0.60. Conflicting classes both go to review when
    IoU >= 0.55, or >= 90% of the smaller box is inside the larger box and the
    area ratio is <= 2.5. The higher model score never resolves the conflict.

    Manual exclusions require >= 80% of a detection inside the region. Equipment
    hints require >= 80% detection coverage AND IoU >= 0.15, so a small partial
    intersection or a huge enclosing region does not erase unrelated detections.
    """
    try:
        width, height = [_number(x) for x in image_size]
    except (TypeError, ValueError):
        raise ValueError("image_size must contain (width, height)") from None
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("image_size must contain finite positive dimensions")
    checked_regions = _regions(regions, width, height)
    suppressed = []
    valid = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            suppressed.append({"detection_index": index, "reasons": ["invalid_detection"]})
            continue
        item = dict(detection)
        item.update(detection_index=index, raw_label=detection.get("label"), reasons=[])
        item["interpretation"] = "unverified_object_candidate"
        label = canonical_label(item.get("label"))
        score = _number(item.get("score"))
        box = _box(item.get("box"))
        if label is None:
            _reason(item, "unsupported_or_fragment_label")
        if score is None or not 0 <= score <= 1:
            _reason(item, "invalid_score")
            # Invalid NaN/Inf must not leak into non-standard JSON output.
            item["score"] = None
        if box is None:
            _reason(item, "invalid_box")
            item["box"] = None
        else:
            clipped = [max(0.0, min(width, box[0])), max(0.0, min(height, box[1])),
                       max(0.0, min(width, box[2])), max(0.0, min(height, box[3]))]
            if _area(clipped) <= 0:
                _reason(item, "invalid_box")
            item["box"] = clipped
        if item["reasons"]:
            suppressed.append(item)
            continue
        item.update(label=label, score=score)
        valid.append(item)

    # Score ordering is limited to duplicates of the same semantic class.
    kept = []
    for item in sorted(valid, key=lambda x: (-x["score"], x["detection_index"])):
        duplicate = next((other for other in kept
                          if other["label"] == item["label"] and _iou(other["box"], item["box"]) >= 0.60), None)
        if duplicate is not None:
            _reason(item, "duplicate_same_class")
            item["duplicate_of"] = duplicate["detection_index"]
            suppressed.append(item)
        else:
            kept.append(item)

    remaining = []
    for item in kept:
        excluded = False
        for region in checked_regions:
            coverage = _intersection(item["box"], region["box"]) / _area(item["box"])
            if coverage < 0.80:
                continue
            if region["kind"] == "exclude" and region["label"] in {"*", item["label"]}:
                _reason(item, "manual_exclusion_region")
                item["region_index"] = region["index"]
                excluded = True
                break
            if region["kind"] == "equipment" and _iou(item["box"], region["box"]) >= 0.15:
                if _incompatible(item["label"], region["label"]):
                    _reason(item, "overlaps_reviewed_equipment")
                    item.setdefault("region_hints", []).append({"region_index": region["index"],
                                                                "label": region["label"]})
                elif item["label"] == region["label"]:
                    item.setdefault("region_hints", []).append({"region_index": region["index"],
                                                                "label": region["label"]})
        (suppressed if excluded else remaining).append(item)

    for index, first in enumerate(remaining):
        for second in remaining[index + 1:]:
            cash_food_overlap = False
            if 'cash' in {first['label'], second['label']} and {first['label'], second['label']} & _FOOD_DRINK:
                cash = first if first['label'] == 'cash' else second
                other = second if first['label'] == 'cash' else first
                cash_food_overlap = (_iou(cash['box'], other['box']) >= .3
                                     or _intersection(cash['box'], other['box']) / _area(cash['box']) >= .5)
            if _incompatible(first["label"], second["label"]) and (_same_object_hypothesis(first["box"], second["box"]) or cash_food_overlap):
                for item, other in ((first, second), (second, first)):
                    _reason(item, "conflicting_class_predictions")
                    item.setdefault("conflicts", []).append({"detection_index": other["detection_index"],
                                                             "label": other["label"]})

    candidates = []
    review = []
    for item in remaining:
        if item["reasons"]:
            item["interpretation"] = "ambiguous_object_requires_review"
            review.append(item)
        else:
            _reason(item, "model_candidate")
            candidates.append(item)
    for item in suppressed:
        item["interpretation"] = "suppressed_candidate"
    return {"candidates": sorted(candidates, key=lambda x: x["detection_index"]),
            "review": sorted(review, key=lambda x: x["detection_index"]),
            "suppressed": sorted(suppressed, key=lambda x: x["detection_index"])}
