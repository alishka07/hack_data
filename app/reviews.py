"""Human annotations are separate from immutable model decisions and jury output."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from uuid import UUID


def _number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def duration_sec(details: dict) -> float | None:
    coverage = details.get("coverage") if isinstance(details, dict) else None
    value = coverage.get("duration_sec") if isinstance(coverage, dict) else None
    return float(value) if _number(value) and value > 0 else None


def default_review_state() -> dict:
    return {"revision": 0, "reviewer": "", "coverage_complete": False,
            "camera_id": "", "session_id": "", "source_group": "",
            "decisions": [], "missed_events": []}


def review_cases(details: dict, result: dict | None = None) -> list[dict]:
    """IDs use original case identity only; confidence/display text can evolve safely."""
    details = details if isinstance(details, dict) else {}
    decision = details.get("case_decision")
    raw = decision.get("cases") if isinstance(decision, dict) else None
    if not isinstance(raw, list):
        events = (result or {}).get("violations", [])
        raw = [{"type_id": item.get("type", "unknown"), "status": "confirmed",
                "timecode_sec": item.get("timecode_sec"), "observed_action": item.get("description", ""),
                "checks": [], "reasons": []} for item in events if isinstance(item, dict)]
        queue = details.get("review_queue", [])
        if isinstance(queue, list):
            for item in queue:
                if (not isinstance(item, dict) or item.get("requires_review") is False
                        or item.get("status") in {"rejected", "confirmed", "diagnostic"}
                        or set(item.get("reason_codes", [])).intersection({"action_rejected", "action_contradiction",
                                                                           "legal_context_rejected", "context_contradiction"})):
                    continue
                raw.append({**item, "status": "review_required", "requires_review": True})
    output = []
    duplicates = Counter()
    for item in raw:
        if not isinstance(item, dict):
            continue
        time = item.get("timecode_sec")
        key = json.dumps([item.get("type_id", "unknown"), item.get("episode_id", ""),
                          float(time) if _number(time) and time >= 0 else None],
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        ordinal = duplicates[key]
        duplicates[key] += 1
        case_id = "case_" + hashlib.sha256(f"{key}\n{ordinal}".encode()).hexdigest()[:24]
        output.append({**deepcopy(item), "case_id": case_id})
    return output


class InvalidReview(ValueError):
    pass


def validate_review(value, cases: list[dict], duration: float | None, allowed_types: set[str]) -> dict:
    """Strict bounded input: booleans and numeric strings are never timestamps."""
    if not isinstance(value, dict):
        raise InvalidReview("Review must be an object")
    keys = {"revision", "reviewer", "coverage_complete", "camera_id", "session_id", "source_group",
            "decisions", "missed_events", "updated_at"}
    if set(value) - keys:
        raise InvalidReview("Unknown review field")
    revision = value.get("revision")
    if type(revision) is not int or not 0 <= revision <= 2**53 - 1:
        raise InvalidReview("A nonnegative integer revision is required")
    result = default_review_state()
    result["revision"] = revision

    def text(record, key, maximum, *, required=False):
        item = record.get(key, "")
        if not isinstance(item, str) or len(item) > maximum or any(ord(char) < 32 and char not in "\n\t" for char in item):
            raise InvalidReview(f"Invalid {key}")
        if required and not item.strip():
            raise InvalidReview(f"{key} is required")
        return item.strip()

    for key in ("reviewer", "camera_id", "session_id", "source_group"):
        result[key] = text(value, key, 120 if key != "source_group" else 200)
    coverage = value.get("coverage_complete", False)
    if type(coverage) is not bool:
        raise InvalidReview("coverage_complete must be true or false")
    result["coverage_complete"] = coverage
    if coverage and duration is None:
        raise InvalidReview("Video duration is unavailable; full-video coverage cannot be attested")

    def interval(item, required=False):
        start, end = item.get("start_sec"), item.get("end_sec")
        if start is None and end is None and not required:
            return None, None
        if not _number(start) or not _number(end) or start < 0 or end < start:
            raise InvalidReview("Both interval bounds must be finite numbers with 0 <= start <= end")
        if duration is None:
            raise InvalidReview("Video duration is unavailable; interval correction cannot be validated")
        if end > duration:
            raise InvalidReview("The interval exceeds the video duration")
        return float(start), float(end)

    decisions, missed = value.get("decisions", []), value.get("missed_events", [])
    if not isinstance(decisions, list) or len(decisions) > min(5000, len(cases)):
        raise InvalidReview("Too many or invalid decisions")
    if not isinstance(missed, list) or len(missed) > 1000:
        raise InvalidReview("Too many or invalid missed events")
    known_ids = {case["case_id"] for case in cases}
    seen = set()
    for item in decisions:
        if not isinstance(item, dict) or set(item) - {"case_id", "decision", "comment", "start_sec", "end_sec"}:
            raise InvalidReview("Invalid decision fields")
        identifier = item.get("case_id")
        if not isinstance(identifier, str) or identifier not in known_ids or identifier in seen:
            raise InvalidReview("A decision references an unknown or duplicate case")
        seen.add(identifier)
        decision = item.get("decision")
        if not isinstance(decision, str) or decision not in {"confirmed", "dismissed", "uncertain"}:
            raise InvalidReview("Invalid human decision")
        start, end = interval(item)
        result["decisions"].append({"case_id": identifier, "decision": decision,
                                    "comment": text(item, "comment", 2000), "start_sec": start, "end_sec": end})
    seen = set()
    for item in missed:
        if not isinstance(item, dict) or set(item) - {"event_id", "type_id", "start_sec", "end_sec", "comment"}:
            raise InvalidReview("Invalid missed event fields")
        identifier = item.get("event_id")
        try:
            valid_id = isinstance(identifier, str) and str(UUID(identifier)) == identifier
        except ValueError:
            valid_id = False
        if not valid_id or identifier in seen:
            raise InvalidReview("Missed events require distinct UUID event_id values")
        seen.add(identifier)
        kind = item.get("type_id")
        if not isinstance(kind, str) or kind not in allowed_types:
            raise InvalidReview("Unknown missed event type")
        start, end = interval(item, required=True)
        result["missed_events"].append({"event_id": identifier, "type_id": kind, "start_sec": start,
                                       "end_sec": end, "comment": text(item, "comment", 2000)})
    if (decisions or missed or coverage) and not result["reviewer"]:
        raise InvalidReview("reviewer is required for human annotations")
    return result
