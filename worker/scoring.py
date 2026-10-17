"""Source-grounded continuous review scores, separate from legal verdicts.

Event items start at 0 and rise with model confidence in a visible occurrence.
Required procedures start at 100 and fall with model confidence in visible
correct completion. These model estimates are uncalibrated; the initial values
are requested baselines rather than measured probabilities. Unknown legal
applicability may coexist with a visible action estimate. Unlike quantities are
deliberately never averaged here.

Helpers are pure: they return a new assessment and never mutate the input.
Within one assessment episode, positive event/completion evidence is monotonic;
later uncertain chunks cannot erase it. Create fresh assessments for each new
transaction/episode. An evidenced exemption stays exempt until that reset.
The caller supplies model confidence, applicability and source evidence. This
module has no access to ground-truth labels and cannot establish legal validity.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import re


KINDS = {"event_occurrence": 0, "required_procedure": 100}
APPLICABILITY = {"unknown", "applicable", "not_applicable"}
ASSESSMENTS = {"unassessed", "observed", "completion_evidence", "completed", "unresolved"}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\.jpe?g\Z", re.I)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean support decision, not a score")


def _confidence(value):
    if not _number(value) or not 0 <= value <= 1:
        raise ValueError("Model confidence must be a finite number between 0 and 1")
    return float(value)


def _identifier(value):
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _initial_score(kind):
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("Unknown catalog scoring kind")
    return KINDS[kind]


def create_initial_assessments(catalog):
    """Accept a catalog {items:[...]} or a sequence of items with id/label/kind.

    Canonical catalog IDs use ``id``; ``item_id`` is accepted for compatibility,
    and both must agree when supplied together. All items start unassessed,
    applicability unknown and confirmed_finding false, even if out of scope.
    """
    items = catalog.get("items") if isinstance(catalog, Mapping) else catalog
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise ValueError("Catalog items must be a sequence")
    assessments, seen = [], set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("A catalog item must be an object")
        identifier = item.get("id", item.get("item_id"))
        if not _identifier(identifier) or identifier in seen:
            raise ValueError("Catalog IDs must be nonempty and unique")
        if "id" in item and "item_id" in item and item["id"] != item["item_id"]:
            raise ValueError("Catalog ID aliases conflict")
        label, kind = item.get("label"), item.get("scoring_kind")
        if not isinstance(label, str) or not label.strip() or len(label) > 240:
            raise ValueError("Catalog item label must be nonempty")
        initial = _initial_score(kind)
        if type(item.get("initial_score")) is not int or item["initial_score"] != initial:
            raise ValueError("Catalog initial score conflicts with its scoring kind")
        seen.add(identifier)
        assessments.append({"item_id": identifier, "label": label, "scoring_kind": kind,
                            "initial_score": initial, "score": float(initial),
                            "model_confidence": None, "calibrated": False,
                            "applicability": "unknown", "assessment": "unassessed",
                            "evidence_refs": [], "source_duration_sec": None,
                            "confirmed_finding": False})
    return assessments


def _assessment(value, kind=None):
    if not isinstance(value, Mapping):
        raise ValueError("Assessment must be an object")
    initial = _initial_score(value.get("scoring_kind"))
    if kind is not None and value["scoring_kind"] != kind:
        raise ValueError("Assessment scoring kind does not match this operation")
    if not _identifier(value.get("item_id")):
        raise ValueError("Assessment item ID is invalid")
    if type(value.get("initial_score")) is not int or value["initial_score"] != initial:
        raise ValueError("Assessment initial score conflicts with its scoring kind")
    if value.get("score") is not None and (not _number(value.get("score")) or not 0 <= value["score"] <= 100):
        raise ValueError("Assessment score must be finite, between 0 and 100, or exempt null")
    if (not isinstance(value.get("applicability"), str) or value["applicability"] not in APPLICABILITY
            or not isinstance(value.get("assessment"), str) or value["assessment"] not in ASSESSMENTS):
        raise ValueError("Assessment state is invalid")
    _bool(value.get("confirmed_finding"), "confirmed_finding")
    if (value["score"] is None) != (value["applicability"] == "not_applicable"):
        raise ValueError("Only not-applicable assessments have an exempt null score")
    positive = value["score"] is not None and (value["score"] > 0 if value["scoring_kind"] == "event_occurrence" else value["score"] < 100)
    if positive:
        expected = "observed" if value["scoring_kind"] == "event_occurrence" else "completed" if value["score"] == 0 else "completion_evidence"
        if value["assessment"] != expected or not value.get("evidence_refs"):
            raise ValueError("An updated estimate needs matching state and positive source evidence")
    if value.get("model_confidence") is not None:
        confidence = _confidence(value["model_confidence"])
        expected_score = round(100 * (confidence if value["scoring_kind"] == "event_occurrence" else 1 - confidence), 6)
        if value["score"] is not None and value["score"] != expected_score:
            raise ValueError("Stored confidence conflicts with the assessment score")
    if value.get("calibrated", False) is not False:
        raise ValueError("These confidence estimates are uncalibrated")
    if value["confirmed_finding"] and not (value["scoring_kind"] == "event_occurrence" and positive and value["applicability"] == "applicable"):
        raise ValueError("Only a supported event may carry separate legal confirmation")
    result = deepcopy(dict(value))
    # Older persisted endpoint states remain readable without inventing a
    # calibration claim. Their endpoint score determines their prior support.
    if "model_confidence" not in result:
        result["model_confidence"] = (result["score"] / 100 if result["scoring_kind"] == "event_occurrence" else 1 - result["score"] / 100) if positive else None
    result["calibrated"] = False
    return result


def _references(evidence_refs, duration_sec, *, max_refs=32):
    if duration_sec is not None and (not _number(duration_sec) or duration_sec <= 0):
        raise ValueError("Source duration must be finite and positive")
    if not isinstance(evidence_refs, Sequence) or isinstance(evidence_refs, (str, bytes, bytearray)) or len(evidence_refs) > max_refs:
        raise ValueError(f"Evidence references must be a sequence of at most {max_refs} source frames")
    if evidence_refs and duration_sec is None:
        raise ValueError("Source duration is required to bound evidence timestamps")
    identities, seen, result = {}, set(), []
    for reference in evidence_refs:
        if not isinstance(reference, Mapping):
            raise ValueError("Each evidence reference must be an object")
        timestamp = reference.get("timestamp_sec")
        if not _number(timestamp) or timestamp < 0 or timestamp > duration_sec:
            raise ValueError("Evidence timestamp is outside the source video")
        canonical = {"timestamp_sec": float(timestamp)}
        for field in ("source_index", "frame_id", "filename"):
            if field not in reference:
                continue
            identifier = reference[field]
            if field == "source_index":
                valid = type(identifier) is int and identifier >= 0
            elif field == "filename":
                valid = isinstance(identifier, str) and _FILENAME.fullmatch(identifier) is not None
            else:
                valid = _identifier(identifier)
            if not valid:
                raise ValueError("Evidence frame identity is invalid")
            identity = (field, identifier)
            if identity in identities and identities[identity] != timestamp:
                raise ValueError("One source frame cannot have conflicting timestamps")
            identities[identity] = timestamp
            canonical[field] = identifier
        if len(canonical) == 1:
            raise ValueError("Evidence needs a source_index, frame_id or JPEG filename")
        key = tuple(sorted(canonical.items()))
        if key not in seen:
            seen.add(key)
            result.append(canonical)
    return sorted(result, key=lambda reference: reference["timestamp_sec"])


def _applicability(value):
    # Exemptions need their own positively evidenced transition.
    if not isinstance(value, str) or value not in {"unknown", "applicable"}:
        raise ValueError("Use mark_not_applicable for an evidenced exemption")


def _evidence_update(result, evidence_refs, duration_sec):
    """Bind an assessment to one source duration and revalidate prior evidence."""
    previous_duration = result.get("source_duration_sec")
    if previous_duration is not None and (not _number(previous_duration) or previous_duration <= 0):
        raise ValueError("Stored source duration is invalid")
    if duration_sec is not None and (not _number(duration_sec) or duration_sec <= 0):
        raise ValueError("Source duration must be finite and positive")
    if previous_duration is not None and duration_sec is not None and duration_sec != previous_duration:
        raise ValueError("Source duration changed within one assessment episode")
    duration = previous_duration if duration_sec is None else duration_sec
    previous_refs = _references(result.get("evidence_refs", []), duration)
    new_refs = _references(evidence_refs, duration)
    _references(previous_refs + new_refs, duration, max_refs=64)
    result["source_duration_sec"] = float(duration) if duration is not None else None
    return previous_refs, new_refs, duration


def _merge_evidence(previous, new, duration, *, prefer_new=False):
    """Validate identities before retaining a bounded, deterministic history.

    When a stronger update arrives, its references must survive truncation;
    otherwise retained references already support the equally strong estimate.
    """
    combined = _references(previous + new, duration, max_refs=64)
    preferred = new if prefer_new else previous
    keys = {tuple(sorted(reference.items())) for reference in preferred}
    retained = preferred + [reference for reference in combined
                             if tuple(sorted(reference.items())) not in keys]
    return sorted(retained[:32], key=lambda reference: reference["timestamp_sec"])


def update_event_confidence(assessment, *, confidence: float, observable: bool,
                            applicability="unknown", evidence_refs=(), duration_sec=None,
                            confirmed_finding: bool = False):
    """Raise occurrence estimate to max(previous, 100 * confidence).

    A visible, source-grounded action can update the estimate with unknown legal
    applicability. ``confirmed_finding`` remains a separately supplied legal
    decision and requires applicable positive evidence, regardless of score.
    """
    result = _assessment(assessment, "event_occurrence")
    confidence = _confidence(confidence)
    _bool(observable, "observable")
    _bool(confirmed_finding, "confirmed_finding")
    _applicability(applicability)
    previous_refs, references, duration = _evidence_update(result, evidence_refs, duration_sec)
    if result["applicability"] == "not_applicable":
        if confirmed_finding:
            raise ValueError("An exempt episode cannot become a legal finding without a new assessment")
        return result
    observed = confidence > 0 and observable and bool(references)
    previously_observed = result["score"] > 0
    effective_applicability = "applicable" if "applicable" in (result["applicability"], applicability) else "unknown"
    if confirmed_finding and not ((observed or previously_observed) and effective_applicability == "applicable"):
        raise ValueError("A legal finding requires a supported, applicable visible event with source evidence")
    prior_confidence = result.get("model_confidence") or 0.0
    # Lower-confidence correlated observations add no confidence or supporting
    # references. Their individual observations belong in the caller's ledger.
    references = (_merge_evidence(previous_refs, references, duration,
                                   prefer_new=confidence > prior_confidence)
                  if observed and confidence >= prior_confidence else previous_refs)
    effective_observed = observed or previously_observed
    best_confidence = max(prior_confidence, confidence) if observed else prior_confidence
    result.update(score=round(100 * best_confidence, 6),
                  model_confidence=best_confidence if effective_observed else None,
                  applicability=effective_applicability,
                  assessment="observed" if effective_observed else "unresolved",
                  evidence_refs=references,
                  confirmed_finding=result["confirmed_finding"] or confirmed_finding)
    return result


def update_procedure_confidence(assessment, *, confidence: float, observable: bool,
                                applicability="unknown", evidence_refs=(), duration_sec=None):
    """Lower outstanding-procedure score to min(previous, 100*(1-confidence)).

    Confidence concerns correct completion of this whole procedure, rather than
    confidence in any incidental action. The caller must establish that scope.
    An outstanding score never becomes a confirmed omission on its own.
    """
    result = _assessment(assessment, "required_procedure")
    confidence = _confidence(confidence)
    _bool(observable, "observable")
    _applicability(applicability)
    previous_refs, references, duration = _evidence_update(result, evidence_refs, duration_sec)
    if result["applicability"] == "not_applicable":
        return result
    observed = confidence > 0 and observable and bool(references)
    previously_supported = result["score"] < 100
    prior_confidence = result.get("model_confidence") or 0.0
    references = (_merge_evidence(previous_refs, references, duration,
                                   prefer_new=confidence > prior_confidence)
                  if observed and confidence >= prior_confidence else previous_refs)
    effective_supported = observed or previously_supported
    best_confidence = max(prior_confidence, confidence) if observed else prior_confidence
    score = round(100 * (1 - best_confidence), 6)
    result.update(score=score,
                  model_confidence=best_confidence if effective_supported else None,
                  applicability="applicable" if "applicable" in (result["applicability"], applicability) else "unknown",
                  assessment=("completed" if score == 0 else "completion_evidence") if effective_supported else "unresolved",
                  evidence_refs=references, confirmed_finding=False)
    return result


def mark_event(assessment, *, supported: bool, observable: bool,
               applicability="unknown", evidence_refs=(), duration_sec=None,
               confirmed_finding: bool = False):
    """Compatibility wrapper for a binary occurrence support decision."""
    _bool(supported, "supported")
    return update_event_confidence(assessment, confidence=1.0 if supported else 0.0,
                                  observable=observable, applicability=applicability,
                                  evidence_refs=evidence_refs, duration_sec=duration_sec,
                                  confirmed_finding=confirmed_finding)


def mark_procedure(assessment, *, completed: bool, observable: bool,
                   applicability="unknown", evidence_refs=(), duration_sec=None):
    """Compatibility wrapper for a binary correct-completion decision."""
    _bool(completed, "completed")
    return update_procedure_confidence(assessment, confidence=1.0 if completed else 0.0,
                                      observable=observable, applicability=applicability,
                                      evidence_refs=evidence_refs, duration_sec=duration_sec)


def mark_not_applicable(assessment, *, evidence_refs, duration_sec=None):
    """Record a positively evidenced exemption; retain initial_score for clarity."""
    result = _assessment(assessment)
    previous_refs, references, duration = _evidence_update(result, evidence_refs, duration_sec)
    if result["applicability"] == "not_applicable":
        if references:
            result["evidence_refs"] = _merge_evidence(previous_refs, references, duration)
        return result
    if not references:
        raise ValueError("Not-applicable status requires positive source evidence")
    result.update(score=None, applicability="not_applicable", assessment="observed",
                  evidence_refs=references, confirmed_finding=False)
    return result
