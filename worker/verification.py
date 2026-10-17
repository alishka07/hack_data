"""Conservative semantic validation beyond a model's JSON response schema."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import re


_PLACEHOLDERS = {
    "", "no", "none", "null", "unknown", "uncertain", "not visible", "not_visible",
    "n/a", "na", "not applicable", "no action", "no visible action", "not observed",
}


def normalize_verification(answer: Mapping, required_checks: Sequence[str], frame_count: int) -> dict:
    """Return an independent, fail-closed response with ``semantic_issues``.

    A response can only retain support; normalization never creates it. Check
    citations refer to this request's frame order, not source frame numbering.
    JSON shape validation alone cannot catch a supported decision whose checks
    say no, whose visibility is unusable, or whose evidence is absent.
    """
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("frame_count must be a nonnegative integer")
    required = list(dict.fromkeys(required_checks))
    if any(not isinstance(name, str) or not name for name in required):
        raise ValueError("required_checks must contain nonempty check names")
    result = deepcopy(dict(answer)) if isinstance(answer, Mapping) else {}
    issues: list[str] = []

    def issue(code: str) -> None:
        if code not in issues:
            issues.append(code)

    if not isinstance(answer, Mapping):
        issue("invalid_response")
    original_decision = result.get("decision")
    decision = (original_decision if isinstance(original_decision, str)
                and original_decision in {"supported", "rejected", "uncertain"} else "uncertain")
    if decision != original_decision:
        issue("invalid_decision")

    def references(value: object, field: str) -> tuple[list[int], bool]:
        if not isinstance(value, list):
            issue("invalid_" + field)
            return [], False
        valid = len(value) <= 3 and all(type(index) is int and 0 <= index < frame_count for index in value)
        if not valid:
            issue("invalid_" + field)
        # Keep valid citations for diagnosis; their existence cannot repair an
        # invalid original citation list or authorize a supported decision.
        return list(dict.fromkeys(index for index in value
                                  if type(index) is int and 0 <= index < frame_count))[:3], valid

    raw_checks = result.get("checks")
    raw_frames = result.get("check_frames")
    if not isinstance(raw_checks, Mapping):
        raw_checks = {}
        issue("invalid_checks")
    if not isinstance(raw_frames, Mapping):
        raw_frames = {}
        issue("invalid_check_frames")
    contradiction = any(value == "no" for value in raw_checks.values())
    checks, check_frames = {}, {}
    support_blocked = False
    for name in required:
        status = raw_checks.get(name)
        if not isinstance(status, str) or status not in {"yes", "no", "unknown", "not_visible"}:
            issue("invalid_check:" + name)
            status = "unknown"
        refs, valid = references(raw_frames.get(name), "check_frames:" + name)
        if status in {"unknown", "not_visible"}:
            # A not-visible predicate has no affirmative source evidence.
            if refs:
                issue("unresolved_check_has_citations:" + name)
            refs = []
            support_blocked = True
            issue("unresolved_check:" + name)
        elif status == "yes" and (not valid or not refs):
            status, refs = "unknown", []
            support_blocked = True
            issue("unsupported_check:" + name)
        checks[name], check_frames[name] = status, refs
    result["checks"], result["check_frames"] = checks, check_frames

    visibility = result.get("visibility")
    if not isinstance(visibility, str) or visibility not in {"clear", "partial", "unusable"}:
        visibility = "unusable"
        issue("invalid_visibility")
    result["visibility"] = visibility
    if visibility != "clear":
        support_blocked = True
        issue("limited_visibility")

    evidence, valid_evidence = references(result.get("evidence_indices"), "evidence_indices")
    result["evidence_indices"] = evidence
    if not valid_evidence or not evidence:
        support_blocked = True
        issue("missing_or_invalid_evidence")

    confidence = result.get("confidence")
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1 or not math.isfinite(confidence):
        confidence = 0.
        issue("invalid_confidence")
    result["confidence"] = float(confidence)
    if confidence <= 0:
        support_blocked = True
        issue("nonpositive_confidence")

    description = result.get("observed_action")
    if not isinstance(description, str):
        description = ""
    description = description.strip()
    canonical = re.sub(r"\s+", " ", description.casefold()).strip(" .!?;:")
    if canonical in _PLACEHOLDERS:
        support_blocked = True
        issue("missing_observed_action")
        description = ""
    result["observed_action"] = description

    # Context responses optionally carry independently reviewed exceptions.
    if "exception_checks" in result or "exception_frames" in result:
        exceptions, exception_refs = result.get("exception_checks"), result.get("exception_frames")
        if not isinstance(exceptions, Mapping) or not isinstance(exception_refs, Mapping):
            result["exception_checks"], result["exception_frames"] = {}, {}
            support_blocked = True
            issue("invalid_exceptions")
        else:
            normalized_exceptions, normalized_refs = {}, {}
            if set(exceptions) != set(exception_refs):
                support_blocked = True
                issue("invalid_exception_fields")
            for name, status in exceptions.items():
                refs, valid = references(exception_refs.get(name), "exception_frames:" + str(name))
                if status == "applies":
                    contradiction = True
                    issue("exception_applies:" + str(name))
                elif status != "excluded" or not valid or not refs:
                    status, refs = "unknown", []
                    support_blocked = True
                    issue("unresolved_exception:" + str(name))
                normalized_exceptions[name], normalized_refs[name] = status, refs
            result["exception_checks"], result["exception_frames"] = normalized_exceptions, normalized_refs

    if contradiction:
        decision = "rejected"
        issue("explicit_contradiction")
    elif decision == "supported" and support_blocked:
        decision = "uncertain"
    if decision != original_decision:
        issue("decision_normalized")
    result["decision"], result["semantic_issues"] = decision, issues
    return result
