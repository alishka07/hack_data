"""Explain the host's case decision without changing the public jury verdict."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


_REASONS = {
    "visible_scenario_supported": "The supplied frames support this physical scenario; policy applicability and a legal violation still require review.",
    "below_calibrated_threshold": "The verifier score is below this rule's empirically validated decision threshold.",
    "supported_finding": "The host evidence and rule checks support a finding; human review is still required.",
    "unsupported_type": "The observed event is outside the automatic rule set.",
    "review_only_rule": "This rule requires human review and cannot emit an automatic finding.",
    "omission_unverifiable_from_sparse_samples": "Sparse frames cannot establish that a required procedure never happened.",
    "insufficient_before_after": "The supplied evidence does not establish context both before and after the event.",
    "source_timestamp_gap": "A material gap in source timestamps limits the event context.",
    "action_rejected": "The action verifier rejected this proposed event.",
    "legal_context_rejected": "The context verifier rejected the proposed rule application.",
    "action_contradiction": "An action check contains explicit contrary evidence.",
    "context_contradiction": "A context check contains explicit contrary evidence.",
    "exceptions_unresolved": "One or more reviewed exceptions could not be excluded with source evidence.",
    "exception_applies": "A reviewed exception applies; this candidate cannot produce an automatic finding.",
    "scene_change_context": "A possible scene cut or exposure change interrupts the transaction context.",
    "action_uncertain": "The visible action remains uncertain.",
    "legal_context_uncertain": "The rule's applicability or transaction context remains uncertain.",
    "action_supported": "The model reported action support, but required source citations, confidence, visibility, or other host evidence checks did not pass.",
    "legal_context_supported": "The model reported context support, but required source citations, confidence, visibility, or other host evidence checks did not pass.",
    "action_checks_unresolved": "One or more visible-action checks remain unknown.",
    "context_checks_unresolved": "One or more applicability or context checks remain unknown.",
    "action_check_failed": "At least one required visible-action check is no.",
    "context_check_failed": "At least one required applicability or context check is no.",
    "action_visibility_limited": "The action view has limited visibility.",
    "context_visibility_limited": "The context view has limited visibility.",
    "context_not_verified": "Transaction and rule applicability checks have not been verified.",
    "candidate_not_confirmed": "The candidate did not pass the complete host evidence gate.",
    "catalog_unresolved": "A catalog assessment remains unresolved; its initial score does not establish a violation.",
    "catalog_event_unconfirmed": "Source-linked model support for an event has not passed the complete legal evidence gate.",
    "catalog_procedure_pending": "A procedure estimate remains pending; this does not prove that the procedure was omitted.",
}


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _key(value: Mapping) -> tuple:
    return (value.get("type_id", value.get("type", "unknown")), value.get("episode_id", ""),
            value.get("candidate_timecode_sec", value.get("timecode_sec")))


def _effective_pass(attempt: Mapping, stage: str) -> tuple[Mapping, Sequence[Mapping]]:
    original = attempt.get(stage) or {}
    refined = attempt.get("refined_" + stage) or {}
    # A refinement may resolve uncertainty, but cannot erase explicit contrary evidence.
    if (original.get("decision") == "rejected" or "no" in original.get("checks", {}).values()
            or "applies" in original.get("exception_checks", {}).values()):
        return original, attempt.get(stage + "_frames", [])
    chosen = "refined_" + stage if refined else stage
    return refined or original, attempt.get(chosen + "_frames", [])


def _pass_checks(answer: Mapping, stage: str, frames: Sequence[Mapping]) -> list[dict]:
    references = answer.get("check_frames", {})
    checks = []
    for name, status in answer.get("checks", {}).items():
        indices = [index for index in references.get(name, []) if type(index) is int and index >= 0]
        evidence = []
        for index in indices:
            if index < len(frames):
                frame = frames[index]
                if (type(frame.get("source_index")) is int and frame["source_index"] >= 0
                        and _number(frame.get("timestamp_sec"))):
                    reference = {"source_index": frame["source_index"], "timestamp_sec": frame["timestamp_sec"]}
                    if reference not in evidence:
                        evidence.append(reference)
        checks.append({"stage": stage, "name": name, "status": status if status in {"yes", "no"} else "unknown",
                       "frame_indices": indices, "evidence_refs": evidence})
    return checks


def _assessment_reason(row: Mapping) -> str | None:
    if row.get("applicability") == "not_applicable" or row.get("confirmed_finding"):
        return None
    if row.get("assessment") in {"unresolved", "partial_support"}:
        return "catalog_unresolved"
    if row.get("assessment") == "unassessed":
        return None
    score = row.get("score")
    if (row.get("scoring_kind") == "event_occurrence" and _number(score) and score > 0
            and row.get("evidence_refs")):
        return "catalog_event_unconfirmed"
    if (row.get("scoring_kind") == "required_procedure" and _number(score) and score > 0
            and row.get("pending_episode_count", 0) > 0):
        return "catalog_procedure_pending"
    return None


def build_case_decision(findings: Sequence[Mapping], review_queue: Sequence[Mapping],
                        attempts: Sequence[Mapping], coverage: Mapping,
                        catalog_assessments: Sequence[Mapping] = ()) -> dict:
    """Build UI explanations from existing decisions, never from a score threshold.

    Only supplied host-gated findings authorize ``confirmed_violation``. A rejected
    candidate remains diagnostic evidence without becoming an active review.
    Explicit host dismissal is respected while unresolved coverage stays visible.
    Check frame indices belong to the displayed verification pass, not the source
    video's frame numbering. Host-provided pass frames resolve them to source
    ``evidence_refs``; older attempts without that mapping retain an empty list.
    Inputs are never modified.
    """
    records: dict[tuple, dict] = {}
    for attempt in attempts:
        records.setdefault(_key(attempt), {"attempt": {}, "queue": [], "findings": []})["attempt"] = attempt
    for queued in review_queue:
        records.setdefault(_key(queued), {"attempt": {}, "queue": [], "findings": []})["queue"].append(queued)
    for finding in findings:
        key = _key(finding)
        if key not in records and "candidate_timecode_sec" not in finding:
            # Older saved findings have only event time and their candidate interval.
            matches = [candidate for candidate in records if candidate[:2] == key[:2]
                       and _number(candidate[2]) and _number(finding.get("start_sec"))
                       and _number(finding.get("end_sec"))
                       and finding["start_sec"] <= candidate[2] <= finding["end_sec"]]
            if matches:
                key = min(matches, key=lambda candidate: abs(candidate[2] - finding["timecode_sec"]))
        records.setdefault(key, {"attempt": {}, "queue": [], "findings": []})["findings"].append(finding)

    cases = []
    for key, record in records.items():
        attempt = record["attempt"]
        (action, action_frames), (context, context_frames) = (_effective_pass(attempt, stage) for stage in ("action", "context"))
        checks = _pass_checks(action, "action", action_frames) + _pass_checks(context, "context", context_frames)
        if "exception_checks" in context:
            exception_view = {"checks": {attempt.get("exception_descriptions", {}).get(name, name):
                                         {"excluded": "yes", "applies": "no"}.get(status, "unknown")
                                         for name, status in context["exception_checks"].items()},
                              "check_frames": {attempt.get("exception_descriptions", {}).get(name, name): refs
                                               for name, refs in context.get("exception_frames", {}).items()}}
            checks.extend(_pass_checks(exception_view, "exception_exclusion", context_frames))
        codes = [code for queued in record["queue"] for code in queued.get("reason_codes", [])]
        for stage, answer in (("action", action), ("context", context)):
            if not answer:
                continue
            values = answer.get("checks", {}).values()
            if "no" in values:
                codes.append(stage + "_check_failed")
            if {"unknown", "not_visible"}.intersection(values):
                codes.append(stage + "_checks_unresolved")
            if answer.get("decision") in {"rejected", "uncertain"}:
                codes.append(("legal_context" if stage == "context" else stage) + "_" + answer["decision"])
            if answer.get("visibility") in {"partial", "unusable"}:
                codes.append(stage + "_visibility_limited")
        rejected = (any(check["status"] == "no" for check in checks)
                    or any(answer.get("decision") == "rejected" for answer in (action, context))
                    or any(code in {"action_rejected", "legal_context_rejected", "action_contradiction", "context_contradiction"}
                           for code in codes))
        status = "confirmed" if record["findings"] else "rejected" if rejected else "review_required"
        if status == "confirmed":
            codes.append("supported_finding")
        elif status == "review_required":
            if action and not context:
                codes.append("context_not_verified")
            if not codes:
                codes.append("candidate_not_confirmed")
        codes = list(dict.fromkeys(codes))
        observed = (action.get("observed_action") or context.get("observed_action")
                    or next((item.get("observed_action") for item in record["queue"] if item.get("observed_action")), "")
                    or next((item.get("description") for item in record["findings"] if item.get("description")), ""))
        requires_review = (bool(record["queue"]) or status == "review_required") and not rejected
        if attempt.get("requires_review") is False and not record["findings"]:
            requires_review = False
        cases.append({"type_id": key[0], "episode_id": key[1], "timecode_sec": key[2],
                      "status": status, "requires_review": requires_review,
                      "observed_action": observed, "action_decision": action.get("decision", "not_checked"),
                      "context_decision": context.get("decision", "not_checked"), "checks": checks,
                      "reason_codes": codes, "reasons": [_REASONS.get(code, code.replace("_", " ")) for code in codes]})

    unresolved = [(row, reason) for row in catalog_assessments if (reason := _assessment_reason(row))]
    for row, reason in unresolved:
        # Keep catalog-only evidence visible without duplicating an existing candidate.
        if any(case["type_id"] == row.get("item_id") and case["requires_review"] for case in cases):
            continue
        times = [ref["timestamp_sec"] for ref in row.get("evidence_refs", []) if _number(ref.get("timestamp_sec"))]
        cases.append({"type_id": row.get("item_id", "unknown"), "episode_id": row.get("episode_id", ""),
                      "timecode_sec": min(times) if times else None, "status": "review_required", "requires_review": True,
                      "observed_action": row.get("label", ""), "action_decision": "not_checked", "context_decision": "not_checked",
                      "checks": [], "reason_codes": [reason], "reasons": [_REASONS[reason]]})

    gaps = coverage.get("visibility_gaps", [])
    incomplete = (_number(coverage.get("windows_planned")) and _number(coverage.get("windows_completed"))
                  and coverage["windows_completed"] < coverage["windows_planned"])
    counts = {"confirmed": len(findings), "review_required": sum(case["requires_review"] for case in cases),
              "rejected": sum(case["status"] == "rejected" for case in cases), "candidates": len(cases),
              "visual_gaps": len(gaps), "unresolved_assessments": len(unresolved)}
    reasons = []
    if findings:
        reasons.append("supported_findings")
    if review_queue and any(case["requires_review"] for case in cases):
        reasons.append("queued_candidates")
    if any(case["status"] == "review_required" and case["requires_review"] for case in cases):
        reasons.append("unresolved_candidates")
    if coverage.get("unresolved_verification_count", 0):
        reasons.append("unresolved_verification")
    if coverage.get("unverified_candidate_count", 0) or coverage.get("verification_complete") is False:
        reasons.append("verification_incomplete")
    if gaps:
        reasons.append("visual_coverage_gaps")
    if incomplete:
        reasons.append("incomplete_screening")
    if unresolved:
        reasons.append("unresolved_catalog_assessments")
    if findings:
        status, title = "confirmed_violation", "Supported violation evidence"
        summary = f"{len(findings)} finding(s) passed the host evidence and rule checks. Human review is required."
        if counts["review_required"] or gaps or incomplete:
            summary += " Additional candidates or coverage limitations remain for review."
    elif reasons:
        status, title = "review_required", "Review required"
        summary = ("No violation passed all required checks. Unresolved evidence or coverage limitations require human review."
                   if any(reason != "queued_candidates" for reason in reasons)
                   else "No violation passed all required checks. Rejected proposals remain in the human review queue.")
    else:
        status, title = "no_supported_violation", "No supported violation"
        summary = "No supported violation was found in the sampled frames. This does not certify compliance."
    return {"status": status, "title": title, "summary": summary, "counts": counts, "reason_codes": reasons, "cases": cases}
