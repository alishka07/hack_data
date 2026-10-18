"""Bounded, source-linked local model confidence for visible catalog checks.

These estimates are uncalibrated evidence confidence, never legal verdicts.
Private visual descriptions are deliberately absent from exported records/errors.
"""
from __future__ import annotations

from copy import deepcopy
import math

from worker.video import AnalysisError, FrameRef


OBSERVABLE_CHECKS = {
    "active_intake_departure": "Unfinished intake established; cashier actually leaves workstation; standing/reaching is insufficient.",
    "cash_intake_without_client_presence": "Identified intake cash accepted/counted while client demonstrably absent from visible client zone; cropped-out is unknown.",
    "previous_transaction_cash_exposed": "Same earlier client's cash visibly persists outside designated storage into distinct next transaction; piles/open drawer alone insufficient.",
    "intake_banknotes_counted": "Complete count of same intake notes with client present, manual/machine; handling/count gesture alone insufficient.",
    "intake_coins_counted": "Identified intake coins completely counted with client present, manually/by equipment; unresolved tiny objects unknown.",
    "intake_client_presence_required": "Client visibly present throughout actual acceptance/counting of identified intake cash; outside view unknown.",
    "previous_cash_stored": "Same accepted cash visibly placed in designated storage before distinct next client transaction; cash lineage required.",
    "loose_cash_issued_without_observed_count": "Same payout cash completely counted with client present before handover; intact qualifying packs/straps and packaged coins have exceptions.",
    "damaged_incomplete_strap_count": "Clearly incomplete/damaged payout strap identified; its notes completely counted; intact full straps exempt.",
    "incomplete_coin_sack_count": "Clearly incomplete payout coin sack identified; its coins completely counted; qualifying intact packaging exempt.",
    "client_recount_supervision": "Actual client recount at window/designated counting room with established cashier/controller oversight; ordinary transactions exempt.",
    "personal_food_at_workstation": "Identifiable food or a snack on the operating cashier work table counts as personal belongings under the user policy; eating is not required; unknown objects or cup contents are insufficient.",
    "workstation_departure_screen_left_on": "Operator actually leaves workstation, then same identified computer/laptop display visibly on at unattended desk; static empty chair or standing insufficient; locked but lit is on; reflection/off/black/standby/closed lid cannot establish on.",
}
_EVENT_IDS = frozenset({"active_intake_departure", "cash_intake_without_client_presence",
                        "previous_transaction_cash_exposed", "personal_food_at_workstation",
                        "workstation_departure_screen_left_on"})
_FIELDS = {"item_id", "assessment", "confidence", "applicability", "frame_indices"}


def _invalid():
    raise AnalysisError("model_schema_invalid", "A local confidence observation contained invalid fields or source references")


def _number(value):
    return type(value) is int or type(value) is float and math.isfinite(value)


def catalog_observation_schema(frame_count):
    """Return the constrained array schema embedded in the window response."""
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("Confidence observations require at least one source frame")
    properties = {
        "item_id": {"type": "string", "enum": list(OBSERVABLE_CHECKS)},
        "assessment": {"type": "string", "enum": ["event_evidence", "completion_evidence", "partial_support", "unresolved"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "applicability": {"type": "string", "enum": ["applicable", "unknown"]},
        "frame_indices": {"type": "array", "items": {"type": "integer", "minimum": 0,
                                                       "maximum": frame_count - 1},
                          "minItems": 1, "maxItems": 3},
    }
    return {"type": "array", "maxItems": 6,
            "items": {"type": "object", "properties": properties,
                      "required": list(properties), "additionalProperties": False}}


def confidence_prompt(catalog):
    """Compact protocol requirements; the source catalog remains authoritative."""
    if not isinstance(catalog, dict) or not isinstance(catalog.get("items"), list):
        raise ValueError("Confidence prompting requires a catalog")
    supplied = {item.get("id"): item for item in catalog["items"] if isinstance(item, dict)}
    lines = []
    for identifier, requirement in OBSERVABLE_CHECKS.items():
        item = supplied.get(identifier)
        expected = "event_occurrence" if identifier in _EVENT_IDS else "required_procedure"
        if item is None or item.get("scoring_kind") != expected:
            raise ValueError("Observable confidence checks conflict with the catalog")
        lines.append(f"{identifier} ({'event' if identifier in _EVENT_IDS else 'procedure'}): {requirement}")
    return (
        "\nAlso return catalog_assessments, at most six applicable or unresolved visible checks, one entry per item. "
        "Use event_evidence for visible support of a possible forbidden event; completion_evidence for visible support of correct protocol completion. "
        "Give intermediate confidence even when uncertain: evidence0.68 means68% occurrence estimate; completion0.82 means18% remaining estimate. "
        "Use partial_support or unresolved for uncertain evidence; either may have intermediate confidence. Zero means no meaningful supporting evidence. "
        "Do not require a confirmed violation or perfect certainty before reporting evidence. "
        "Confidence is your self-estimated UNCALIBRATED support0..1 for the event or correct completion; account for the weakest condition without forcing every unknown to zero. "
        "Use actual frame_indices1..3. Partial visibility lowers certainty; unseen conditions stay unknown. "
        "Never infer omitted actions between samples or clear a whole transaction from a generic count gesture. "
        "Applicability is applicable only when its trigger/context is established, otherwise unknown. "
        "No manual-only checks, legal conclusions, identity or unseen records.\n" + "\n".join(lines)
    )


def collect_confidence_observations(answer, frames, window_id):
    """Validate a whole window atomically and bind indices to original source PTS.

    Missing ``catalog_assessments`` is tolerated for older synthetic runtimes.
    The actual model response schema still requires the field.
    """
    if (not isinstance(answer, dict) or not isinstance(answer.get("visibility"), str)
            or answer["visibility"] not in {"clear", "partial", "unusable"}
            or type(window_id) is not int or window_id < 0 or not isinstance(frames, list) or not frames):
        _invalid()
    source_times = {}
    for frame in frames:
        if (not isinstance(frame, FrameRef) or type(frame.index) is not int or frame.index < 0
                or not _number(frame.timestamp) or frame.timestamp < 0):
            _invalid()
        if frame.index in source_times and source_times[frame.index] != frame.timestamp:
            _invalid()
        source_times[frame.index] = frame.timestamp
    supplied = answer.get("catalog_assessments", [])
    if not isinstance(supplied, list) or len(supplied) > 6:
        _invalid()
    seen, result = set(), []
    for observation in supplied:
        if not isinstance(observation, dict) or set(observation) != _FIELDS:
            _invalid()
        identifier = observation["item_id"]
        if not isinstance(identifier, str) or identifier not in OBSERVABLE_CHECKS or identifier in seen:
            _invalid()
        state, confidence = observation["assessment"], observation["confidence"]
        expected = "event_evidence" if identifier in _EVENT_IDS else "completion_evidence"
        if (not isinstance(state, str) or state not in {expected, "partial_support", "unresolved"}
                or not _number(confidence) or not 0 <= confidence <= 1
                or not isinstance(observation["applicability"], str)
                or observation["applicability"] not in {"applicable", "unknown"}):
            _invalid()
        indices = observation["frame_indices"]
        if not isinstance(indices, list) or not 1 <= len(indices) <= 3:
            _invalid()
        references = {}
        for index in indices:
            if type(index) is not int or not 0 <= index < len(frames):
                _invalid()
            frame = frames[index]
            references[frame.index] = {"source_index": frame.index, "timestamp_sec": float(frame.timestamp)}
        ordered = sorted(references.values(), key=lambda ref: (ref["timestamp_sec"], ref["source_index"]))
        canonical_state = state
        if state == "unresolved" and confidence > 0:
            canonical_state = "partial_support"
        elif state == "partial_support" and confidence == 0:
            canonical_state = "unresolved"
        result.append({"item_id": identifier, "assessment": canonical_state, "model_assessment": state, "confidence": float(confidence),
                       "applicability": observation["applicability"], "evidence_refs": ordered,
                       "start_sec": ordered[0]["timestamp_sec"], "end_sec": ordered[-1]["timestamp_sec"],
                       "window_id": window_id, "observable": answer["visibility"] != "unusable",
                       "window_start_sec": float(min(source_times.values())),
                       "window_end_sec": float(max(source_times.values()))})
        seen.add(identifier)
    return result


def scope_confidence_observations(observations, ledger):
    """Assign host-side scopes after all boundary cues have entered the ledger.

    An observed start is still a tentative boundary, not proof of cash/person
    identity. Cross-boundary references and unbounded observations stay local
    to their window, so one count cannot clear unrelated transactions.
    """
    result = []
    for observation in observations:
        row = deepcopy(observation)
        references = row["evidence_refs"]
        middle = references[len(references) // 2]["timestamp_sec"]
        identifier = ledger.episode_id(middle)
        same_scope = all(ledger.episode_id(ref["timestamp_sec"]) == identifier for ref in references)
        if identifier.startswith("tentative-start-") and same_scope:
            row.update(episode_id=identifier, scope_certainty="observed_start")
        else:
            row.update(episode_id=f"unscoped-window-{row['window_id']}", scope_certainty="window_only")
            if not same_scope:
                row["applicability"] = "unknown"
        result.append(row)
    return result
