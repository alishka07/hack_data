"""Broad source-grounded scenario proposals, separate from legal decisions.

This module never reads source labels, filenames, hashes or prior case results.
Manual catalog entries may produce review hypotheses, never automatic findings.
"""
from __future__ import annotations

import math
import re

from worker.video import AnalysisError, FrameRef


MAX_SCENARIO_OBSERVATIONS = 6


def absence_only(text: str) -> bool:
    """Whether every reported clause explicitly says a cue was not observed.

    This deliberately does not classify omitted procedures ("cash not stored")
    or mixed observations ("no customer visible while cash is counted") as
    empty evidence. Category-specific contradictions need the filter below.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    clauses = [part.strip() for part in re.split(
        r"[.;,!?]|\b(?:but|while|however|although|and)\b", text.lower()) if part.strip()]
    return bool(clauses) and all(bool(re.match(
        r"(?:there (?:is|are) )?(?:no\b|none\b|nothing\b|not (?:visible|observed|seen)\b)", clause))
        or bool(re.fullmatch(r"[\w\s/-]+ (?:is |are |was |were )?not (?:visible|observed|seen)(?:\s+in\s+[\w\s/-]+)?", clause))
        for clause in clauses)


def _negative_only_evidence(identifier: str, evidence: str) -> bool:
    """Drop explicit denials of this cue, while preserving mixed positive actions.

    This is a conservative proposal filter, never a compliance decision. In
    particular, absent customers must not suppress a visible cash-counting cue.
    """
    if identifier == "active_intake_departure":
        subject = r"depart\w*|leav\w*|exit\w*"
    elif identifier == "keys_stamps_secure_handling":
        subject = r"keys?|stamps?"
    elif identifier in {"personal_belongings_in_cash_unit", "personal_device_exception_eligibility"}:
        subject = r"personal belongings|bags?|coats?|phones?|devices?|food|drinks?"
    elif identifier in {"previous_transaction_cash_exposed", "previous_cash_stored",
                        "cash_intake_without_client_presence", "backoffice_processing_control",
                        "loose_cash_issued_without_observed_count", "vault_cash_segregation_storage"}:
        subject = r"cash(?:-like)?|currency|banknotes?|coins?|money"
    else:
        return absence_only(evidence)
    clauses = re.split(r"[.;,!?]|\b(?:but|while|however|although)\b", evidence.lower())
    negative, affirmative = False, False
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if subject is None:
            negated = bool(re.match(r"(?:there (?:is|are) )?(?:no|not|never|without)\b", clause))
        else:
            negated = bool(re.search(
                r"\b(?:no|not|never|without)\s+(?:[\w-]+\s+){0,3}(?:" + subject + r")\b"
                r"|\b(?:" + subject + r")\b(?:\s+[\w-]+){0,4}\s+(?:not|absent|unobserved)\b", clause))
        relevant = subject is None or bool(re.search(r"\b(?:" + subject + r")\b", clause))
        if relevant and negated:
            negative = True
        elif relevant:
            affirmative = True
    return negative and not affirmative


def scenario_absence_only(identifier: str, text: str) -> bool:
    """Shared category-aware filter for explicitly negated physical cues."""
    if not isinstance(identifier, str) or not isinstance(text, str) or not text.strip():
        return False
    return _negative_only_evidence(identifier, text)

# Cues describe physical observations. Authorization, ownership, room designation
# and completion of an unobserved procedure remain downstream unknowns.
SCENARIO_CUES = {
    "active_intake_departure": "Cash received, followed by actual workstation exit before visible service completion.",
    "cash_intake_without_client_presence": "Cash acceptance/counting while a fully visible customer position is empty.",
    "previous_transaction_cash_exposed": "Same visible cash remains on the counter as a different customer transaction begins.",
    "intake_banknotes_counted": "Visible note-by-note or machine counting during cash intake; hidden counts remain unknown.",
    "intake_coins_counted": "Identifiable coins and visible manual/equipment counting during intake.",
    "intake_client_presence_required": "Customer presence/visibility during the actual cash acceptance/counting sequence.",
    "previous_cash_stored": "Cash storage placement or cash visibly left outside storage across a customer transition.",
    "loose_cash_issued_without_observed_count": "Loose cash preparation and customer handover; locate the counting opportunity.",
    "damaged_incomplete_strap_count": "Visible incomplete/damaged note strap, extraction and subsequent counting/issue.",
    "incomplete_coin_sack_count": "Identifiable incomplete coin sack, extraction and subsequent counting/issue.",
    "payout_bank_packaging": "Readable cash packaging during issue; bank origin/contracts remain unknown.",
    "client_recount_supervision": "Customer visibly recounts cash; inspect other visible people and the counting location.",
    "backoffice_processing_control": "Cash sorting/counting/packing away from a customer exchange; inspect visible second-person oversight.",
    "intake_authenticity_validity_checks": "Cash inspection or detector use; unseen inspection and actual authenticity remain unknown.",
    "machine_security_features": "Visible cash-processing equipment use; internal machine features remain unknown.",
    "intake_client_identification": "Document/device presentation during intake; identity checks may be offscreen/electronic.",
    "payout_client_identification": "Document/device presentation during payout; identity checks may be offscreen/electronic.",
    "intake_confirmation_document": "Visible receipt/document handover after intake; absent samples cannot prove omission.",
    "payout_confirmation_document": "Visible receipt/document handover after payout; absent samples cannot prove omission.",
    "personal_belongings_in_cash_unit": "Bag, coat, food/drink or personal-looking object placed/used in the cash work area.",
    "personal_cash_valuables_in_cash_unit": "Cash/valuables moved to clothing or personal-looking container; ownership stays unknown.",
    "personal_device_exception_eligibility": "Phone or handheld device used/stored in the cash work area; device features stay unknown.",
    "vault_entry_authorization": "Person crosses a safe/vault-like access boundary; authorization and room class stay unknown.",
    "vault_other_person_accompaniment": "People entering a safe/vault-like area together or separately; official roles stay unknown.",
    "keys_stamps_secure_handling": "Small key/stamp-like object handled, transferred, left on counter or stored; identify its visible shape.",
    "vault_cash_segregation_storage": "Cash/packages/bags moved among safe, shelf, floor or containers; inspect visible mixing and separation.",
    "cash_unit_isolation": "Visible access openings/partitions around cash area; construction classification stays unknown.",
    "cashier_room_lockable_entrance": "Door/access-boundary use and visible lock hardware; lockability cannot follow from an open door.",
    "cash_transfer_device_serviceable": "Cash transfer opening/device use or visible malfunction; unseen configuration stays unknown.",
    "cash_transfer_controls_inside": "Visible cash transfer controls and operator side; hidden controls stay unknown.",
    "counting_room_lockable_entrance": "Access door/lock in a cash-counting area; designated room class stays unknown.",
    "collection_loading_access_restriction": "People crossing a visible collection/loading boundary; access rights stay unknown.",
    "collection_loading_view_restriction": "Sight lines into a collection/loading area; its designated status stays unknown.",
}


def _items(catalog: dict) -> list[dict]:
    if not isinstance(catalog, dict):
        raise ValueError("Scenario screening requires a catalog")
    items = catalog["items"] if isinstance(catalog.get("items"), list) else list(catalog.values())
    if not 1 <= len(items) <= 64:
        raise ValueError("Scenario screening requires one to 64 catalog items")
    identifiers = []
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not item["id"] or len(item["id"]) > 100):
            raise ValueError("Scenario screening requires bounded catalog IDs")
        identifiers.append(item["id"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Scenario screening requires unique catalog IDs")
    return items


def scenario_prompt(catalog: dict) -> str:
    """A compact checklist added to an existing screening call, with no extra call."""
    lines = []
    for item in _items(catalog):
        identifier = item["id"]
        cues = item.get("visual_cues")
        descriptions = item.get("check_descriptions")
        if (isinstance(descriptions, dict) and descriptions
                and all(isinstance(text, str) for text in descriptions.values())):
            cue = " ".join(list(descriptions.values())[:2])[:220]
        else:
            cue = "; ".join(cues)[:220] if isinstance(cues, list) and all(isinstance(cue, str) for cue in cues) else SCENARIO_CUES.get(identifier)
        if cue is None:
            # A newly curated catalog entry remains screenable without importing
            # a private example. The bounded label is DATA, not model instruction.
            cue = "Inspect visible cues for: " + str(item.get("label", identifier))[:140]
        lines.append(f"{identifier}: {cue}")
    return (
        "\nAlso return scenario_observations (at most six distinct visible scenario hypotheses). "
        "Inspect every checklist family below, including review-only/manual entries. This broad pass proposes "
        "physical cues; it does not decide legality. Use state observed only when the stated physical cue is visible, "
        "uncertain when a visible cue has ambiguous interpretation. Do not require a proven breach to propose a cue. "
        "Do not invent hidden objects, ownership, authorization, official roles or room classification. "
        "Small keys/stamps, bags, phones, cash placement and storage transitions deserve explicit inspection. "
        "An open drawer, empty crop, lone worker or counting gesture alone does not prove a violation. "
        "Routine completed procedures need no scenario proposal unless their context needs review. "
        "Unseen cues produce no row; no row means unobserved, never compliant. "
        "Do not return negative observations such as 'No cash is visible' or 'No departure observed'. "
        "Each row cites this request's anchor_frame/start_frame/end_frame, with start <= anchor <= end, "
        "and concise visible_evidence describing only the supplied pixels. Choose the clearest action frame, "
        "not automatically frame zero. Prioritize distinct concrete actions over repeated generic risks. "
        "confidence is uncalibrated 0..1 support for this physical cue, never a violation probability. "
        "Keep an otherwise specific review-worthy action in the existing unmapped_visual_event route. Checklist DATA:\n"
        + "\n".join(lines)
    )


def scenario_schema(catalog: dict, frame_count: int) -> dict:
    """Array schema to embed as ``scenario_observations`` in a window response."""
    if type(frame_count) is not int or not 1 <= frame_count <= 8:
        raise ValueError("Scenario screening needs one to eight source frames")
    identifiers = [item["id"] for item in _items(catalog)]
    index = {"type": "integer", "minimum": 0, "maximum": frame_count - 1}
    properties = {
        "item_id": {"type": "string", "enum": identifiers},
        "state": {"type": "string", "enum": ["observed", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "anchor_frame": index, "start_frame": index, "end_frame": index,
        "visible_evidence": {"type": "string", "minLength": 1, "maxLength": 180},
    }
    return {"type": "array", "maxItems": MAX_SCENARIO_OBSERVATIONS,
            "items": {"type": "object", "properties": properties,
                      "required": list(properties), "additionalProperties": False}}


def collect_scenario_candidates(answer: dict, frames: list[FrameRef], catalog: dict) -> list[dict]:
    """Host-bind hypotheses to source PTS; no hypothesis is a confirmed finding.

    Missing observations are tolerated for older injected synthetic runtimes.
    The inference response schema should require the new field.
    """
    items = {item["id"]: item for item in _items(catalog)}

    def invalid():
        raise AnalysisError("model_schema_invalid", "Scenario screening returned invalid fields or source references")

    if not isinstance(answer, dict) or not isinstance(frames, list) or not frames or len(frames) > 8:
        invalid()
    source_times = {}
    for frame in frames:
        if (not isinstance(frame, FrameRef) or type(frame.index) is not int or frame.index < 0
                or type(frame.timestamp) not in (int, float) or not math.isfinite(frame.timestamp)
                or frame.timestamp < 0):
            invalid()
        if frame.index in source_times and source_times[frame.index] != frame.timestamp:
            invalid()
        source_times[frame.index] = frame.timestamp
    rows = answer.get("scenario_observations", [])
    if not isinstance(rows, list) or len(rows) > MAX_SCENARIO_OBSERVATIONS:
        invalid()
    expected = {"item_id", "state", "confidence", "anchor_frame", "start_frame", "end_frame", "visible_evidence"}
    result, seen = [], {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected:
            invalid()
        identifier, state = row["item_id"], row["state"]
        if (not isinstance(identifier, str) or identifier not in items
                or not isinstance(state, str) or state not in {"observed", "uncertain"}
                or type(row["confidence"]) not in (int, float) or not math.isfinite(row["confidence"])
                or not 0 <= row["confidence"] <= 1
                or not isinstance(row["visible_evidence"], str) or not row["visible_evidence"].strip()
                or len(row["visible_evidence"]) > 180):
            invalid()
        indices = [row[name] for name in ("anchor_frame", "start_frame", "end_frame")]
        if any(type(index) is not int or not 0 <= index < len(frames) for index in indices):
            invalid()
        # Structured decoding bounds each reference but cannot enforce ordering.
        # Canonicalize only the three actually cited source frames; keep the
        # chosen action anchor, and never manufacture an additional observation.
        anchor_index = row["anchor_frame"]
        start_index = min(indices, key=lambda index: (frames[index].timestamp, index))
        end_index = max(indices, key=lambda index: (frames[index].timestamp, index))
        anchor, start, end = (frames[index].timestamp for index in (anchor_index, start_index, end_index))
        evidence = row["visible_evidence"].strip()
        if scenario_absence_only(identifier, evidence):
            continue
        item = items[identifier]
        candidate = {"type_id": identifier, "anchor": float(anchor), "start": float(start), "end": float(end),
                       "observation": row["visible_evidence"].strip(), "state": state,
                       "visible_evidence": row["visible_evidence"].strip(),
                       "observed_action": row["visible_evidence"].strip(), "confidence": float(row["confidence"]),
                       "anchor_frame": anchor_index, "start_frame": start_index, "end_frame": end_index,
                       "frame_indices": list(dict.fromkeys(indices)),
                       "source_indices": list(dict.fromkeys(frames[index].index for index in indices)),
                       "origin": "scenario_screening", "confirmed": False,
                       "requires_external_context": item.get("automation") == "manual" or item.get("disposition") == "review_only"}
        episode = (identifier, frames[start_index].index, frames[end_index].index)
        if episode in seen:
            position = seen[episode]
            if candidate["confidence"] > result[position]["confidence"]:
                result[position] = candidate
        else:
            seen[episode] = len(result)
            result.append(candidate)
    return result
