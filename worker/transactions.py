"""Source-bound transaction episodes and review proposals, never legal findings.

The state is the latest *observed* phase, not proof of uninterrupted activity
between sampled frames. Transaction kind stays local to each window: an unknown
cue is never filled from an earlier intake or payout observation.
"""
from __future__ import annotations

from copy import deepcopy
import math

from worker.video import AnalysisError


_STATE_FOR_ACTION = {
    "transaction_start": "started",
    "cash_received": "active",
    "cash_count_manual": "active",
    "cash_count_machine": "active",
    "cash_prepared": "active",
    "cash_handed_over": "handover",
    "cash_stored": "storage",
    "transaction_end": "completed",
}
_ORDER = {"transaction_start": 0, "transaction_cue": 1,
          "cash_received": 2, "cash_prepared": 2,
          "cash_count_manual": 2, "cash_count_machine": 2,
          "cash_handed_over": 3, "cash_stored": 3, "transaction_end": 4}


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _refs(records: list[dict]) -> list[dict]:
    """Union authoritative references, detecting conflicting source clocks."""
    result: dict[int, dict] = {}
    for record in records:
        for ref in record.get("refs", []):
            index, timestamp = ref.get("source_index"), ref.get("timestamp_sec")
            if type(index) is not int or index < 0 or not _number(timestamp):
                raise AnalysisError("ledger_source_invalid", "A transaction observation has an invalid source reference")
            if index in result and result[index]["timestamp_sec"] != timestamp:
                raise AnalysisError("ledger_source_conflict", "A source frame has conflicting transaction timestamps")
            result[index] = {"source_index": index, "timestamp_sec": timestamp}
    return sorted(result.values(), key=lambda ref: (ref["timestamp_sec"], ref["source_index"]))


def _same_observation(left: dict, right: dict) -> bool:
    if any(left.get(key) != right.get(key) for key in ("kind", "state", "transaction_kind", "phase")):
        return False
    a, b = left["refs"], right["refs"]
    if not a or not b:
        return False
    # Nearby timestamps alone cannot merge genuinely repeated actions. A shared
    # source frame, or overlapping nonzero observation intervals in different
    # windows, establishes that these windows describe the same action interval.
    if {ref["source_index"] for ref in a} & {ref["source_index"] for ref in b}:
        return True
    return (not set(left["window_ids"]) & set(right["window_ids"])
            and max(a[0]["timestamp_sec"], b[0]["timestamp_sec"])
            < min(a[-1]["timestamp_sec"], b[-1]["timestamp_sec"]))


def _records(data: dict) -> list[dict]:
    records = []
    for raw in data.get("facts", []) + data.get("transaction_cues", []):
        if not raw.get("refs"):
            continue
        record = deepcopy(raw)
        record["refs"] = _refs([record])
        record["source_indices"] = [ref["source_index"] for ref in record["refs"]]
        record["time_sec"] = record["refs"][0]["timestamp_sec"]
        record["window_ids"] = [record["window_id"]]
        record["notes"] = [record.get("note", "")]
        records.append(record)
    # Verify the clock across windows as well as within each record.
    _refs(records)
    merged: list[dict] = []
    for record in sorted(records, key=lambda row: (row["time_sec"], row["window_id"], row["kind"])):
        match = next((previous for previous in merged if _same_observation(previous, record)), None)
        if match is None:
            merged.append(record)
            continue
        match["refs"] = _refs([match, record])
        match["source_indices"] = [ref["source_index"] for ref in match["refs"]]
        match["window_ids"] = sorted(set(match["window_ids"] + record["window_ids"]))
        match["notes"] = list(dict.fromkeys(match["notes"] + record["notes"]))
    return sorted(merged, key=lambda row: (row["time_sec"],
                  5 if row["kind"] == "transaction_cue" and row.get("phase") == "complete"
                  else _ORDER.get(row["kind"], 2), row["kind"]))


def _episode_id(anchor: float, records: list[dict]) -> str:
    boundaries = [record for record in records if record["state"] == "observed"
                  and record["kind"] in {"transaction_start", "transaction_end"}
                  and (record["time_sec"] <= anchor if record["kind"] == "transaction_start"
                       else record["time_sec"] < anchor)]
    if not boundaries:
        return "tentative-unbounded"
    latest = max(boundaries, key=lambda row: (row["time_sec"], row["kind"] == "transaction_start"))
    label = "start" if latest["kind"] == "transaction_start" else "after-end"
    return f"tentative-{label}-{latest['source_indices'][0]:010d}"


def _unknown(reason: str, start: float, end: float, records: list[dict], **extra) -> dict:
    references = _refs(records)
    return {"reason": reason, "start_sec": start, "end_sec": end,
            "refs": references, "source_indices": [ref["source_index"] for ref in references], **extra}


def merge_proposals(proposals: list[dict]) -> list[dict]:
    """Merge overlap-window duplicates, retaining separately observed episodes.

    Proximity alone is insufficient. Shared action evidence is required so a
    second departure or handover cannot disappear merely because it is nearby.
    """
    merged: list[dict] = []
    for raw in sorted(proposals, key=lambda row: (row["anchor"], row["type_id"])):
        proposal = deepcopy(raw)
        match = next((previous for previous in merged
                      if previous["type_id"] == proposal["type_id"]
                      and previous["episode_id"] == proposal["episode_id"]
                      and set(previous.get("action_source_indices", previous["source_indices"]))
                      & set(proposal.get("action_source_indices", proposal["source_indices"]))
                      and max(previous["start"], proposal["start"]) <= min(previous["end"], proposal["end"])), None)
        if match is None:
            merged.append(proposal)
        else:
            match["start"] = min(match["start"], proposal["start"])
            match["end"] = max(match["end"], proposal["end"])
            match["source_indices"] = sorted(set(match["source_indices"] + proposal["source_indices"]))
            match["refs"] = _refs([match, proposal])
            if "action_source_indices" in match:
                match["action_source_indices"] = sorted(set(match["action_source_indices"]
                    + proposal.get("action_source_indices", proposal["source_indices"])))
    return merged


def build_transactions(ledger, *, perception: dict | None = None) -> dict:
    """Build deterministic, auditable state from an ObservationLedger or snapshot.

    Perception contributes visibility unknowns only. Tracks, missing detections,
    zones and motion cannot establish cash identity, roles, absence or completion.
    Returned proposals need the separate action/applicability/rule verifier.
    """
    data = ledger.as_dict() if hasattr(ledger, "as_dict") else deepcopy(ledger)
    if not isinstance(data, dict):
        raise TypeError("ledger must be an ObservationLedger or its dictionary snapshot")
    records = _records(data)
    episodes: dict[str, dict] = {}
    transitions, proposals, unknowns = [], [], []
    cues_by_window = {cue["window_id"]: cue for cue in data.get("transaction_cues", [])}

    def episode_for(anchor: float) -> dict:
        identity = _episode_id(anchor, records)
        return episodes.setdefault(identity, {
            "episode_id": identity, "start_sec": anchor, "end_sec": anchor,
            "state": "unknown", "transaction_kind": "unknown", "observed_start": False,
            "observed_end": False, "facts": [], "transaction_cues": [], "refs": [],
            "source_indices": [], "window_ids": [], "unknowns": [],
            "continuity": "unestablished_between_sampled_observations",
        })

    def add_unknown(item: dict) -> None:
        unknowns.append(item)
        episode = episode_for(item["start_sec"])
        episode["start_sec"] = min(episode["start_sec"], item["start_sec"])
        episode["end_sec"] = max(episode["end_sec"], item["end_sec"])
        episode["window_ids"] = sorted(set(episode["window_ids"] + item.get("window_ids", [])))
        if item["reason"] not in episode["unknowns"]:
            episode["unknowns"].append(item["reason"])

    # Preserve each window's unknown cue even when no frame could support it.
    # In particular, an unknown later window never inherits an earlier kind.
    for window in data.get("windows", []):
        extent = window.get("sampled_extent_sec", [])
        if len(extent) != 2 or not all(_number(value) for value in extent):
            continue
        start, end = extent
        cue = window.get("transaction", cues_by_window.get(window["window_id"], {}))
        if cue.get("transaction_kind", "unknown") == "unknown" or cue.get("phase", "unknown") == "unknown":
            add_unknown(_unknown("transaction_context_unknown", start, end, [cue],
                                 window_ids=[window["window_id"]]))
        if window.get("visibility") in {"partial", "unusable"}:
            window_records = [record for record in records if window["window_id"] in record["window_ids"]]
            add_unknown(_unknown("visibility_" + window["visibility"], start, end, window_records,
                                 window_ids=[window["window_id"]]))

    for record in records:
        episode = episode_for(record["time_sec"])
        episode["start_sec"] = min(episode["start_sec"], record["time_sec"])
        episode["end_sec"] = max(episode["end_sec"], record["refs"][-1]["timestamp_sec"])
        episode["refs"] = _refs([episode, record])
        episode["source_indices"] = [ref["source_index"] for ref in episode["refs"]]
        episode["window_ids"] = sorted(set(episode["window_ids"] + record["window_ids"]))
        key = "transaction_cues" if record["kind"] == "transaction_cue" else "facts"
        episode[key].append(deepcopy(record))
        if record["state"] != "observed":
            add_unknown(_unknown("uncertain_observation", record["time_sec"],
                                 record["refs"][-1]["timestamp_sec"], [record], kind=record["kind"]))
            continue
        target = _STATE_FOR_ACTION.get(record["kind"])
        if record["kind"] == "transaction_cue":
            target = {"active": "active", "complete": "completed"}.get(record.get("phase"))
        if target:
            previous = episode["state"]
            # Cash storage or handover alone cannot establish completed service.
            # Preserve a new observed active action after a completed cue as a
            # contradiction rather than silently reopening the same episode.
            if previous == "completed" and target not in {"completed", "started"}:
                add_unknown(_unknown("action_after_completion_without_new_start", record["time_sec"],
                                     record["refs"][-1]["timestamp_sec"], [record]))
                target = "unknown"
            if previous != target:
                transitions.append({"episode_id": episode["episode_id"], "from_state": previous,
                                    "to_state": target, "timestamp_sec": record["time_sec"],
                                    "kind": record["kind"], "refs": deepcopy(record["refs"]),
                                    "source_indices": list(record["source_indices"]),
                                    "window_ids": list(record["window_ids"]),
                                    "reason": "source_grounded_observed_phase" if target != "unknown"
                                    else "conflicting_observed_phases"})
            episode["state"] = target
            episode["observed_start"] |= record["kind"] == "transaction_start"
            episode["observed_end"] |= target == "completed"

        # These are verification opportunities, not claims that a rule applies.
        type_id = {"workstation_departure": "active_intake_departure",
                   "cash_handed_over": "loose_cash_issued_without_observed_count",
                   "access_boundary_crossing": "unmapped_visual_event"}.get(record["kind"])
        context_records = [record]
        reason = "observed_action_requires_applicability_and_exception_checks"
        if record["kind"] in {"cash_received", "cash_count_manual", "cash_count_machine"}:
            departures = [other for other in records if other["kind"] == "client_departure"
                          and other["state"] == "observed"
                          and set(other["window_ids"]) & set(record["window_ids"])
                          and other["time_sec"] <= record["time_sec"]
                          and _episode_id(other["time_sec"], records) == episode["episode_id"]]
            if departures:
                departure = max(departures, key=lambda row: row["time_sec"])
                arrivals = [other for other in records if other["kind"] == "client_arrival"
                            and other["state"] == "observed"
                            and departure["time_sec"] < other["time_sec"] <= record["time_sec"]]
                if not arrivals:
                    type_id = "cash_intake_without_client_presence"
                    context_records.append(departure)
                    reason = "observed_client_departure_and_cash_action_require_context_review_absence_unestablished"
        if type_id:
            references = _refs(context_records)
            context_kinds = {cues_by_window.get(window_id, {}).get("transaction_kind", "unknown")
                             for window_id in record["window_ids"]}
            proposals.append({"type_id": type_id, "anchor": record["time_sec"],
                              "start": references[0]["timestamp_sec"], "end": references[-1]["timestamp_sec"],
                              "observation": "; ".join(record["notes"]), "episode_id": episode["episode_id"],
                              "source_indices": [ref["source_index"] for ref in references],
                              "action_source_indices": list(record["source_indices"]),
                              "refs": references, "reason": reason,
                              "transaction_kind": next(iter(context_kinds)) if len(context_kinds) == 1 else "unknown",
                              "disposition": "verification_required"})

    # Conflicting source-grounded cues remain explicit instead of choosing a
    # preferred interpretation or treating repeated VLM answers as independent.
    cues = [record for record in records if record["kind"] == "transaction_cue"]
    for index, cue in enumerate(cues):
        for other in cues[index + 1:]:
            shared = set(cue["source_indices"]) & set(other["source_indices"])
            known_kinds = {cue.get("transaction_kind"), other.get("transaction_kind")} - {"unknown", None}
            known_phases = {cue.get("phase"), other.get("phase")} - {"unknown", None}
            if shared and (len(known_kinds) > 1 or len(known_phases) > 1):
                add_unknown(_unknown("conflicting_transaction_cues", min(cue["time_sec"], other["time_sec"]),
                                     max(cue["refs"][-1]["timestamp_sec"], other["refs"][-1]["timestamp_sec"]),
                                     [cue, other]))

    # Anonymous tracking gaps may reduce knowledge, never create cash facts.
    source_lookup = {ref["source_index"]: ref for record in records for ref in record["refs"]}
    for gap in (perception or {}).get("visibility_gaps", []):
        start, end = gap.get("start_sec"), gap.get("end_sec")
        if not _number(start) or not _number(end) or end < start:
            continue
        references = _refs([{"refs": gap.get("refs", [])}])
        references = _refs([{"refs": references}, {"refs": [source_lookup[index]
                           for index in gap.get("source_indices", []) if index in source_lookup]}])
        # Source-index-only detector gaps remain identifiable, but no timestamp
        # is manufactured when their frame was not among ledger observations.
        unresolved = sorted({index for index in gap.get("source_indices", [])
                             if type(index) is int and index >= 0
                             and index not in {ref["source_index"] for ref in references}})
        add_unknown(_unknown("perception_visibility_gap", start, end, [{"refs": references}],
                             perception_reason=gap.get("reason", "visibility_unknown"),
                             unresolved_source_indices=unresolved))

    for episode in episodes.values():
        local_cues = [cue for window_id, cue in cues_by_window.items() if window_id in episode["window_ids"]]
        kinds = {cue.get("transaction_kind", "unknown") for cue in local_cues}
        episode["transaction_kind"] = next(iter(kinds)) if len(kinds) == 1 else "unknown"
        if not episode["observed_start"]:
            episode["unknowns"].append("start_boundary_unobserved")
        if not episode["observed_end"]:
            episode["unknowns"].append("end_boundary_unobserved")
        if "conflicting_transaction_cues" in episode["unknowns"]:
            episode["transaction_kind"] = "unknown"
            episode["state"] = "unknown"

    return {"version": "1.0", "semantics": "Observed phases and verification candidates only; sampled gaps do not prove continuity, client absence, cash ownership, omitted procedures or legal violations.",
            "episodes": sorted(episodes.values(), key=lambda episode: (episode["start_sec"], episode["episode_id"])),
            "transitions": transitions, "proposals": merge_proposals(proposals), "unknowns": unknowns}
