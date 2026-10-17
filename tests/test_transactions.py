from copy import deepcopy
import json
from pathlib import Path

import pytest

from worker.ledger import ObservationLedger
from worker.transactions import build_transactions, merge_proposals
from worker.video import AnalysisError, FrameRef


def add(ledger, window, times, facts=(), *, kind="unknown", phase="unknown", visibility="clear", cue_indices=None):
    frames = [FrameRef(round(time * 30), time, Path(f"unused-{time}.jpg")) for time in times]
    ledger.add(window, frames, {"visibility": visibility,
        "visible_facts": [{"kind": item[0], "frame_indices": item[1],
                           "state": item[2] if len(item) > 2 else "observed",
                           "note": "Visible " + item[0]} for item in facts],
        "transaction": {"kind": kind, "phase": phase,
                        "frame_indices": cue_indices if cue_indices is not None else []}})


def test_explicit_actions_build_source_bound_transaction_states():
    ledger = ObservationLedger()
    add(ledger, 0, [1, 2, 3, 4, 5], [
        ("transaction_start", [0]), ("cash_received", [1]),
        ("cash_handed_over", [2]), ("cash_stored", [3]), ("transaction_end", [4])])
    result = build_transactions(ledger)
    assert [row["to_state"] for row in result["transitions"]] == ["started", "active", "handover", "storage", "completed"]
    episode = result["episodes"][0]
    assert episode["observed_start"] and episode["observed_end"]
    assert episode["state"] == "completed"
    assert episode["transaction_kind"] == "unknown"
    for transition in result["transitions"]:
        assert transition["timestamp_sec"] == transition["refs"][0]["timestamp_sec"]
        assert transition["source_indices"] == [ref["source_index"] for ref in transition["refs"]]
    json.dumps(result, allow_nan=False)


def test_handover_and_storage_do_not_invent_completion_or_missing_count():
    ledger = ObservationLedger()
    add(ledger, 0, [1, 2], [("cash_handed_over", [0]), ("cash_stored", [1])])
    result = build_transactions(ledger)
    assert result["episodes"][0]["state"] == "storage"
    assert not result["episodes"][0]["observed_end"]
    assert not any(row["to_state"] == "completed" for row in result["transitions"])
    assert result["proposals"][0]["type_id"] == "loose_cash_issued_without_observed_count"
    assert result["proposals"][0]["disposition"] == "verification_required"
    assert "no_count" not in json.dumps(result)


def test_unknown_later_window_does_not_inherit_transaction_kind():
    ledger = ObservationLedger()
    add(ledger, 0, [0, 1], [("transaction_start", [0])], kind="intake", phase="active", cue_indices=[1])
    add(ledger, 1, [2, 3], [("workstation_departure", [1])])
    result = build_transactions(ledger)
    assert result["episodes"][0]["transaction_kind"] == "unknown"
    assert result["proposals"][0]["transaction_kind"] == "unknown"
    assert "transaction_context_unknown" in result["episodes"][0]["unknowns"]
    assert result["proposals"][0]["type_id"] == "active_intake_departure"
    assert result["proposals"][0]["refs"] == [{"source_index": 90, "timestamp_sec": 3}]


def test_uncertain_action_and_tracking_disappearance_only_reduce_knowledge():
    ledger = ObservationLedger()
    add(ledger, 0, [0, 1], [("workstation_departure", [1], "uncertain")], visibility="partial")
    perception = {"visibility_gaps": [{"start_sec": 1, "end_sec": 2, "source_indices": [30],
                                      "reason": "detection_gap", "track_id": 7}],
                  "events": [{"kind": "track_disappeared", "timestamp_sec": 1}]}
    result = build_transactions(ledger, perception=perception)
    assert result["proposals"] == result["transitions"] == []
    assert {row["reason"] for row in result["unknowns"]} >= {
        "transaction_context_unknown", "uncertain_observation", "visibility_partial", "perception_visibility_gap"}
    assert all(not episode["observed_end"] for episode in result["episodes"])


def test_overlapping_window_reports_merge_but_repeated_episodes_survive():
    ledger = ObservationLedger()
    add(ledger, 0, [0, 1, 2], [("transaction_start", [0]), ("workstation_departure", [1, 2])])
    add(ledger, 1, [1, 2, 3], [("workstation_departure", [0, 1]), ("transaction_end", [2])])
    add(ledger, 2, [4, 5, 6], [("transaction_start", [0]), ("workstation_departure", [1]), ("transaction_end", [2])])
    result = build_transactions(ledger)
    proposals = [row for row in result["proposals"] if row["type_id"] == "active_intake_departure"]
    assert len(proposals) == 2
    assert proposals[0]["source_indices"] == [30, 60]
    assert proposals[0]["episode_id"] != proposals[1]["episode_id"]
    assert len([fact for episode in result["episodes"] for fact in episode["facts"]
                if fact["kind"] == "workstation_departure"]) == 2


def test_close_repeated_departures_with_return_are_not_merged():
    ledger = ObservationLedger()
    add(ledger, 0, [0, 1, 2, 3], [("transaction_start", [0]), ("workstation_departure", [1]),
                                 ("workstation_return", [2]), ("workstation_departure", [3])])
    proposals = build_transactions(ledger)["proposals"]
    assert [row["anchor"] for row in proposals] == [1, 3]


def test_input_order_and_snapshot_do_not_change_result_or_mutate_input():
    forward, reverse = ObservationLedger(), ObservationLedger()
    for ledger, order in ((forward, [0, 1]), (reverse, [1, 0])):
        for window in order:
            add(ledger, window, [window * 2, window * 2 + 1], [("cash_received", [0])])
    snapshot = forward.as_dict()
    original = deepcopy(snapshot)
    assert build_transactions(forward) == build_transactions(reverse) == build_transactions(snapshot)
    assert snapshot == original


def test_cash_action_after_client_departure_is_only_a_context_review_candidate():
    ledger = ObservationLedger()
    add(ledger, 0, [0, 1, 2], [("transaction_start", [0]), ("client_departure", [1]),
                             ("cash_count_machine", [2])], kind="intake", phase="active", cue_indices=[0])
    result = build_transactions(ledger)
    proposal = result["proposals"][0]
    assert proposal["type_id"] == "cash_intake_without_client_presence"
    assert proposal["source_indices"] == [30, 60]
    assert "absence_unestablished" in proposal["reason"]
    assert "client_demonstrably_absent" not in json.dumps(result)


def test_counting_alone_or_arrival_after_departure_does_not_propose_absence():
    ledger = ObservationLedger()
    add(ledger, 0, [0, 1, 2], [("client_departure", [0]), ("client_arrival", [1]), ("cash_count_manual", [2])])
    add(ledger, 1, [4], [("cash_received", [0])])
    assert build_transactions(ledger)["proposals"] == []


def test_complete_cue_after_handover_on_same_source_does_not_contradict_action():
    ledger = ObservationLedger()
    add(ledger, 0, [1], [("cash_handed_over", [0])], kind="payout", phase="complete", cue_indices=[0])
    result = build_transactions(ledger)
    assert [row["to_state"] for row in result["transitions"]] == ["handover", "completed"]
    assert result["episodes"][0]["state"] == "completed"
    assert "action_after_completion_without_new_start" not in [row["reason"] for row in result["unknowns"]]


def test_conflicting_cues_are_not_resolved_by_choosing_last_model_answer():
    ledger = ObservationLedger()
    add(ledger, 0, [1], kind="intake", phase="active", cue_indices=[0])
    add(ledger, 1, [1], kind="payout", phase="complete", cue_indices=[0])
    result = build_transactions(ledger)
    assert result["episodes"][0]["state"] == "unknown"
    assert result["episodes"][0]["transaction_kind"] == "unknown"
    assert any(row["reason"] == "conflicting_transaction_cues" for row in result["unknowns"])


def test_conflicting_source_clock_fails_explicitly():
    ledger = ObservationLedger()
    add(ledger, 0, [1], [("cash_received", [0])])
    snapshot = ledger.as_dict()
    duplicate = deepcopy(snapshot["facts"][0])
    duplicate["refs"][0]["timestamp_sec"] = 2
    duplicate["window_id"] = 1
    snapshot["facts"].append(duplicate)
    with pytest.raises(AnalysisError, match="conflicting transaction timestamps") as error:
        build_transactions(snapshot)
    assert error.value.code == "ledger_source_conflict"


def test_proposal_merge_requires_same_episode_and_shared_action_source():
    proposal = {"type_id": "active_intake_departure", "episode_id": "one", "anchor": 2,
                "start": 1, "end": 3, "source_indices": [30, 60], "action_source_indices": [60],
                "refs": [{"source_index": 30, "timestamp_sec": 1}, {"source_index": 60, "timestamp_sec": 2}],
                "observation": "Visible departure", "reason": "verification_required"}
    duplicate = deepcopy(proposal)
    repeated = deepcopy(proposal)
    repeated["action_source_indices"] = [90]
    repeated["source_indices"] = [30, 90]
    repeated["refs"][-1] = {"source_index": 90, "timestamp_sec": 3}
    next_episode = {**deepcopy(proposal), "episode_id": "two"}
    assert len(merge_proposals([proposal, duplicate, repeated, next_episode])) == 3
