"""Scenario recall proposals remain bounded, source linked and separate from verdicts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from worker.runtime import _validate_json_schema
from worker.scenario_screening import (absence_only, collect_scenario_candidates, scenario_absence_only, scenario_prompt,
                                      scenario_schema)
from worker.video import AnalysisError, FrameRef


ROOT = Path(__file__).resolve().parents[1]
IDS = ("previous_transaction_cash_exposed", "previous_cash_stored",
       "cash_intake_without_client_presence", "backoffice_processing_control",
       "keys_stamps_secure_handling", "vault_cash_segregation_storage",
       "personal_belongings_in_cash_unit", "active_intake_departure",
       "loose_cash_issued_without_observed_count")


def registry():
    return {identifier: {"id": identifier, "label": identifier, "disposition": "review_only",
                         "visual_cues": [f"Visible cue for {identifier}"],
                         "unknowns": ["Ownership and applicability"]} for identifier in IDS}


def frames():
    return [FrameRef(30, 1., Path("synthetic-a.jpg")), FrameRef(90, 3., Path("synthetic-b.jpg")),
            FrameRef(150, 5., Path("synthetic-c.jpg"))]


def observation(**changes):
    return {"item_id": "keys_stamps_secure_handling", "state": "observed", "confidence": .71,
            "anchor_frame": 1, "start_frame": 0, "end_frame": 2,
            "visible_evidence": "Small key-shaped object placed on the counter.", **changes}


def response(*rows):
    return {"scenario_observations": list(rows)}


def test_all_scenarios_enter_one_existing_call_without_private_case_data():
    scenarios = registry()
    prompt = scenario_prompt(scenarios)
    assert all(identifier in prompt for identifier in IDS)
    assert "no row means unobserved, never compliant" in prompt
    assert "does not decide legality" in prompt and "uncalibrated" in prompt
    assert len(prompt) < 6000
    schema = scenario_schema(scenarios, 8)
    assert set(schema["items"]["properties"]["item_id"]["enum"]) == set(IDS)
    assert schema["maxItems"] == 6
    assert "open_labels" not in prompt and "cashier_" not in prompt


def test_catalog_compatibility_keeps_manual_scenarios_screenable():
    catalog = json.loads((ROOT / "rules/catalog.json").read_text(encoding="utf-8"))
    prompt = scenario_prompt(catalog)
    assert all(item["id"] in prompt for item in catalog["items"])
    assert "keys_stamps_secure_handling" in prompt and "personal_belongings_in_cash_unit" in prompt
    assert len(prompt) < 11000


@pytest.mark.parametrize("state", ["observed", "uncertain"])
def test_physical_hypotheses_never_become_automatic_findings(state):
    answer = response(observation(state=state))
    before = deepcopy(answer)
    _validate_json_schema(answer["scenario_observations"], scenario_schema(registry(), 3))
    result = collect_scenario_candidates(answer, frames(), registry())
    assert answer == before
    assert len(result) == 1
    row = result[0]
    assert (row["type_id"], row["anchor"], row["start"], row["end"]) == ("keys_stamps_secure_handling", 3., 1., 5.)
    assert row["source_indices"] == [90, 30, 150]
    assert row["frame_indices"] == [1, 0, 2]
    assert row["anchor_frame"] == 1 and row["observed_action"] == before["scenario_observations"][0]["visible_evidence"]
    assert row["state"] == state and row["confidence"] == .71
    assert row["confirmed"] is False and row["requires_external_context"] is True


def test_missing_field_does_not_invent_negative_or_positive_targets():
    assert collect_scenario_candidates({}, frames(), registry()) == []
    assert collect_scenario_candidates(response(), frames(), registry()) == []


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), True, "0.7", -.01, 1.01])
def test_invalid_confidence_is_rejected_before_any_downstream_candidate(confidence):
    with pytest.raises(AnalysisError, match="invalid fields or source references"):
        collect_scenario_candidates(response(observation(confidence=confidence)), frames(), registry())
    with pytest.raises(AnalysisError):
        _validate_json_schema([observation(confidence=confidence)], scenario_schema(registry(), 3))


@pytest.mark.parametrize("changes", [
    {"item_id": "a-private-case-name"}, {"state": "confirmed"}, {"anchor_frame": True},
    {"anchor_frame": 3}, {"start_frame": -1}, {"visible_evidence": " "},
    {"visible_evidence": "a" * 181}, {"source_hash": "not-an-inference-input"},
])
def test_unknown_ids_and_invalid_or_extra_source_fields_fail_atomically(changes):
    valid = observation(item_id="personal_belongings_in_cash_unit")
    with pytest.raises(AnalysisError) as exc:
        collect_scenario_candidates(response(valid, observation(**changes)), frames(), registry())
    assert exc.value.code == "model_schema_invalid"


def test_duplicate_episodes_are_deduplicated_and_distinct_same_type_episodes_survive():
    rows = [observation(), observation(confidence=.9),
            observation(start_frame=2, anchor_frame=2, end_frame=2)]
    result = collect_scenario_candidates(response(*rows), frames(), registry())
    assert len(result) == 2 and result[0]["confidence"] == .9
    assert result[1]["start"] == result[1]["end"] == 5.


def test_excess_rows_and_conflicting_source_pts_are_rejected():
    too_many = [observation(item_id=identifier) for identifier in IDS[:7]]
    with pytest.raises(AnalysisError):
        collect_scenario_candidates(response(*too_many), frames(), registry())
    source = frames()
    source[2] = FrameRef(30, 5., Path("synthetic-conflict.jpg"))
    with pytest.raises(AnalysisError):
        collect_scenario_candidates(response(observation()), source, registry())


@pytest.mark.parametrize("changes", [
    {"anchor_frame": 0, "start_frame": 1, "end_frame": 2},
    {"anchor_frame": 2, "start_frame": 1, "end_frame": 0},
    {"anchor_frame": 1, "start_frame": 2, "end_frame": 0},
])
def test_schema_valid_reference_order_is_canonicalized_from_cited_source_frames(changes):
    row = observation(**changes)
    _validate_json_schema([row], scenario_schema(registry(), 3))
    answer = response(row)
    before = deepcopy(answer)
    candidate = collect_scenario_candidates(answer, frames(), registry())[0]
    assert answer == before
    assert (candidate["start"], candidate["end"]) == (1., 5.)
    assert candidate["anchor"] == frames()[changes["anchor_frame"]].timestamp
    assert set(candidate["source_indices"]) == {30, 90, 150}


@pytest.mark.parametrize("identifier,evidence", [
    ("previous_transaction_cash_exposed", "No cash-like bundle is visibly exposed on the workstation surface."),
    ("loose_cash_issued_without_observed_count", "No cash-like bundle is passed toward another person or client side."),
    ("cash_intake_without_client_presence", "No cash-like bundle is visibly accepted or counted by the person."),
    ("active_intake_departure", "Person remains seated at workstation, no departure motion observed."),
    ("active_intake_departure", "The person reaches toward a cabinet, not visibly leaving the workstation."),
    ("previous_transaction_cash_exposed", "The person is seated at the workstation, but no cash-like bundle is visibly exposed or handled."),
])
def test_explicitly_negative_scenario_rows_are_not_positive_proposals(identifier, evidence):
    assert scenario_absence_only(identifier, evidence)
    assert collect_scenario_candidates(response(observation(item_id=identifier, visible_evidence=evidence)), frames(), registry()) == []


@pytest.mark.parametrize("evidence", [
    "Cash is visibly counted without a customer present.",
    "No customer is visible while cash is visibly counted.",
    "A cash-like bundle is handled, but no client is visible nearby.",
    "Cash is placed on the counter, but no cash remains in the final frame.",
])
def test_negative_context_does_not_suppress_a_positive_cash_action(evidence):
    assert not scenario_absence_only("cash_intake_without_client_presence", evidence)
    assert len(collect_scenario_candidates(response(observation(item_id="cash_intake_without_client_presence", visible_evidence=evidence)), frames(), registry())) == 1


@pytest.mark.parametrize("text", ["No cash is visible.", "No departure observed.",
    "There are no keys visible; no stamps observed.", "Cash is not visible."])
def test_shared_absence_filter_recognizes_explicit_empty_observations(text):
    assert absence_only(text)


@pytest.mark.parametrize("text", ["No client visible while cash counted.",
    "Cash not stored and next client approaches.", "Cash is counted without a client.",
    "No cash visible; a folder is handled.", "No customer visible and cash is counted.", ""])
def test_shared_absence_filter_preserves_positive_and_mixed_actions(text):
    assert not absence_only(text)


@pytest.mark.parametrize("count", [0, 9, True, 1.0])
def test_screening_never_exceeds_existing_eight_frame_budget(count):
    with pytest.raises(ValueError):
        scenario_schema(registry(), count)


def test_new_curated_scenario_has_bounded_data_fallback():
    scenarios = {"new_visible_case": {"id": "new_visible_case", "label": "A visible new object", "disposition": "review_only"}}
    prompt = scenario_prompt(scenarios)
    assert "new_visible_case: Inspect visible cues for: A visible new object" in prompt
    with pytest.raises(ValueError):
        scenario_prompt({"items": [{"id": "duplicate"}, {"id": "duplicate"}]})


def test_registry_machine_cue_keys_do_not_replace_readable_physical_conditions():
    scenarios = registry()
    scenarios["keys_stamps_secure_handling"].update(
        visual_cues=["key_handling", "stamp_handling"],
        check_descriptions={"key_or_stamp_visually_supported": "A key-like object or stamp is visibly identifiable.",
                            "handling_or_placement_observed": "Visible handling or placement occurs."})
    prompt = scenario_prompt(scenarios)
    assert "A key-like object or stamp is visibly identifiable. Visible handling or placement occurs." in prompt
    assert "key_handling; stamp_handling" not in prompt


def test_real_registry_prompt_keeps_existing_multimodal_context_headroom():
    from worker.scenarios import load_scenarios
    scenarios = load_scenarios(ROOT)
    prompt = scenario_prompt(scenarios)
    assert len(scenarios) == 9 and len(prompt) < 4200
    for line in prompt.split("Checklist DATA:\n", 1)[1].splitlines():
        assert len(line.split(": ", 1)[1]) <= 220
