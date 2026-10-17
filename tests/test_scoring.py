"""Deterministic score/state checks without labels, private media or inference."""
from copy import deepcopy

import pytest

from worker.scoring import (create_initial_assessments, mark_event, mark_not_applicable,
                            mark_procedure, update_event_confidence,
                            update_procedure_confidence)


CATALOG = {"version": "synthetic", "items": [
    {"id": "synthetic_event", "label": "Synthetic observable event", "scoring_kind": "event_occurrence", "initial_score": 0},
    {"id": "synthetic_procedure", "label": "Synthetic required procedure", "scoring_kind": "required_procedure", "initial_score": 100},
]}
REFS = [{"source_index": 5, "timestamp_sec": 2.5}, {"filename": "event-001-00.jpg", "timestamp_sec": 3.0}]


def test_initial_scores_are_requested_states_not_findings_or_probability():
    event, procedure = create_initial_assessments(CATALOG)
    assert (event["score"], procedure["score"]) == (0, 100)
    assert event["label"] == CATALOG["items"][0]["label"]
    for item in (event, procedure):
        assert item["assessment"] == "unassessed"
        assert item["applicability"] == "unknown"
        assert item["evidence_refs"] == []
        assert item["confirmed_finding"] is False
        assert item["model_confidence"] is None and item["calibrated"] is False


def test_event_requires_all_positive_support_and_separate_legal_confirmation():
    event = create_initial_assessments(CATALOG)[0]
    observed = mark_event(event, supported=True, observable=True, applicability="applicable", evidence_refs=REFS, duration_sec=10)
    assert observed["score"] == 100 and observed["assessment"] == "observed"
    assert observed["confirmed_finding"] is False
    confirmed = mark_event(event, supported=True, observable=True, applicability="applicable", evidence_refs=REFS, duration_sec=10, confirmed_finding=True)
    assert confirmed["confirmed_finding"] is True
    for settings in ({"supported": False}, {"observable": False}, {"evidence_refs": []}):
        params = {"supported": True, "observable": True, "applicability": "applicable", "evidence_refs": REFS, "duration_sec": 10, **settings}
        unresolved = mark_event(event, **params)
        assert unresolved["score"] == 0 and unresolved["assessment"] == "unresolved"
        with pytest.raises(ValueError):
            mark_event(event, **params, confirmed_finding=True)


def test_procedure_does_not_complete_or_become_violation_from_missing_evidence():
    procedure = create_initial_assessments(CATALOG)[1]
    completed = mark_procedure(procedure, completed=True, observable=True, applicability="applicable", evidence_refs=REFS, duration_sec=10)
    assert completed["score"] == 0 and completed["assessment"] == "completed"
    for settings in ({"completed": False}, {"observable": False}, {"evidence_refs": []}):
        result = mark_procedure(procedure, **{"completed": True, "observable": True, "applicability": "applicable", "evidence_refs": REFS, "duration_sec": 10, **settings})
        assert result["score"] == 100 and result["assessment"] == "unresolved"
        assert result["confirmed_finding"] is False


def test_not_applicable_is_exempt_null_with_initial_state_retained():
    procedure = create_initial_assessments(CATALOG)[1]
    exempt = mark_not_applicable(procedure, evidence_refs=REFS, duration_sec=10)
    assert exempt["score"] is None and exempt["initial_score"] == 100
    assert exempt["applicability"] == "not_applicable" and exempt["confirmed_finding"] is False
    with pytest.raises(ValueError):
        mark_not_applicable(procedure, evidence_refs=[], duration_sec=10)


@pytest.mark.parametrize("reference", [
    {"source_index": 1, "timestamp_sec": -1}, {"source_index": 1, "timestamp_sec": 11},
    {"source_index": 1, "timestamp_sec": float("nan")}, {"source_index": 1, "timestamp_sec": float("inf")},
    {"source_index": True, "timestamp_sec": 1}, {"source_index": -1, "timestamp_sec": 1},
    {"source_index": 1, "timestamp_sec": True}, {"timestamp_sec": 1},
    {"filename": "../outside.jpg", "timestamp_sec": 1}, {"frame_id": "", "timestamp_sec": 1},
])
def test_source_references_cannot_be_missing_invalid_or_out_of_bounds(reference):
    with pytest.raises(ValueError):
        mark_event(create_initial_assessments(CATALOG)[0], supported=True, observable=True,
                   applicability="applicable", evidence_refs=[reference], duration_sec=10)


def test_evidence_needs_source_duration_and_consistent_frame_identity():
    event = create_initial_assessments(CATALOG)[0]
    with pytest.raises(ValueError):
        mark_event(event, supported=True, observable=True, evidence_refs=REFS)
    with pytest.raises(ValueError):
        mark_event(event, supported=True, observable=True, duration_sec=10, evidence_refs=[
            {"source_index": 1, "timestamp_sec": 1}, {"source_index": 1, "timestamp_sec": 2}])
    with pytest.raises(ValueError):
        mark_event(event, supported=0.95, observable=True)


def test_helpers_are_pure_and_cannot_apply_wrong_item_kind():
    event, procedure = create_initial_assessments(CATALOG)
    original = deepcopy(event)
    references = deepcopy(REFS)
    result = mark_event(event, supported=True, observable=True, applicability="applicable", evidence_refs=references, duration_sec=10)
    references[0]["timestamp_sec"] = 9
    assert event == original
    assert result["evidence_refs"][0]["timestamp_sec"] == 2.5
    with pytest.raises(ValueError):
        mark_event(procedure, supported=True, observable=True)
    with pytest.raises(ValueError):
        mark_procedure(event, completed=True, observable=True)


def test_catalog_rejects_duplicate_ids_alias_conflicts_and_wrong_initial_scores():
    with pytest.raises(ValueError):
        create_initial_assessments([CATALOG["items"][0], CATALOG["items"][0]])
    for item in ({**CATALOG["items"][0], "initial_score": 100},
                 {**CATALOG["items"][1], "initial_score": 0},
                 {**CATALOG["items"][0], "item_id": "different"}):
        with pytest.raises(ValueError):
            create_initial_assessments([item])


def test_later_unknown_chunks_cannot_erase_an_observed_event_or_legal_confirmation():
    event = mark_event(create_initial_assessments(CATALOG)[0], supported=True, observable=True,
                       applicability="applicable", evidence_refs=REFS, duration_sec=10, confirmed_finding=True)
    later = mark_event(event, supported=False, observable=False)
    assert later["score"] == 100 and later["assessment"] == "observed"
    assert later["applicability"] == "applicable" and later["confirmed_finding"] is True
    assert later["evidence_refs"] == event["evidence_refs"]
    assert event["source_duration_sec"] == later["source_duration_sec"] == 10
    assert create_initial_assessments(CATALOG)[0]["score"] == 0


def test_later_unknown_chunks_cannot_erase_completion_but_new_episode_resets_it():
    procedure = mark_procedure(create_initial_assessments(CATALOG)[1], completed=True, observable=True,
                               applicability="applicable", evidence_refs=REFS, duration_sec=10)
    later = mark_procedure(procedure, completed=False, observable=False)
    assert later["score"] == 0 and later["assessment"] == "completed"
    assert later["applicability"] == "applicable" and later["confirmed_finding"] is False
    assert later["evidence_refs"] == procedure["evidence_refs"]
    assert create_initial_assessments(CATALOG)[1]["score"] == 100


def test_positive_refs_merge_without_aliasing_but_unknown_refs_do_not_become_support():
    event = mark_event(create_initial_assessments(CATALOG)[0], supported=True, observable=True,
                       applicability="applicable", evidence_refs=REFS, duration_sec=10)
    new_refs = [{"source_index": 8, "timestamp_sec": 4}]
    merged = mark_event(event, supported=True, observable=True, applicability="applicable", evidence_refs=new_refs)
    assert len(merged["evidence_refs"]) == 3 and len(event["evidence_refs"]) == 2
    unknown = mark_event(merged, supported=False, observable=False,
                         evidence_refs=[{"source_index": 9, "timestamp_sec": 5}])
    assert unknown["evidence_refs"] == merged["evidence_refs"]
    with pytest.raises(ValueError, match="duration changed"):
        mark_event(event, supported=False, observable=False, duration_sec=12)
    with pytest.raises(ValueError, match="conflicting timestamps"):
        mark_event(event, supported=True, observable=True, applicability="applicable",
                   evidence_refs=[{"source_index": 5, "timestamp_sec": 8}])


@pytest.mark.parametrize("index", [0, 1])
def test_exemption_stays_exempt_until_a_fresh_assessment(index):
    initial = create_initial_assessments(CATALOG)[index]
    exempt = mark_not_applicable(initial, evidence_refs=REFS, duration_sec=10)
    if index == 0:
        later = mark_event(exempt, supported=True, observable=True, applicability="applicable", evidence_refs=REFS)
    else:
        later = mark_procedure(exempt, completed=True, observable=True, applicability="applicable", evidence_refs=REFS)
    assert later["score"] is None and later["applicability"] == "not_applicable"
    assert later["evidence_refs"] == exempt["evidence_refs"]
    assert create_initial_assessments(CATALOG)[index]["score"] == initial["initial_score"]


def test_event_confidence_rises_continuously_and_unknown_cannot_erase_evidence():
    initial = create_initial_assessments(CATALOG)[0]
    first = update_event_confidence(initial, confidence=0.42, observable=True,
                                    evidence_refs=REFS, duration_sec=10)
    second = update_event_confidence(first, confidence=0.81, observable=True,
                                     applicability="applicable", evidence_refs=REFS)
    unknown = update_event_confidence(second, confidence=0.99, observable=False)
    assert [item["score"] for item in (initial, first, second, unknown)] == [0, 42, 81, 81]
    assert first["applicability"] == "unknown" and first["assessment"] == "observed"
    assert second["model_confidence"] == unknown["model_confidence"] == 0.81
    assert unknown["applicability"] == "applicable"
    assert all(item["confirmed_finding"] is False for item in (initial, first, second, unknown))
    assert first["evidence_refs"] == second["evidence_refs"] == unknown["evidence_refs"]


def test_procedure_confidence_falls_only_with_correct_completion_evidence():
    initial = create_initial_assessments(CATALOG)[1]
    first = update_procedure_confidence(initial, confidence=0.65, observable=True,
                                        evidence_refs=REFS, duration_sec=10)
    second = update_procedure_confidence(first, confidence=0.91, observable=True,
                                         applicability="applicable", evidence_refs=REFS)
    unknown = update_procedure_confidence(second, confidence=1.0, observable=False)
    assert [item["score"] for item in (initial, first, second, unknown)] == [100, 35, 9, 9]
    assert first["applicability"] == "unknown"
    assert first["assessment"] == second["assessment"] == unknown["assessment"] == "completion_evidence"
    assert second["model_confidence"] == unknown["model_confidence"] == 0.91
    completed = update_procedure_confidence(second, confidence=1.0, observable=True, evidence_refs=REFS)
    assert completed["score"] == 0 and completed["assessment"] == "completed"
    assert completed["confirmed_finding"] is False
    assert create_initial_assessments(CATALOG)[1]["score"] == 100


@pytest.mark.parametrize("index,updater,confidence,expected", [
    (0, update_event_confidence, 0.42, 42),
    (1, update_procedure_confidence, 0.65, 35),
])
def test_repeated_correlated_evidence_does_not_accumulate(index, updater, confidence, expected):
    initial = create_initial_assessments(CATALOG)[index]
    result = updater(initial, confidence=confidence, observable=True, evidence_refs=REFS, duration_sec=10)
    for _ in range(10):
        result = updater(result, confidence=confidence, observable=True, evidence_refs=REFS)
    assert result["score"] == expected
    assert len(result["evidence_refs"]) == len(REFS)
    weaker = updater(result, confidence=0.2, observable=True,
                     evidence_refs=[{"source_index": 10, "timestamp_sec": 6}])
    assert weaker["score"] == expected
    assert weaker["model_confidence"] == confidence


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1, True, None, "0.5"])
@pytest.mark.parametrize("index,updater", [(0, update_event_confidence), (1, update_procedure_confidence)])
def test_confidence_is_finite_bounded_and_not_a_boolean(index, updater, confidence):
    with pytest.raises(ValueError, match="confidence"):
        updater(create_initial_assessments(CATALOG)[index], confidence=confidence,
                observable=True, evidence_refs=REFS, duration_sec=10)


def test_partial_score_and_legal_confirmation_are_independent():
    event = create_initial_assessments(CATALOG)[0]
    with pytest.raises(ValueError, match="applicable"):
        update_event_confidence(event, confidence=0.81, observable=True,
                                evidence_refs=REFS, duration_sec=10, confirmed_finding=True)
    confirmed = update_event_confidence(event, confidence=0.81, observable=True,
                                        applicability="applicable", evidence_refs=REFS,
                                        duration_sec=10, confirmed_finding=True)
    assert confirmed["score"] == 81 and confirmed["confirmed_finding"] is True
    later = update_event_confidence(confirmed, confidence=0.0, observable=False)
    assert later["score"] == 81 and later["confirmed_finding"] is True


@pytest.mark.parametrize("index,updater", [(0, update_event_confidence), (1, update_procedure_confidence)])
def test_confidence_without_positive_visible_evidence_leaves_initial_baseline(index, updater):
    initial = create_initial_assessments(CATALOG)[index]
    for params in ({"confidence": 0.9, "observable": False, "evidence_refs": REFS},
                   {"confidence": 0.9, "observable": True, "evidence_refs": []},
                   {"confidence": 0.0, "observable": True, "evidence_refs": REFS}):
        result = updater(initial, **params, duration_sec=10)
        assert result["score"] == initial["initial_score"]
        assert result["model_confidence"] is None
        assert result["evidence_refs"] == []
        assert result["assessment"] == "unresolved"


def test_confidence_state_keeps_bounded_source_references_without_score_inflation():
    event = create_initial_assessments(CATALOG)[0]
    for index in range(40):
        event = update_event_confidence(event, confidence=0.5, observable=True,
                                        evidence_refs=[{"source_index": index, "timestamp_sec": index / 10}],
                                        duration_sec=10)
    assert event["score"] == 50 and event["model_confidence"] == 0.5
    assert len(event["evidence_refs"]) == 32
    # An earlier source frame can carry the new strongest evidence, and must
    # survive a full evidence buffer even though its timestamp sorts first.
    stronger = update_event_confidence(event, confidence=0.9, observable=True,
                                       evidence_refs=[{"frame_id": "strongest", "timestamp_sec": 0}],
                                       duration_sec=10)
    assert stronger["score"] == 90 and len(stronger["evidence_refs"]) == 32
    assert any(reference.get("frame_id") == "strongest" for reference in stronger["evidence_refs"])
    with pytest.raises(ValueError, match="conflicting timestamps"):
        update_event_confidence(stronger, confidence=0.1, observable=True,
                                evidence_refs=[{"frame_id": "strongest", "timestamp_sec": 2}])


def test_confidence_helpers_reject_tampered_scores_confidences_and_calibration_claims():
    initial = create_initial_assessments(CATALOG)[0]
    for tampered in ({**initial, "score": float("nan")},
                     {**initial, "score": 101},
                     {**initial, "model_confidence": 0.5},
                     {**initial, "calibrated": True}):
        with pytest.raises(ValueError):
            update_event_confidence(tampered, confidence=0.0, observable=False)
