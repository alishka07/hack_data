"""Synthetic confidence/reference/scope checks; no private media or inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from worker.catalog import clip_assessments, load_catalog
from worker.confidence import (OBSERVABLE_CHECKS, catalog_observation_schema,
                               collect_confidence_observations, confidence_prompt,
                               scope_confidence_observations)
from worker.ledger import ObservationLedger
from worker.runtime import _validate_json_schema
from worker.video import AnalysisError, FrameRef


ROOT = Path(__file__).resolve().parents[1]
EVENT = "active_intake_departure"
PROCEDURE = "intake_banknotes_counted"


def frames(*times):
    return [FrameRef(round(t * 30), t, Path(f"synthetic-{i}.jpg")) for i, t in enumerate(times)]


def observation(identifier=EVENT, state="event_evidence", confidence=.62, indices=None):
    return {"item_id": identifier, "assessment": state, "confidence": confidence,
            "applicability": "applicable", "frame_indices": [0] if indices is None else indices}


def response(*items, visibility="clear"):
    return {"visibility": visibility, "catalog_assessments": list(items)}


def test_thirteen_visible_catalog_checks_enter_schema_and_compact_prompt():
    catalog = json.loads((ROOT / "rules/catalog.json").read_text(encoding="utf-8"))
    visible = {item["id"] for item in catalog["items"] if item["automation"] != "manual"}
    schema = catalog_observation_schema(8)
    assert set(OBSERVABLE_CHECKS) == visible and len(visible) == 13
    assert set(schema["items"]["properties"]["item_id"]["enum"]) == visible
    prompt = confidence_prompt(catalog)
    assert all(identifier in prompt for identifier in visible)
    assert "UNCALIBRATED" in prompt and "weakest condition" in prompt
    assert "generic count gesture" in prompt and "unseen conditions stay unknown" in prompt
    assert len(prompt) < 4500
    with pytest.raises(ValueError):
        catalog_observation_schema(0)


def test_source_references_are_canonical_not_model_supplied_and_input_is_unchanged():
    supplied = response(observation(indices=[2, 0, 2]))
    previous = deepcopy(supplied)
    rows = collect_confidence_observations(supplied, frames(1., 2., 3.), 7)
    assert supplied == previous
    assert rows[0]["confidence"] == .62 and rows[0]["observable"] is True
    assert rows[0]["evidence_refs"] == [{"source_index": 30, "timestamp_sec": 1.},
                                         {"source_index": 90, "timestamp_sec": 3.}]
    assert (rows[0]["start_sec"], rows[0]["end_sec"], rows[0]["window_id"]) == (1., 3., 7)
    assert (rows[0]["window_start_sec"], rows[0]["window_end_sec"]) == (1., 3.)
    assert collect_confidence_observations({"visibility": "clear"}, frames(0.), 0) == []


def test_screen_departure_confidence_is_an_event_and_requires_source_references():
    identifier = "workstation_departure_screen_left_on"
    answer = response(observation(identifier, confidence=.73, indices=[0, 1, 2]))
    source = frames(1., 2., 3.)
    _validate_json_schema(answer["catalog_assessments"], catalog_observation_schema(3))
    collected = collect_confidence_observations(answer, source, 0)
    assert collected[0]["item_id"] == identifier and collected[0]["assessment"] == "event_evidence"
    scoped = scope_confidence_observations(collected, ObservationLedger())
    rows, _ = clip_assessments(load_catalog(ROOT), [], [], 4, confidence_observations=scoped)
    row = next(row for row in rows if row["item_id"] == identifier)
    assert row["score"] == 73 and row["confirmed_finding"] is False
    assert len(row["evidence_refs"]) == 3
    with pytest.raises(AnalysisError):
        collect_confidence_observations(response(observation(identifier, "completion_evidence")), source, 0)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), True, -.01, 1.01, "0.5"])
def test_confidence_is_finite_bounded_and_never_a_boolean(confidence):
    supplied = response(observation(confidence=confidence))
    with pytest.raises(AnalysisError):
        collect_confidence_observations(supplied, frames(0.), 0)
    with pytest.raises(AnalysisError):
        _validate_json_schema(supplied["catalog_assessments"], catalog_observation_schema(1))


@pytest.mark.parametrize("indices", [[], [-1], [1], [True], [0.0], None, [0, 0, 0, 0]])
def test_evidence_must_reference_one_to_three_supplied_frames(indices):
    item = observation()
    item["frame_indices"] = indices
    with pytest.raises(AnalysisError) as error:
        collect_confidence_observations(response(item), frames(0.), 0)
    assert error.value.code == "model_schema_invalid"
    assert "0.62" not in str(error.value)


@pytest.mark.parametrize("item", [
    observation(EVENT, "completion_evidence"), observation(PROCEDURE, "event_evidence"),
    observation("not_in_catalog"),
    {**observation(), "private_description": "synthetic ignored content"},
    {**observation(), "applicability": "not_applicable"},
])
def test_wrong_kind_unknown_id_and_extra_fields_reject(item):
    with pytest.raises(AnalysisError):
        collect_confidence_observations(response(item), frames(0.), 0)


def test_duplicate_item_is_rejected_atomically_and_no_more_than_six_items_allowed():
    with pytest.raises(AnalysisError):
        collect_confidence_observations(response(observation(), observation()), frames(0.), 0)
    with pytest.raises(AnalysisError):
        collect_confidence_observations(response(*(observation() for _ in range(7))), frames(0.), 0)


def test_partial_visibility_preserves_estimate_unusable_never_observable():
    item = observation(PROCEDURE, "completion_evidence", .83)
    partial = collect_confidence_observations(response(item, visibility="partial"), frames(0.), 0)[0]
    assert partial["observable"] is True and partial["confidence"] == .83
    unusable = collect_confidence_observations(response(item, visibility="unusable"), frames(0.), 0)[0]
    assert unusable["observable"] is False
    unresolved = collect_confidence_observations(response(observation(state="unresolved", confidence=0)), frames(0.), 0)[0]
    assert unresolved["confidence"] == 0


def test_uncertain_nonzero_confidence_is_preserved_as_partial_support():
    answer = response(observation(PROCEDURE, "unresolved", .35), observation(EVENT, "partial_support", .68))
    rows = collect_confidence_observations(answer, frames(0.), 0)
    assert len(rows) == 2
    assert rows[0]["assessment"] == "partial_support" and rows[0]["model_assessment"] == "unresolved"
    assert rows[0]["confidence"] == .35 and rows[1]["confidence"] == .68
    assert all(row["observable"] and row["evidence_refs"] for row in rows)


@pytest.mark.parametrize("identifier", [EVENT, PROCEDURE])
@pytest.mark.parametrize("applicability", ["unknown", "applicable"])
def test_zero_partial_support_survives_schema_collection_scope_and_aggregation(identifier, applicability):
    answer = response({**observation(identifier, "partial_support", 0, indices=[2, 0, 2]),
                       "applicability": applicability})
    previous = deepcopy(answer)
    source = frames(1., 2., 3.)
    _validate_json_schema(answer["catalog_assessments"], catalog_observation_schema(len(source)))
    collected = collect_confidence_observations(answer, source, 7)
    assert answer == previous
    assert collected[0]["assessment"] == "unresolved"
    assert collected[0]["model_assessment"] == "partial_support"
    assert collected[0]["confidence"] == 0
    expected_refs = [{"source_index": 30, "timestamp_sec": 1.},
                     {"source_index": 90, "timestamp_sec": 3.}]
    assert collected[0]["evidence_refs"] == expected_refs

    ledger = ObservationLedger()
    if applicability == "applicable":
        ledger.add(0, frames(0.), ledger_answer([("transaction_start", 0)]))
    scoped = scope_confidence_observations(collected, ledger)
    assert scoped[0]["evidence_refs"] == expected_refs
    assert scoped[0]["model_assessment"] == "partial_support"
    assert scoped[0]["applicability"] == applicability
    rows, episodes = clip_assessments(load_catalog(ROOT), [], [], 4, confidence_observations=scoped)
    summary = next(row for row in rows if row["item_id"] == identifier)
    assert summary["score"] == summary["initial_score"]
    assert summary["assessment"] == "unresolved"
    assert summary["model_confidence"] is None
    assert summary["confirmed_finding"] is False
    if identifier == PROCEDURE and applicability == "unknown":
        assert episodes == []
        assert summary["unscoped_unknown_observation_count"] == 1
    else:
        assert len(episodes) == 1
        assert episodes[0]["score"] == summary["initial_score"]


def ledger_answer(boundaries):
    return {"visibility": "clear", "visible_facts": [
        {"kind": kind, "state": "observed", "note": "Synthetic boundary", "frame_indices": [index]}
        for kind, index in boundaries], "transaction": {"kind": "unknown", "phase": "unknown", "frame_indices": []}}


def test_scoping_happens_after_all_boundaries_and_never_changes_input():
    ledger = ObservationLedger()
    ledger.add(0, frames(1., 4., 5., 8.), ledger_answer([
        ("transaction_start", 0), ("transaction_end", 1),
        ("transaction_start", 2), ("transaction_end", 3)]))
    first = collect_confidence_observations(response(observation()), frames(2.), 1)
    second = collect_confidence_observations(response(observation(PROCEDURE, "completion_evidence", .83)), frames(6.), 2)
    previous = deepcopy(first + second)
    scoped = scope_confidence_observations(first + second, ledger)
    assert first + second == previous
    assert scoped[0]["episode_id"] != scoped[1]["episode_id"]
    assert all(row["scope_certainty"] == "observed_start" for row in scoped)


def test_unbounded_after_end_and_cross_boundary_evidence_keep_separate_window_scopes():
    ledger = ObservationLedger()
    ledger.add(0, frames(1., 3.), ledger_answer([("transaction_start", 0), ("transaction_end", 1)]))
    before = collect_confidence_observations(response(observation()), frames(0.), 3)
    after = collect_confidence_observations(response(observation()), frames(4.), 4)
    crossing = collect_confidence_observations(response(observation(indices=[0, 1])), frames(2., 4.), 5)
    rows = scope_confidence_observations(before + after + crossing, ledger)
    assert [row["episode_id"] for row in rows] == ["unscoped-window-3", "unscoped-window-4", "unscoped-window-5"]
    assert all(row["scope_certainty"] == "window_only" for row in rows)
    assert rows[-1]["applicability"] == "unknown"


@pytest.mark.parametrize("source", [
    [FrameRef(-1, 1., Path("synthetic.jpg"))],
    [FrameRef(1, float("nan"), Path("synthetic.jpg"))],
    [FrameRef(1, 1., Path("synthetic.jpg")), FrameRef(1, 2., Path("synthetic.jpg"))],
])
def test_invalid_source_metadata_never_creates_confidence_support(source):
    with pytest.raises(AnalysisError):
        collect_confidence_observations(response(observation()), source, 0)
