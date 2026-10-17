from copy import deepcopy
import json
from pathlib import Path

import pytest

from worker.ledger import ACTION_KINDS, ObservationLedger
from worker.video import AnalysisError, FrameRef


def frames(*times):
    return [FrameRef(round(t * 30), t, Path(f"synthetic-{i}.jpg")) for i, t in enumerate(times)]


def answer(kind="unknown", phase="unknown", facts=None, transaction_indices=None):
    return {"visibility": "clear", "visible_facts": facts or [],
            "transaction": {"kind": kind, "phase": phase,
                            "frame_indices": transaction_indices if transaction_indices is not None else []},
            "candidates": [{"ignored": "owned by the pipeline"}]}


def fact(kind="cash_received", indices=None, state="observed", note="Visible action"):
    return {"kind": kind, "state": state, "frame_indices": [0] if indices is None else indices, "note": note}


@pytest.mark.parametrize("indices", [[-1], [2], [True], [0.5], ["0"], [], None])
def test_invalid_action_references_fail_without_partial_add(indices):
    ledger = ObservationLedger()
    item = fact()
    item["frame_indices"] = indices
    with pytest.raises(AnalysisError) as error:
        ledger.add(0, frames(0.0, 1.0), answer(facts=[item]))
    assert error.value.code == "model_schema_invalid"
    assert ledger.as_dict()["windows"] == []


def test_invalid_transaction_reference_fails():
    with pytest.raises(AnalysisError) as error:
        ObservationLedger().add(0, frames(1.0), answer("intake", "active", transaction_indices=[1]))
    assert error.value.code == "model_schema_invalid"


@pytest.mark.parametrize("kind,phase", [("intake", "active"), ("unknown", "active"), ("payout", "unknown")])
def test_observed_transaction_information_requires_references(kind, phase):
    with pytest.raises(AnalysisError):
        ObservationLedger().add(0, frames(1.0), answer(kind, phase))


def test_unknown_transaction_does_not_inherit_previous_intake():
    ledger = ObservationLedger()
    ledger.add(0, frames(0.0, 1.0), answer("intake", "active", transaction_indices=[0]))
    ledger.add(1, frames(2.0, 3.0), answer())
    cues = ledger.as_dict()["transaction_cues"]
    assert [cue["transaction_kind"] for cue in cues] == ["intake", "unknown"]
    assert cues[-1]["state"] == "uncertain"
    assert cues[-1]["refs"] == []
    assert ledger.relevant_facts(2.0, 3.0) == []


def test_source_indices_and_timestamps_come_from_frames():
    ledger = ObservationLedger()
    supplied = [FrameRef(301, 10.033, Path("one.jpg")), FrameRef(330, 11.0, Path("two.jpg"))]
    ledger.add(7, supplied, answer(facts=[fact(indices=[1, 0, 1])]))
    stored = ledger.relevant_facts(10, 12)[0]
    assert stored["source_indices"] == [301, 330]
    assert stored["time_sec"] == 10.033
    assert stored["refs"] == [{"source_index": 301, "timestamp_sec": 10.033},
                               {"source_index": 330, "timestamp_sec": 11.0}]
    json.dumps(ledger.as_dict(), allow_nan=False)


def test_observed_boundaries_separate_tentative_episodes():
    ledger = ObservationLedger()
    ledger.add(0, frames(1.0, 3.0), answer(facts=[fact("transaction_start", [0]), fact("transaction_end", [1])]))
    ledger.add(1, frames(5.0, 7.0), answer(facts=[fact("transaction_start", [0]), fact("transaction_end", [1])]))
    assert ledger.episode_id(2) != ledger.episode_id(6)
    assert ledger.episode_id(4) not in {ledger.episode_id(2), ledger.episode_id(6)}
    assert ledger.episode_id(3) == ledger.episode_id(2)
    assert ledger.episode_id(2).startswith("tentative-")


def test_uncertain_boundary_does_not_create_episode_or_identity():
    ledger = ObservationLedger()
    ledger.add(0, frames(1.0), answer(facts=[fact("transaction_start", state="uncertain")]))
    assert ledger.episode_id(0) == ledger.episode_id(2) == "tentative-unbounded"
    data = ledger.as_dict()
    assert "person_id" not in data and "cash_id" not in data
    assert data["facts"][0]["state"] == "uncertain"


def test_no_counting_omission_or_face_identity_is_inferred():
    ledger = ObservationLedger()
    ledger.add(0, frames(0.0, 1.0), answer(facts=[fact("cash_handed_over")]))
    kinds = [item["kind"] for item in ledger.as_dict()["facts"]]
    assert kinds == ["cash_handed_over"]
    assert all("face" not in kind.split("_") and "identity" not in kind.split("_")
               and "not_counted" not in kind for kind in ACTION_KINDS)
    with pytest.raises(AnalysisError):
        ledger.add(1, frames(2.0), answer(facts=[fact("cash_not_counted")]))


def test_context_limit_does_not_discard_private_ledger_facts():
    ledger = ObservationLedger()
    for window in range(12):
        ledger.add(window, frames(float(window)), answer(facts=[fact()]))
    selected = ledger.relevant_facts(0, 12, limit=3)
    assert len(selected) == 3
    assert selected[0]["time_sec"] == 0 and selected[-1]["time_sec"] == 11
    assert len(ledger.as_dict()["facts"]) == 12
    selected[0]["state"] = "changed outside ledger"
    assert ledger.as_dict()["facts"][0]["state"] == "observed"


def test_readding_window_replaces_atomically_and_note_length_is_bounded():
    ledger = ObservationLedger()
    ledger.add(0, frames(1.0), answer(facts=[fact()]))
    previous = deepcopy(ledger.as_dict())
    with pytest.raises(AnalysisError):
        ledger.add(0, frames(2.0), answer(facts=[fact(note="x" * 321)]))
    assert ledger.as_dict() == previous
    ledger.add(0, frames(2.0), answer())
    assert len(ledger.as_dict()["windows"]) == 1
    assert ledger.as_dict()["facts"] == []


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_source_timestamp_fails(stamp):
    with pytest.raises(AnalysisError):
        ObservationLedger().add(0, [FrameRef(0, stamp, Path("unused.jpg"))], answer())
