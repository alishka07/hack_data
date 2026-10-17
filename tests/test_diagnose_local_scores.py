"""Diagnostic boundaries using synthetic frames and mocked inference only."""
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from scripts.diagnose_local_scores import (FOCUSED_CHECKS, combined_rejection_counts, focused_prompt, focused_schema,
                                          run_private, selected_windows, validate_focused)
from worker.runtime import _validate_json_schema
from worker.video import AnalysisError, Budget, FrameRef


def test_partial_confidence_survives_unknown_conditions_and_binds_source_frames():
    conditions = FOCUSED_CHECKS[3][3]
    answer = {"confidence": .35, "confidence_quality": "weak", "applicability": "unknown", "frame_indices": [1, 1],
              "conditions": dict(zip(conditions, ["yes", "unknown", "unknown"]))}
    frames = [FrameRef(15, .5, Path("first.jpg")), FrameRef(30, 1., Path("second.jpg"))]
    _validate_json_schema(answer, focused_schema(len(frames), conditions))
    actual = validate_focused(answer, frames, conditions)
    assert actual["confidence"] == .35
    assert actual["evidence_refs"] == [{"source_index": 30, "timestamp_sec": 1.}]
    assert "unknown condition does not automatically force confidence to zero" in focused_prompt(FOCUSED_CHECKS[3][2], "procedure", conditions)


@pytest.mark.parametrize("confidence,indices", [(.35, []), (.35, [2]), (.35, [True]), (float("nan"), [0]), (True, [0])])
def test_invalid_source_references_or_confidence_are_not_accepted(confidence, indices):
    conditions = FOCUSED_CHECKS[0][3]
    answer = {"confidence": confidence, "confidence_quality": "weak", "applicability": "unknown", "frame_indices": indices,
              "conditions": {name: "unknown" for name in conditions}}
    with pytest.raises(AnalysisError) as error:
        validate_focused(answer, [FrameRef(0, 0., Path("first.jpg"))], conditions)
    assert error.value.code == "diagnostic_response_invalid"


def test_focused_private_probe_outputs_no_private_prompt_reply_or_filename(monkeypatch, tmp_path, capsys):
    import scripts.diagnose_local_scores as diagnostic
    info = SimpleNamespace(duration=18., width=464, height=848)
    monkeypatch.setattr(diagnostic, "select_video", lambda *args: info)
    frames = [FrameRef(i * 30, float(i), tmp_path / f"private-clip-secret-{i}.jpg") for i in range(3)]
    class Cache:
        def __init__(self, *args):
            pass
        def get(self, targets, height):
            assert len(targets) == 3 and height == 0
            return frames
    monkeypatch.setattr(diagnostic, "FrameCache", Cache)
    calls = []
    class Runtime:
        def ask(self, prompt, frames, schema, budget, max_tokens):
            calls.append(prompt)
            return {"confidence": .35, "confidence_quality": "weak", "applicability": "unknown", "frame_indices": [0],
                    "conditions": {name: "unknown" for name in schema["properties"]["conditions"]["properties"]}}
    args = SimpleNamespace(selection=tmp_path / "private-labels.xlsx", longest=False, windows=1, focused_only=True)
    result = run_private(Runtime(), args, tmp_path, Budget(time.time() + 15))
    assert len(calls) == 6
    assert result["records"][-1]["procedure_nonzero_count"] == 3
    output = capsys.readouterr().out
    assert "private-clip-secret" not in output and "private-labels" not in output
    assert "intake_banknotes_counted" not in output and "0.35" not in output
    assert len(list(tmp_path.glob("window-*-focused-*.json"))) == 6


def test_window_selection_covers_first_middle_and_end_without_duplicate_windows():
    assert len(selected_windows(5., 3)) == 1
    windows = selected_windows(142., 3)
    assert len(windows) == 3 and windows[0][0] == 0 and windows[-1][-1] == 142.


def test_combined_rejection_diagnostic_distinguishes_partial_confidence_from_bad_references():
    supplied = {"catalog_assessments": [
        {"item_id": "active_intake_departure", "assessment": "unresolved", "confidence": .35, "frame_indices": [0]},
        {"item_id": "intake_banknotes_counted", "assessment": "unresolved", "confidence": .55, "frame_indices": [1]},
    ]}
    assert combined_rejection_counts(supplied) == {
        "duplicate_item_count": 0, "scoring_kind_state_mismatch_count": 0,
        "unresolved_with_nonzero_confidence_count": 2, "out_of_range_refs_count": 0, "empty_support_refs_count": 0,
    }
