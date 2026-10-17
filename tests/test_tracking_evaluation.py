"""Offline comparison diagnostics must preserve unknowns and source clocks."""
import json

import pytest

from scripts.evaluate_tracking import load_timeline, summarize_assignments


def test_reused_tracker_id_records_unknown_gap_instead_of_continuous_visibility():
    box = {"bbox": [.1, .1, .4, .9], "confidence": .8}
    frames = [(0, 0., [box]), (6, .2, []), (12, .4, [box])]
    result = summarize_assignments([["track-1"], [], ["track-1"]], frames)
    assert result["unique_track_ids"] == 1
    assert result["contiguous_observed_track_segments"] == 2
    assert result["resumed_ids_across_unobserved_samples"] == 1
    assert result["unobserved_samples_inside_resumed_ids"] == 1
    assert result["max_resumption_span_sec"] == .4
    assert "unknown visibility" in result["gap_semantics"]
    assert not any(key in result for key in ("accuracy", "identity_switches", "HOTA", "IDF1"))


def test_replay_rejects_missing_sampled_frames_or_conflicting_detection_pts(tmp_path):
    report = {"coverage": {"first_timestamp_sec": 0., "frames_analyzed": 3},
              "motion": [{"source_index": 6, "timestamp_sec": .2}], "detections": []}
    path = tmp_path / "perception.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete sampled motion timeline"):
        load_timeline(path)
    report["motion"].append({"source_index": 12, "timestamp_sec": .4})
    report["detections"] = [{"kind": "person", "source_index": 12, "timestamp_sec": .6,
                              "bbox": [.1, .1, .4, .9], "confidence": .8}]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not bound"):
        load_timeline(path)
