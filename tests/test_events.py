"""Temporal proposal recall and uncertainty on synthetic geometry."""
from dataclasses import replace
from pathlib import Path

import pytest

from worker.events import EventConfig, propose_events
from worker.video import AnalysisError, VideoInfo


def setup_report(end=12, step=.2):
    pts = tuple(round(i * step, 6) for i in range(round(end / step) + 1))
    info = VideoInfo(Path("synthetic.mov"), end, 320, 200, pts, 50.)
    report = {"detections": [], "motion": [], "poses": [], "scene": {"zones": []},
              "frame_samples": [{"source_index": i, "timestamp_sec": t, "person_detection_available": True}
                                for i, t in enumerate(pts)]}
    return info, report


def zone(report, name, box):
    report["scene"]["zones"].append({"name": name, "bbox": box, "source_indices": [0], "status": "source_linked_hypothesis"})


def detection(info, index, *, kind="person", box=(.1, .1, .3, .9), track="track-1", continuity="adjacent_detection"):
    return {"id": f"{kind}-{index}-{track}", "kind": kind, "source_index": index, "timestamp_sec": info.pts[index],
            "bbox": list(box), "confidence": .8, "track_id": track, "continuity": continuity}


def test_zone_entry_dwell_and_exit_preserve_pts_and_no_legal_inference():
    info, report = setup_report(5)
    zone(report, "cash_zone", [.4, 0., 1., 1.])
    for i, timestamp in enumerate(info.pts):
        report["detections"].append(detection(info, i, box=(.5, .1, .7, .9) if 1 <= timestamp < 4 else (.1, .1, .3, .9)))
    result = propose_events(report, info, replace(EventConfig(), dwell_sec=2.))
    assert [row["reason"] for row in result["proposals"]] == ["zone_entry", "zone_dwell", "zone_exit"]
    assert result["proposals"][1]["sampled_duration_sec"] == 2.
    assert all(row["status"] == "unconfirmed_trigger" for row in result["proposals"])
    assert all(ref["timestamp_sec"] == info.pts[ref["source_index"]] for row in result["proposals"] for ref in row["refs"])


def test_missing_person_detection_breaks_dwell_even_if_track_id_reappears():
    info, report = setup_report(4)
    zone(report, "counter", [0., 0., 1., 1.])
    report["detections"] = [detection(info, i, continuity="new" if t == 2.2 else "adjacent_detection")
                            for i, t in enumerate(info.pts) if t != 2.]
    result = propose_events(report, info, replace(EventConfig(), dwell_sec=3.))
    assert not any(row["reason"] == "zone_dwell" for row in result["proposals"])


def test_cut_invalidates_old_zone_and_occupancy_run():
    info, report = setup_report(4)
    zone(report, "cash_zone", [0., 0., 1., 1.])
    report["motion"] = [{"timestamp_sec": 1., "scene_change_suspected": True}]
    report["detections"] = [detection(info, i) for i in range(len(info.pts))]
    result = propose_events(report, info, replace(EventConfig(), dwell_sec=2.))
    assert not result["proposals"]
    assert all(row["timestamp_sec"] < 1. for row in result["occupancy"])


def test_empty_workstation_is_only_candidate_and_multiple_people_have_no_roles():
    info, report = setup_report(5)
    zone(report, "workstation", [0., 0., 1., 1.])
    zone(report, "cash_zone", [0., 0., 1., 1.])
    report["detections"] = [detection(info, i, track=track) for i, t in enumerate(info.pts) if t <= 1.
                            for track in ("a", "b")]
    result = propose_events(report, info)
    assert {row["reason"] for row in result["proposals"]} == {"multiple_people_in_cash_zone", "cashier_absence_candidate"}
    assert all(row["visibility"] == "unknown" and row["role"] == "unknown" for row in result["occupancy"])
    report["frame_samples"] = []
    assert not any(row["reason"] == "cashier_absence_candidate" for row in propose_events(report, info)["proposals"])


def test_wrist_near_object_uses_exact_frame_and_not_hidden_keypoint():
    info, report = setup_report(1)
    report["detections"] = [detection(info, 0, kind="cash", box=(.5, .5, .6, .6))]
    report["poses"] = [{"source_index": 0, "timestamp_sec": 0., "track_id": "a", "keypoints": [
        {"name": "left_wrist", "status": "supported", "x": .51, "y": .51},
        {"name": "right_wrist", "status": "not_visible", "x": .51, "y": .51}]}]
    result = propose_events(report, info)
    assert len(result["interactions"]) == 1 and result["interactions"][0]["contact"] == "not_visible"
    report["poses"][0].update(source_index=1, timestamp_sec=.2)
    assert not propose_events(report, info)["interactions"]


def test_unattended_needs_repeated_object_and_same_frame_person_processing():
    info, report = setup_report(12, 1.)
    report["detections"] = [detection(info, i, kind="bag") for i in (0, 5, 10)]
    result = propose_events(report, info)
    assert [row["reason"] for row in result["proposals"]] == ["object_unattended_candidate"]
    report["detections"].append(detection(info, 0))
    assert not propose_events(report, info)["proposals"]  # attended t=0 cannot start a 10-second unattended run
    report["detections"].pop()
    report["detections"].append(detection(info, 5))
    assert not propose_events(report, info)["proposals"]
    report["detections"].pop()
    report["object_samples"] = [{"source_index": i, "timestamp_sec": info.pts[i]} for i in (0, 3, 5, 10)]
    assert not propose_events(report, info)["proposals"]  # explicitly missed object breaks persistence


def test_foreign_frame_and_invalid_configuration_fail(monkeypatch):
    info, report = setup_report(1)
    report["detections"] = [{**detection(info, 0), "timestamp_sec": .1}]
    with pytest.raises(AnalysisError, match="presentation timestamps"):
        propose_events(report, info)
    monkeypatch.setenv("FORTE_EVENT_DWELL_SEC", "nan")
    with pytest.raises(AnalysisError, match="finite and positive"):
        EventConfig.from_env()
