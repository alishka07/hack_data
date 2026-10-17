"""Empirical threshold isolation and original source evidence selection."""
import json

import cv2
import numpy as np
import pytest

from worker.fusion import ThresholdPolicy, fit_thresholds, joint_score, select_evidence
from worker.video import AnalysisError, FrameRef


def rows():
    return [{"rule_id": "a", "score": .8 if label else .3, "label": label, "reviewed": True,
             "group_id": f"{partition}-{index}", "event_id": f"{partition}-{index}", "partition": partition}
            for partition in ("calibration", "validation") for index, label in enumerate([True] * 5 + [False] * 5)]


def test_threshold_tunes_only_on_disjoint_groups_and_is_version_bound(tmp_path):
    artifact = fit_thresholds(rows(), model_id="exact-model", pipeline_version="4.0", minimum_examples=10)
    assert artifact["rules"]["a"]["status"] == "validated"
    assert artifact["rules"]["a"]["threshold"] == .8
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    policy = ThresholdPolicy("exact-model", "4.0", path)
    assert not policy.evaluate("a", {"confidence": .79}, {"confidence": .99})["accepted"]
    assert policy.evaluate("a", {"confidence": .9}, {"confidence": .9})["threshold_calibrated"]
    assert not policy.evaluate("unseen-rule", {"confidence": .5}, {"confidence": .5})["threshold_calibrated"]
    with pytest.raises(AnalysisError):
        ThresholdPolicy("changed-model", "4.0", path)


def test_data_leakage_unreviewed_rows_and_repeated_candidates_rejected():
    source = rows()
    source[-1]["group_id"] = source[0]["group_id"]
    with pytest.raises(ValueError, match="crosses"):
        fit_thresholds(source, model_id="m", pipeline_version="4.0")
    source = rows()
    source[0]["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        fit_thresholds(source, model_id="m", pipeline_version="4.0")
    source = rows()
    with pytest.raises(ValueError, match="Duplicate"):
        fit_thresholds(source + [source[0]], model_id="m", pipeline_version="4.0")


def test_validation_failure_never_deploys_calibration_cutoff(tmp_path):
    source = rows()
    for row in source:
        if row["partition"] == "validation" and not row["label"]:
            row["score"] = .99
    artifact = fit_thresholds(source, model_id="m", pipeline_version="4.0", minimum_examples=10)
    assert artifact["rules"]["a"]["status"] == "validation_failed"
    path = tmp_path / "p.json"
    path.write_text(json.dumps(artifact))
    policy = ThresholdPolicy("m", "4.0", path)
    assert not policy.evaluate("a", {"confidence": .9}, {"confidence": .9})["threshold_calibrated"]
    assert fit_thresholds(source[:3], model_id="m", pipeline_version="4.0")["rules"]["a"]["status"] == "insufficient_data"


def test_repeated_correlated_model_calls_never_accumulate_confidence():
    assert joint_score({"confidence": .8}, {"confidence": .6}) == .6
    for value in (float("nan"), True, -.1, 1.1):
        with pytest.raises(AnalysisError):
            joint_score({"confidence": value}, {"confidence": .8})


def test_many_outcomes_from_one_source_cannot_satisfy_calibration_sample_size():
    source = rows()
    for row in source:
        row["group_id"] = row["partition"]
    artifact = fit_thresholds(source, model_id="m", pipeline_version="4.0", minimum_examples=10)
    assert artifact["rules"]["a"]["status"] == "insufficient_data"


def test_evidence_retains_action_and_cited_before_after_only(tmp_path):
    frames = []
    for index in range(5):
        path = tmp_path / f"{index}.jpg"
        pixels = np.zeros((30, 30), np.uint8)
        pixels[:, index::3] = 255
        assert cv2.imwrite(str(path), pixels)
        frames.append(FrameRef(index, float(index), path))
    selected, event_time = select_evidence(frames, [2, 3], frames, [0, 2, 4], 2.)
    assert event_time == 2.
    assert [frame.index for frame in selected] == [0, 2, 4]
    assert len({frame.index for frame in select_evidence(frames, [2, 2], frames, [2, 2], 2.)[0]}) == 1
