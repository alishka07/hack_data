"""Approved-adapter contracts using fake local CPU features, never bank media."""
from copy import deepcopy
import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from worker import temporal
from worker.event_features import FEATURE_VERSION
from worker.video import AnalysisError, FrameRef, VideoInfo, nearest_index


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class BudgetSpy:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def remaining(self, *args):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise AnalysisError("deadline_exceeded", "Synthetic deadline")
        return 60


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    for key in ("FORTE_TEMPORAL_ENABLED", "FORTE_TEMPORAL_MODEL", "FORTE_TEMPORAL_EVALUATION",
                "FORTE_TEMPORAL_ENCODER", "FORTE_TEMPORAL_THREADS"):
        monkeypatch.delenv(key, raising=False)
    assets = tmp_path / "models/action-encoder"
    assets.mkdir(parents=True)
    weights = assets / "synthetic-weights.pth"
    weights.write_bytes(b"local-only synthetic weights")
    manifest = {"encoder": "r3d_18", "pretraining": "KINETICS400_V1", "feature_dimension": 512,
                "source_code_version": "torchvision==synthetic-test", "weights_filename": weights.name,
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "preprocessing": "full_frame_letterbox112_kinetics400_mean_std_v1"}
    save(assets / "manifest.json", manifest)
    config = {"encoder": manifest, "window_seconds": 8, "stride_seconds": 4,
              "frames_per_window": 16, "context_features": FEATURE_VERSION}
    model = {"format_version": 2, "model_type": "r3d18_weak_anchor_event_video", "task": "dataset_violation_prediction",
             "feature_config": config, "feature_fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
             "threshold": .6, "approved_for_inference": True,
             "classifier": {"format": "forte_weak_anchor_event_v1", "feature_dim": 1024,
                "pca": {"mean": [0.] * 1024, "scale": [1.] * 1024, "constant_features": [False] * 1024,
                        "n_components": 1, "components": [[1.] + [0.] * 1023], "output_scale": [1.]},
                "head": {"weights": [4.], "bias": 0.}}}
    report = {"approved_for_inference": True, "feature_fingerprint": model["feature_fingerprint"],
              "metrics": {"event": {"balanced_accuracy": .85, "recall": .8, "specificity": .9}},
              "balanced_accuracy_gain": .1,
              "inference_gate": {"minimum_nested_balanced_accuracy": .75, "minimum_nested_recall": .7,
                                 "minimum_nested_specificity": .7, "minimum_balanced_accuracy_gain_over_baseline": .05}}
    model_path = tmp_path / "models/case-detector-v2/model.json"
    report_path = tmp_path / "docs/event-model-evaluation.json"

    def write():
        save(model_path, model)
        report["final_model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        report["feature_fingerprint"] = model["feature_fingerprint"]
        save(report_path, report)

    write()
    info = VideoInfo(tmp_path / "ignored-name.mov", 12., 640, 480,
                     (0., .31, .9, 2.01, 3.4, 4.4, 5.8, 7.1, 8.3, 9.9, 11.8), 5.)
    state = {"loads": 0, "encodes": 0, "requests": [], "bad_source": False, "bad_features": False}

    class Cache:
        def get(self, targets, height):
            assert height == 112
            indices = [nearest_index(info.pts, target) for target in targets]
            state["requests"].append(indices)
            return [FrameRef(index, info.pts[index] + (.1 if state["bad_source"] else 0), Path(f"{index}.jpg"))
                    for index in indices]

    class Encoder:
        def encode_clips(self, clips):
            assert len(clips) == 1 and len(clips[0]) == 16
            state["encodes"] += 1
            if state["bad_features"]:
                return np.zeros((1, 511))
            result = np.zeros((1, 512), dtype=np.float32)
            result[0, 0] = -1 if state["encodes"] == 1 else 1
            return result

    def load_encoder(supplied, threads):
        assert supplied == assets and threads == 4
        state["loads"] += 1
        return Encoder()

    monkeypatch.setattr(temporal, "_load_encoder", load_encoder)
    import cv2
    monkeypatch.setattr(cv2, "imread", lambda *args: np.zeros((4, 4, 3), dtype=np.uint8))
    return {"root": tmp_path, "model": model, "report": report, "write": write, "info": info,
            "cache": Cache(), "state": state, "model_path": model_path, "report_path": report_path, "weights": weights}


def run(adapter, budget=None):
    return temporal.temporal_proposals(adapter["root"], adapter["info"], adapter["cache"], budget or BudgetSpy())


def test_default_failed_gate_reports_metrics_without_importing_torch_or_loading_encoder(adapter, monkeypatch):
    adapter["model"]["approved_for_inference"] = False
    adapter["report"]["approved_for_inference"] = False
    adapter["report"]["metrics"]["event"] = {"balanced_accuracy": .393483709273183,
                                              "recall": .5238095238095238, "specificity": .2631578947368421}
    adapter["write"]()
    original_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torchvision"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    budget = BudgetSpy()
    result = run(adapter, budget)
    assert result["status"] == "disabled_failed_gate" and result["proposals"] == []
    assert result["evaluation"]["balanced_accuracy"] == pytest.approx(.3934837)
    assert adapter["state"]["loads"] == adapter["state"]["encodes"] == budget.calls == 0


def test_missing_model_is_not_evaluated_by_default_and_enable_fails_closed(adapter, monkeypatch):
    adapter["model_path"].unlink()
    assert run(adapter)["status"] == "disabled_not_evaluated"
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_model_unapproved"
    assert adapter["state"]["loads"] == 0


def test_enabled_does_not_bypass_failed_artifact_or_failed_evaluation(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    adapter["model"]["approved_for_inference"] = False
    adapter["write"]()
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_model_unapproved"
    adapter["model"]["approved_for_inference"] = True
    adapter["report"]["approved_for_inference"] = False
    adapter["write"]()
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_model_unapproved"
    assert adapter["state"]["loads"] == 0


def test_changing_approval_booleans_cannot_override_failed_measured_metrics(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    adapter["report"]["metrics"]["event"]["balanced_accuracy"] = .39
    adapter["write"]()
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_model_unapproved"
    assert adapter["state"]["loads"] == 0


def test_approved_adapter_is_still_disabled_until_explicitly_enabled(adapter):
    assert run(adapter)["status"] == "disabled_by_configuration"
    assert adapter["state"]["loads"] == 0


def test_approved_cpu_adapter_binds_actual_source_pts_and_has_no_legal_type(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    budget = BudgetSpy()
    result = run(adapter, budget)
    assert result["status"] == "enabled_approved" and result["device"] == "cpu"
    assert result["windows_completed"] == 2 and adapter["state"]["encodes"] == 2
    assert adapter["state"]["loads"] == 1 and budget.calls > 32
    assert len(result["proposals"]) == 1
    proposal = result["proposals"][0]
    assert proposal["reason"] == "temporal_score" and proposal["score"] > result["threshold"]
    assert proposal["timecode_sec"] == 8.3
    assert proposal["start_sec"] == 4.4 and proposal["end_sec"] == 11.8
    for ref in proposal["refs"]:
        assert ref["timestamp_sec"] == adapter["info"].pts[ref["source_index"]]
    assert proposal["source_indices"] == sorted(set(adapter["state"]["requests"][1]))
    assert not {"type", "type_id", "verdict", "legal_decision", "dataset_prediction"} & set(proposal)
    assert result["score_calibrated"] is False
    json.dumps(result, allow_nan=False)


def test_adapter_keeps_all_above_threshold_windows_instead_of_only_best_one(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    adapter["model"]["threshold"] = 0
    adapter["write"]()
    assert len(run(adapter)["proposals"]) == 2


@pytest.mark.parametrize("change", [
    {"model_type": "unsupported"}, {"format_version": 1}, {"feature_fingerprint": "0" * 64},
    {"threshold": float("nan")},
])
def test_configured_artifact_corruption_fails_before_encoder_loading(adapter, monkeypatch, change):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    adapter["model"].update(change)
    if "threshold" in change:
        adapter["model_path"].write_text(json.dumps(adapter["model"]), encoding="utf-8")
        adapter["report"]["final_model_sha256"] = hashlib.sha256(adapter["model_path"].read_bytes()).hexdigest()
        save(adapter["report_path"], adapter["report"])
    else:
        adapter["write"]()
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_model_invalid" and adapter["state"]["loads"] == 0


def test_model_bytes_must_match_the_evaluated_artifact(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    adapter["model_path"].write_text(adapter["model_path"].read_text() + " ", encoding="utf-8")
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_model_invalid" and adapter["state"]["loads"] == 0


def test_encoder_weights_corruption_fails_before_loading_optional_runtime(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    adapter["weights"].write_bytes(b"modified bytes")
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_model_invalid" and adapter["state"]["loads"] == 0


def test_configured_bad_json_fails_explicitly(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    adapter["report_path"].write_text("{broken", encoding="utf-8")
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_model_invalid"


@pytest.mark.parametrize("problem,code", [("bad_source", "temporal_source_invalid"),
                                           ("bad_features", "temporal_model_invalid")])
def test_misaligned_frames_and_invalid_encoder_output_cannot_be_scored(adapter, monkeypatch, problem, code):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    adapter["state"][problem] = True
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == code


def test_deadline_during_later_window_fails_instead_of_returning_partial_clean_result(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "1")
    with pytest.raises(AnalysisError) as error:
        run(adapter, BudgetSpy(fail_at=25))
    assert error.value.code == "deadline_exceeded"
    assert adapter["state"]["encodes"] == 1


def test_invalid_enable_setting_fails_explicitly(adapter, monkeypatch):
    monkeypatch.setenv("FORTE_TEMPORAL_ENABLED", "true")
    with pytest.raises(AnalysisError) as error:
        run(adapter)
    assert error.value.code == "temporal_configuration_invalid"
