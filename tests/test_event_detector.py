"""Offline inference contracts with a tiny fake local encoder and numeric head."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import predict_event
from worker import event_detector
from worker.event_detector import EventDetector
from worker.event_features import FEATURE_VERSION


@pytest.fixture
def assets(tmp_path, monkeypatch):
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    weight_file = encoder_dir / "synthetic-weights.pth"
    weight_file.write_bytes(b"synthetic local encoder weights")
    manifest = {"encoder": "r3d_18", "pretraining": "KINETICS400_V1", "feature_dimension": 512,
                "source_code_version": "torchvision==0.28.0+cpu",
                "preprocessing": "full_frame_letterbox112_kinetics400_mean_std_v1",
                "weights_filename": weight_file.name,
                "weights_sha256": hashlib.sha256(weight_file.read_bytes()).hexdigest()}
    (encoder_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"encoder": manifest, "window_seconds": 8, "stride_seconds": 4,
              "frames_per_window": 16, "context_features": FEATURE_VERSION}
    component = [1.] + [0.] * 1023
    classifier = {"format": "forte_weak_anchor_event_v1", "feature_dim": 1024,
                  "pca": {"mean": [0.] * 1024, "scale": [1.] * 1024,
                          "constant_features": [False] * 1024, "n_components": 1,
                          "components": [component], "output_scale": [1.]},
                  "head": {"weights": [4.], "bias": 0.}}
    model = {"format_version": 2, "model_type": "r3d18_weak_anchor_event_video",
             "task": "dataset_violation_prediction", "feature_config": config,
             "feature_fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
             "classifier": classifier, "threshold": .6, "approved_for_inference": True}
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    video = tmp_path / "synthetic.mov"
    video.write_bytes(b"synthetic video placeholder")
    state = {"loads": 0, "calls": [], "negative": False, "invalid_windows": False}

    class FakeEncoder:
        def clip_features(self, supplied):
            state["calls"].append(Path(supplied))
            features = np.zeros((2, 512), dtype=np.float32)
            features[:, 0] = [-1., -1.] if state["negative"] else [-1., 1.]
            windows = [{"start_sec": 0., "end_sec": 8., "center_sec": 4.},
                       {"start_sec": 4., "end_sec": 12., "center_sec": 8.}]
            return features, windows[:1] if state["invalid_windows"] else windows

    def create_encoder(supplied, threads):
        assert Path(supplied) == encoder_dir and threads == 4
        state["loads"] += 1
        return FakeEncoder()

    monkeypatch.setattr(event_detector, "_encoder", create_encoder)
    return model_path, encoder_dir, video, model, state, weight_file


def save_model(path, model):
    path.write_text(json.dumps(model, allow_nan=False), encoding="utf-8")


def test_load_once_predict_from_windows_and_report_only_estimated_input_bounds(assets):
    path, encoder, video, _, state, _ = assets
    detector = EventDetector(path, encoder)
    positive = detector.predict(video)
    assert positive["dataset_prediction"] == "violation"
    assert positive["score"] > positive["threshold"]
    assert positive["best_window"] == {"start_sec": 4., "end_sec": 12., "center_sec": 8.}
    assert positive["timing_scope"] == "highest_scoring_input_window_not_event_boundaries"
    assert positive["score_calibrated"] is False
    assert positive["approved_for_inference"] is True
    assert not {"category", "bbox", "legal_decision", "event_start_sec"} & set(positive)
    state["negative"] = True
    negative = detector.predict(video)
    assert negative["dataset_prediction"] == "no_violation"
    assert negative["score"] < negative["threshold"]
    assert state["loads"] == 1 and state["calls"] == [video, video]
    json.dumps(positive, allow_nan=False)


def test_unapproved_models_require_an_explicit_experimental_override(assets):
    path, encoder, video, model, state, _ = assets
    model["approved_for_inference"] = False
    save_model(path, model)
    with pytest.raises(ValueError, match="approval gate"):
        EventDetector(path, encoder)
    assert state["loads"] == 0
    result = EventDetector(path, encoder, experimental=True).predict(video)
    assert result["approved_for_inference"] is False
    assert state["loads"] == 1


def test_actual_weight_bytes_must_match_even_in_experimental_mode(assets):
    path, encoder, _, _, state, weights = assets
    weights.write_bytes(b"changed weight bytes")
    with pytest.raises(ValueError, match="checksum"):
        EventDetector(path, encoder, experimental=True)
    assert state["loads"] == 0


def test_local_encoder_manifest_must_match_the_artifact(assets):
    path, encoder, _, _, state, _ = assets
    manifest_path = encoder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_code_version"] = "different version"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        EventDetector(path, encoder)
    assert state["loads"] == 0


def test_local_manifest_may_add_provenance_without_changing_the_training_snapshot(assets):
    path, encoder, video, _, state, _ = assets
    manifest_path = encoder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_revision"] = "additional immutable source revision"
    manifest["source_file"] = "torchvision/models/video/resnet.py"
    manifest["source_file_sha256"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert EventDetector(path, encoder).predict(video)["dataset_prediction"] == "violation"
    assert state["loads"] == 1


@pytest.mark.parametrize("change", [
    {"format_version": 1}, {"task": "legal_compliance"},
    {"feature_fingerprint": "0" * 64}, {"threshold": 1.1},
    {"approved_for_inference": 1},
])
def test_incompatible_artifacts_fail_before_loading_an_encoder(assets, change):
    path, encoder, _, model, state, _ = assets
    model.update(change)
    save_model(path, model)
    with pytest.raises(ValueError):
        EventDetector(path, encoder)
    assert state["loads"] == 0


def test_recomputed_fingerprint_cannot_enable_an_unsupported_sampler(assets):
    path, encoder, _, model, state, _ = assets
    model["feature_config"]["frames_per_window"] = 8
    model["feature_fingerprint"] = hashlib.sha256(json.dumps(model["feature_config"], sort_keys=True).encode()).hexdigest()
    save_model(path, model)
    with pytest.raises(ValueError, match="configuration"):
        EventDetector(path, encoder)
    assert state["loads"] == 0


def test_invalid_window_metadata_and_missing_videos_never_produce_a_clean_result(assets):
    path, encoder, video, _, state, _ = assets
    detector = EventDetector(path, encoder)
    state["invalid_windows"] = True
    with pytest.raises(ValueError, match="matching lengths"):
        detector.predict(video)
    with pytest.raises(FileNotFoundError):
        detector.predict(video.parent / "missing.mov")


def test_prediction_does_not_read_training_metadata_or_modify_the_model(assets, monkeypatch):
    path, encoder, video, _, _, _ = assets
    detector = EventDetector(path, encoder)
    original = deepcopy(detector.model)

    def no_metadata_reads(*_args, **_kwargs):
        raise AssertionError("Prediction must not read training workbooks, manifests or caches")

    monkeypatch.setattr(Path, "read_text", no_metadata_reads)
    detector.predict(video)
    assert detector.model == original


def test_cli_prints_only_prediction_and_propagates_the_experimental_flag(assets, monkeypatch, capsys):
    path, encoder, video, _, _, _ = assets
    captured = {}

    class FakeDetector:
        def __init__(self, model, directory, **kwargs):
            captured.update(model=model, encoder=directory, **kwargs)

        def predict(self, supplied):
            assert supplied == video
            return {"dataset_prediction": "violation", "score": .8, "score_calibrated": False}

    monkeypatch.setattr(predict_event, "EventDetector", FakeDetector)
    assert predict_event.main([str(video), "--model", str(path), "--encoder", str(encoder), "--experimental"]) == 0
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out)["dataset_prediction"] == "violation"
    assert captured == {"model": path, "encoder": encoder, "threads": 4, "experimental": True}


def test_cli_failure_is_nonzero_with_no_prediction_on_stdout(assets, monkeypatch, capsys):
    _, _, video, _, _, _ = assets

    def unavailable(*_args, **_kwargs):
        raise ValueError("Model was not approved")

    monkeypatch.setattr(predict_event, "EventDetector", unavailable)
    assert predict_event.main([str(video)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"error": "Model was not approved"}
