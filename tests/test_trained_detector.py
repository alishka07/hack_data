"""Synthetic model-loading and pixel-only inference contracts, without Torch."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import predict_case
from worker import trained_detector as detector


def artifact(*, approved=True, weights=b"synthetic encoder weights"):
    config = {"encoder": {"encoder": "dinov2_vits14", "weights_sha256": hashlib.sha256(weights).hexdigest()},
              "max_frames": 12, "preprocessing": "full_frame_letterbox_224_imagenet_v1",
              "pooling": "mean_std_last_minus_first_v1"}
    return {"format_version": 1, "model_type": "dinov2_linear_binary_video", "task": "dataset_violation_prediction",
            "approved_for_inference": approved, "threshold": .5, "feature_config": config,
            "feature_fingerprint": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "classifier": {"format": "forte_linear_classifier_v1", "feature_dim": 1152,
                           "mean": [0.] * 1152, "scale": [1.] * 1152,
                           "constant_features": [False] * 1152, "weights": [1.] + [0.] * 1151, "bias": 0.}}


def write_model(tmp_path, model):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    return path


def test_unapproved_model_is_refused_before_video_or_encoder_access(tmp_path, monkeypatch):
    path = write_model(tmp_path, artifact(approved=False))
    monkeypatch.setattr(detector, "DinoEncoder", lambda *args, **kwargs: pytest.fail("Unapproved model must not load the encoder"))
    with pytest.raises(ValueError, match="approval gate"):
        detector.predict_video(tmp_path / "unopened.mov", path, tmp_path / "unopened_encoder")
    loaded = detector.load_model_artifact(path, experimental=True)
    result = detector.predict_features(loaded, np.zeros(1152), experimental=True)
    assert result["approved_for_inference"] is False and result["score_calibrated"] is False


@pytest.mark.parametrize("field,value", [("format_version", True), ("model_type", "other"),
                                        ("task", "legal_verdict"), ("approved_for_inference", "yes"),
                                        ("threshold", True), ("threshold", -1),
                                        ("threshold", float("nan")), ("threshold", 1.1),
                                        ("feature_fingerprint", "changed")])
def test_invalid_artifact_metadata_is_rejected_before_encoding(tmp_path, field, value):
    model = artifact()
    model[field] = value
    with pytest.raises(ValueError):
        detector.load_model_artifact(write_model(tmp_path, model))


@pytest.mark.parametrize("field,value", [("preprocessing", "center_crop"), ("pooling", "mean_only"),
                                        ("max_frames", True), ("max_frames", 1), ("max_frames", 513),
                                        ("encoder", {"encoder": "different", "weights_sha256": "a" * 64})])
def test_incompatible_feature_configuration_cannot_silently_change_predictions(tmp_path, field, value):
    model = artifact()
    model["feature_config"][field] = value
    model["feature_fingerprint"] = hashlib.sha256(json.dumps(model["feature_config"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError):
        detector.load_model_artifact(write_model(tmp_path, model))


def test_corrupt_classifier_is_rejected_by_loader(tmp_path):
    model = artifact()
    model["classifier"]["scale"][0] = 0
    with pytest.raises(ValueError):
        detector.load_model_artifact(write_model(tmp_path, model))


def test_prediction_uses_fitted_transform_and_threshold_with_no_category_claims():
    model = artifact()
    previous = deepcopy(model)
    features = np.zeros(1152)
    features[0] = -2
    result = detector.predict_features(model, features)
    assert result["dataset_prediction"] == "no_violation"
    assert result["score"] == pytest.approx(1 / (1 + np.exp(2)))
    assert result["score_calibrated"] is False
    assert set(result) == {"dataset_prediction", "score", "threshold", "approved_for_inference", "score_calibrated"}
    features[0] = 0
    assert detector.predict_features(model, features)["dataset_prediction"] == "violation"
    assert model == previous
    for invalid in (np.zeros(384), np.zeros((2, 1152)), np.full(1152, np.nan)):
        with pytest.raises(ValueError):
            detector.predict_features(model, invalid)


def test_video_prediction_reads_only_model_pixels_and_matching_local_encoder_assets(tmp_path, monkeypatch):
    model_path = write_model(tmp_path, artifact())
    video, assets = tmp_path / "unrelated_name.mov", tmp_path / "encoder"
    video.write_bytes(b"synthetic pixel source")
    assets.mkdir()
    (assets / "dinov2_vits14_pretrain.pth").write_bytes(b"synthetic encoder weights")
    calls, reads = [], []
    real_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        reads.append(path)
        assert path == model_path, "Inference must never open a workbook or manifest"
        return real_read_text(path, *args, **kwargs)

    class FakeEncoder:
        def __init__(self, directory, threads):
            calls.append((directory, threads))

        def clip_features(self, path, max_frames):
            calls.append((path, max_frames))
            assert path.read_bytes() == b"synthetic pixel source"
            return np.zeros(1152), [0., 1.], np.zeros((2, 384))

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(detector, "DinoEncoder", FakeEncoder)
    result = detector.predict_video(video, model_path, assets, threads=2)
    assert calls == [(assets, 2), (video, 12)] and reads == [model_path]
    assert result["score"] == .5
    (assets / "dinov2_vits14_pretrain.pth").write_bytes(b"wrong encoder")
    with pytest.raises(ValueError, match="checksum"):
        detector.predict_video(video, model_path, assets)
    assert len(calls) == 2


def test_cli_requires_explicit_experimental_override_for_unapproved_artifact(tmp_path, capsys):
    model_path = write_model(tmp_path, artifact(approved=False))
    code = predict_case.main([str(tmp_path / "unopened.mov"), "--model", str(model_path)])
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ""
    assert "--experimental" in json.loads(captured.err)["error"]


def test_cli_outputs_only_concise_json_and_passes_explicit_research_options(tmp_path, monkeypatch, capsys):
    expected = {"dataset_prediction": "violation", "score": .7, "threshold": .6,
                "approved_for_inference": False, "score_calibrated": False}
    calls = []

    def predict(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(predict_case, "predict_video", predict)
    video, model, encoder = (tmp_path / name for name in ("video.mov", "model.json", "encoder"))
    assert predict_case.main([str(video), "--model", str(model), "--encoder", str(encoder),
                              "--threads", "2", "--experimental"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert calls == [((video, model, encoder), {"threads": 2, "experimental": True})]
