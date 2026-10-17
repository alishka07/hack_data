"""Semantic inference and runtime ownership, using no video/model calls."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from scripts import predict_semantic
from worker import semantic_detector
from worker.semantic_assets import recipe_fingerprint
from worker.semantic_detector import PureSemanticDetector, SemanticDetector
from worker.semantic_features import SEMANTIC_KEYS, SEMANTIC_VERSION
from worker.video import AnalysisError


@pytest.fixture
def setup(tmp_path, monkeypatch):
    recipe = {"semantic_version": SEMANTIC_VERSION, "observation": {"sampling": "synthetic-only"},
              "runtime": {"context_tokens": 8192}, "assets": {"model": {"sha256": "a" * 64}}}
    fingerprint = recipe_fingerprint(recipe)
    dimension = 52
    model = {"format_version": 3, "model_type": "qwen3vl_semantic_binary_video",
             "task": "dataset_violation_prediction", "semantic_recipe": deepcopy(recipe),
             "recipe_fingerprint": fingerprint, "encoder_fingerprint": fingerprint,
             "semantic_version": SEMANTIC_VERSION, "feature_family": "main",
             "classifier": {"format": "forte_linear_classifier_v1", "feature_dim": dimension,
                            "mean": [0.] * dimension, "scale": [1.] * dimension,
                            "weights": [4.] + [0.] * (dimension - 1), "bias": 0.},
             "threshold": .6, "approved_for_inference": True,
             "evaluation_scope": "exploratory_nested_group_cv", "category_heads": []}
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    video = tmp_path / "synthetic.mov"
    video.write_bytes(b"synthetic video placeholder")
    state = {"recipe_checks": 0, "encoder_loads": 0, "predictions": 0,
             "observations": dict.fromkeys(SEMANTIC_KEYS, "unknown")}
    state["observations"]["cash_visible"] = "yes"
    runtime = object()

    def actual_recipe(root):
        assert root == tmp_path
        state["recipe_checks"] += 1
        return recipe

    class FakeEncoder:
        def __init__(self, root, data, supplied_runtime):
            assert root == tmp_path and data == tmp_path / "data"
            assert supplied_runtime is runtime
            state["encoder_loads"] += 1

        def clip_features(self, supplied_video):
            assert supplied_video == video
            state["predictions"] += 1
            return deepcopy(state["observations"]), {"timestamps": [0., 3.]}

    monkeypatch.setattr(semantic_detector, "encoder_recipe", actual_recipe)
    monkeypatch.setattr(semantic_detector, "SemanticEncoder", FakeEncoder)
    return tmp_path, model_path, video, model, recipe, runtime, state


def save_model(path, model):
    path.write_text(json.dumps(model, allow_nan=False), encoding="utf-8")


def test_predicts_without_categories_times_or_training_metadata_and_loads_once(setup):
    root, path, video, _, _, runtime, state = setup
    assert PureSemanticDetector is SemanticDetector
    detector = SemanticDetector(root, path, runtime)
    result = detector.predict(video)
    assert result["dataset_prediction"] == "violation" and result["score"] > .9
    assert result["score_calibrated"] is False and result["approved_for_inference"] is True
    assert set(result) == {"dataset_prediction", "score", "threshold", "approved_for_inference", "score_calibrated"}
    state["observations"]["cash_visible"] = "no"
    assert detector.predict(video)["dataset_prediction"] == "no_violation"
    assert state["recipe_checks"] == state["encoder_loads"] == 1
    assert state["predictions"] == 2
    json.dumps(result, allow_nan=False)


def test_pairs_family_uses_the_matching_377_feature_head(setup):
    root, path, video, model, _, runtime, state = setup
    model["feature_family"] = "pairs"
    model["classifier"].update(feature_dim=377, mean=[0.] * 377, scale=[1.] * 377,
                                weights=[0.] * 377)
    model["classifier"]["weights"][52] = 4.
    state["observations"]["loose_notes_on_counter"] = "yes"
    save_model(path, model)
    assert SemanticDetector(root, path, runtime).predict(video)["dataset_prediction"] == "violation"


def test_unapproved_models_are_blocked_unless_explicitly_experimental(setup):
    root, path, video, model, _, runtime, state = setup
    model["approved_for_inference"] = False
    save_model(path, model)
    with pytest.raises(ValueError, match="approval gate"):
        SemanticDetector(root, path, runtime)
    assert state["recipe_checks"] == state["encoder_loads"] == 0
    assert SemanticDetector(root, path, runtime, experimental=True).predict(video)["approved_for_inference"] is False


@pytest.mark.parametrize("change", [
    {"format_version": 2}, {"task": "legal_compliance"}, {"semantic_version": "different"},
    {"feature_family": "unknown"}, {"threshold": 1.1}, {"approved_for_inference": 1},
    {"encoder_fingerprint": "0" * 64}, {"recipe_fingerprint": "0" * 64},
    {"evaluation_scope": "training_fit"}, {"category_heads": ["invented-category"]},
])
def test_invalid_artifacts_fail_before_loading_encoder_or_local_assets(setup, change):
    root, path, _, model, _, runtime, state = setup
    model.update(change)
    save_model(path, model)
    with pytest.raises(ValueError):
        SemanticDetector(root, path, runtime)
    assert state["recipe_checks"] == state["encoder_loads"] == 0


def test_changed_actual_recipe_is_rejected_even_when_model_fingerprint_is_valid(setup):
    root, path, _, _, recipe, runtime, state = setup
    recipe["runtime"]["context_tokens"] = 4096
    with pytest.raises(ValueError, match="recipe or assets"):
        SemanticDetector(root, path, runtime, experimental=True)
    assert state["encoder_loads"] == 0


def test_prediction_does_not_read_a_workbook_cache_or_model_again(setup, monkeypatch):
    root, path, video, _, _, runtime, _ = setup
    detector = SemanticDetector(root, path, runtime)
    original = deepcopy(detector.model)

    def no_metadata_reads(*_args, **_kwargs):
        raise AssertionError("Prediction must not consult a workbook, cache or known-video manifest")

    monkeypatch.setattr(Path, "read_text", no_metadata_reads)
    detector.predict(video)
    assert detector.model == original


def test_invalid_answers_and_missing_video_do_not_produce_a_clean_prediction(setup):
    root, path, video, _, _, runtime, state = setup
    detector = SemanticDetector(root, path, runtime)
    state["observations"]["verdict"] = "yes"
    with pytest.raises(ValueError):
        detector.predict(video)
    with pytest.raises(FileNotFoundError):
        detector.predict(root / "missing.mov")


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_semantic, "ROOT", tmp_path)
    monkeypatch.delenv("FORTE_ATTACH_MODEL", raising=False)
    video = tmp_path / "synthetic.mov"
    video.write_bytes(b"synthetic")
    state = {"calls": [], "busy": False, "start_failure": False,
             "predict_failure": False, "validation_failure": False}

    class FakeRuntime:
        def __init__(self, root, data):
            assert root == tmp_path and data == tmp_path / "data"
            self.external = os.environ.get("FORTE_ATTACH_MODEL") == "1"
            self.process = self.log = None
            state["calls"].append("construct-attached" if self.external else "construct-managed")

        def start(self):
            state["calls"].append("start")
            if state["busy"] and not self.external:
                raise AnalysisError("model_port_busy", "Existing server owns the port")
            if not self.external:
                self.process = object()
            if state["start_failure"]:
                raise AnalysisError("model_start_failed", "Managed startup failed")

        def stop(self):
            state["calls"].append("stop")
            self.process = self.log = None

    class FakeDetector:
        def __init__(self, root, model, runtime, experimental=False):
            assert root == tmp_path
            assert model == tmp_path / "models/case-detector-v3/model.json"
            state["calls"].append("validate-experimental" if experimental else "validate")
            if state["validation_failure"]:
                raise ValueError("Model was not approved")

        def predict(self, supplied):
            assert supplied == video
            state["calls"].append("predict")
            if state["predict_failure"]:
                raise ValueError("Invalid observations")
            return {"dataset_prediction": "violation", "score": .8, "score_calibrated": False}

    monkeypatch.setattr(predict_semantic, "LocalRuntime", FakeRuntime)
    monkeypatch.setattr(predict_semantic, "SemanticDetector", FakeDetector)
    return video, state


def test_cli_owns_and_stops_only_its_managed_server(cli, capsys):
    video, state = cli
    assert predict_semantic.main([str(video)]) == 0
    assert state["calls"] == ["construct-managed", "validate", "start", "predict", "stop"]
    output = capsys.readouterr()
    assert not output.err and json.loads(output.out)["dataset_prediction"] == "violation"


def test_cli_explicit_attach_never_stops_the_existing_server_and_restores_env(cli, monkeypatch, capsys):
    video, state = cli
    monkeypatch.setenv("FORTE_ATTACH_MODEL", "0")
    assert predict_semantic.main([str(video), "--attach", "--experimental"]) == 0
    assert state["calls"] == ["construct-attached", "validate-experimental", "start", "predict"]
    assert os.environ["FORTE_ATTACH_MODEL"] == "0"
    assert json.loads(capsys.readouterr().out)["dataset_prediction"] == "violation"


@pytest.mark.parametrize(("failure", "expected"), [
    ("busy", ["construct-managed", "validate", "start"]),
    ("start_failure", ["construct-managed", "validate", "start", "stop"]),
    ("predict_failure", ["construct-managed", "validate", "start", "predict", "stop"]),
    ("validation_failure", ["construct-managed", "validate"]),
])
def test_cli_failures_preserve_ownership_and_never_print_a_clean_result(cli, capsys, failure, expected):
    video, state = cli
    state[failure] = True
    assert predict_semantic.main([str(video)]) == 2
    assert state["calls"] == expected
    output = capsys.readouterr()
    assert output.out == "" and "error" in json.loads(output.err)


def test_missing_video_does_not_construct_or_start_a_runtime(cli, capsys):
    video, state = cli
    assert predict_semantic.main([str(video.parent / "missing.mov")]) == 2
    assert state["calls"] == []
    assert capsys.readouterr().out == ""
