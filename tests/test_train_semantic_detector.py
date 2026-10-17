"""Synthetic trainer/cache tests: no private media, GPU, sockets or model assets."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts import train_semantic_detector as train
from worker.grouped_validation import evaluate_oof_predictions, nested_group_folds
from worker.semantic_assets import encoder_recipe, recipe_fingerprint
from worker.semantic_encoder import semantic_recipe
from worker.semantic_features import SEMANTIC_KEYS
from worker.video import AnalysisError


def answer():
    return {key: "unknown" for key in SEMANTIC_KEYS}


def metadata():
    return {"timestamps": [0., 1.], "source_indices": [0, 30], "duration_sec": 2.,
            "frame_dimensions": [{"width": 160, "height": 448}] * 2, "recipe": semantic_recipe()}


def groups(positive=21, negative=19):
    return [{"video_sha256": hashlib.sha256(f"{label}-{index}".encode()).hexdigest(), "binary_label": label,
             "video_path": f"private-{label}-{index}.mov", "anchors_sec": [1.] if label else []}
            for label, count in ((1, positive), (0, negative)) for index in range(count)]


def payload(fingerprint, digest):
    return {"recipe_sha256": fingerprint, "video_sha256": digest, "answer": answer(), "metadata": metadata()}


def test_recipe_hashes_actual_assets_and_contains_no_private_paths(tmp_path, monkeypatch):
    paths = []
    for name in ("FORTE_MODEL_PATH", "FORTE_MMPROJ_PATH", "FORTE_LLAMA_SERVER"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        monkeypatch.setenv(name, str(path))
        paths.append(path)
    recipe = encoder_recipe(tmp_path)
    assert str(tmp_path) not in json.dumps(recipe)
    assert recipe["runtime"]["context_tokens"] == 8192 and recipe["runtime"]["max_output_tokens"] == 768
    assert recipe["assets"]["model"]["sha256"] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert recipe_fingerprint(recipe) == recipe_fingerprint(dict(reversed(list(recipe.items()))))
    original = recipe_fingerprint(recipe)
    paths[0].write_bytes(b"different asset")
    assert recipe_fingerprint(encoder_recipe(tmp_path)) != original


def test_future_shared_validator_change_fails_before_asset_access(tmp_path, monkeypatch):
    from worker import semantic_assets

    def different_validator(value, schema):
        return None

    monkeypatch.setattr(semantic_assets, "encoder_validator", different_validator)
    with pytest.raises(ValueError, match="differs from the frozen"):
        semantic_assets.encoder_recipe(tmp_path)


@pytest.mark.parametrize("defect", ["label_field", "foreign_recipe", "foreign_video", "missing_observable",
                                   "foreign_source_time", "source_order", "oversized_frame"])
def test_invalid_or_label_contaminated_cache_is_rejected(defect):
    record = payload("recipe", "video")
    if defect == "label_field":
        record["binary_label"] = 1
    elif defect == "foreign_recipe":
        record["recipe_sha256"] = "different"
    elif defect == "foreign_video":
        record["video_sha256"] = "different"
    elif defect == "missing_observable":
        record["answer"].pop(SEMANTIC_KEYS[0])
    elif defect == "foreign_source_time":
        record["metadata"]["timestamps"][-1] = 999
    elif defect == "source_order":
        record["metadata"]["source_indices"] = [30, 0]
    else:
        record["metadata"]["frame_dimensions"][0] = {"width": 2000, "height": 448}
    with pytest.raises((ValueError, AnalysisError)):
        train.validate_feature_record(record, "recipe", "video")


def test_complete_cache_never_constructs_a_runtime(tmp_path, monkeypatch):
    recipe = {"synthetic": "recipe"}
    fingerprint = recipe_fingerprint(recipe)
    source = groups(positive=1, negative=0)
    digest = source[0]["video_sha256"]
    train.write_json(tmp_path / "training/semantic_features" / fingerprint / (digest + ".json"), payload(fingerprint, digest))
    monkeypatch.setattr(train, "LocalRuntime", lambda *_: pytest.fail("Cached training must not construct a runtime"))
    answers, audit = train.collect_features(source, recipe, tmp_path, train_only=True)
    assert answers[digest] == answer() and audit == {"groups": 1, "cached": 1, "missing": 0, "invalid": 0}


def test_train_only_fails_before_runtime_construction_when_cache_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "LocalRuntime", lambda *_: pytest.fail("Train-only must not construct a runtime"))
    with pytest.raises(ValueError, match="every exact-recipe"):
        train.collect_features(groups(positive=1, negative=0), {"synthetic": "recipe"}, tmp_path, train_only=True)


@pytest.mark.parametrize("failure", [None, "startup", "extraction"])
def test_only_video_path_reaches_one_owned_encoder_and_runtime_always_stops(tmp_path, monkeypatch, capsys, failure):
    events, source = [], groups(positive=2, negative=0)

    class Runtime:
        external = False

        def __init__(self, *_):
            events.append("construct")

        def start(self):
            events.append("start")
            if failure == "startup":
                raise AnalysisError("model_unavailable", "Synthetic startup failure")

        def stop(self):
            events.append("stop")

    class Encoder:
        def __init__(self, root, data_dir, runtime):
            assert isinstance(runtime, Runtime)

        def clip_features(self, video):
            assert str(video) in {group["video_path"] for group in source}
            events.append("pixels")
            if failure == "extraction":
                raise AnalysisError("model_request_failed", "Synthetic extraction failure")
            return answer(), metadata()

    monkeypatch.setattr(train, "LocalRuntime", Runtime)
    monkeypatch.setattr(train, "SemanticEncoder", Encoder)
    recipe = {"synthetic": "recipe"}
    if failure:
        with pytest.raises(AnalysisError):
            train.collect_features(source, recipe, tmp_path)
    else:
        answers, _ = train.collect_features(source, recipe, tmp_path)
        assert len(answers) == 2 and events.count("pixels") == 2
        for path in (tmp_path / "training/semantic_features").glob("*/*.json"):
            cached = json.loads(path.read_text())
            assert set(cached) == {"recipe_sha256", "video_sha256", "answer", "metadata"}
            assert "private" not in path.read_text() and "binary_label" not in path.read_text()
    assert events.count("construct") == events.count("start") == events.count("stop") == 1
    assert "private" not in capsys.readouterr().out


def test_joint_selection_prefers_larger_penalty_then_main_only_after_performance_ties(monkeypatch):
    ids = ["a", "b", "c", "d"]
    records = {key: {"binary_label": index % 2} for index, key in enumerate(ids)}
    folds = [{"train_ids": ids[:2], "validation_ids": ids[2:]}, {"train_ids": ids[2:], "validation_ids": ids[:2]}]
    vectors = {family: {key: np.zeros(2) for key in ids} for family in ("main", "pairs")}
    protocol = {"supervised_head": {"candidate_families": ["pairs", "main"], "regularization_candidates": [.3, 3., 30.]}}
    calls = []
    monkeypatch.setattr(train, "fit_model", lambda chosen, *_: calls.append(set(chosen)) or {})
    monkeypatch.setattr(train, "predict_probabilities", lambda *_: np.array([.1, .9]))
    selected = train.select_configuration(ids, folds, records, vectors, protocol)
    assert selected["regularization"] == 30 and selected["feature_family"] == "main"
    assert selected["training_selection_metrics"]["balanced_accuracy"] == 1
    assert len(calls) == 12 and all(chosen in (set(ids[:2]), set(ids[2:])) for chosen in calls)


def test_inner_selection_rejects_outer_test_ids_before_any_fit(monkeypatch):
    protocol = {"supervised_head": {"candidate_families": ["main"], "regularization_candidates": [3.]}}
    monkeypatch.setattr(train, "fit_model", lambda *_: pytest.fail("Escaped test data must not fit"))
    with pytest.raises(ValueError, match="escape"):
        train.select_configuration(["train", "validation"],
                                   [{"train_ids": ["train"], "validation_ids": ["outer_test"]}], {}, {}, protocol)


def test_baseline_comparison_requires_exact_saved_fold_membership_and_metrics(tmp_path):
    source = groups(positive=5, negative=5)
    manifest = {"source_sha256": "synthetic-source", "groups": source}
    folds = nested_group_folds(source)
    fold_by_id = {digest: row["fold"] for row in folds for digest in row["test_ids"]}
    rows = [{"video_sha256": row["video_sha256"], "probability": .9 if row["binary_label"] else .1,
             "threshold": .5, "fold": fold_by_id[row["video_sha256"]]} for row in source]
    report = {"dataset_sha256": "synthetic-source", "video_groups": 10, "outer_folds": 5, "inner_folds": 4,
              "metrics": {"baseline": evaluate_oof_predictions(source, rows)}}
    path = tmp_path / "baseline.json"
    train.write_json(path, report)
    private = tmp_path / "training/synthetic-event"
    train.write_json(private / "manifest.json", {**manifest, "folds": folds})
    train.write_json(private / "predictions.json", {"baseline": rows})
    assert train.verified_baseline(path, manifest, folds, tmp_path)["metrics"] == report["metrics"]["baseline"]
    rows[0]["fold"] = 999
    train.write_json(private / "predictions.json", {"baseline": rows})
    with pytest.raises(ValueError, match="prediction snapshot"):
        train.verified_baseline(path, manifest, folds, tmp_path)


def test_final_full_data_threshold_does_not_rewrite_outer_evaluation(tmp_path, monkeypatch):
    source = groups()
    manifest = {"source_sha256": "synthetic-source", "groups": source}
    protocol = {"supervised_head": {"candidate_families": ["main"], "regularization_candidates": [3.]},
                "inference_gate": {"minimum_nested_balanced_accuracy": .75, "minimum_nested_recall": .7,
                                   "minimum_nested_specificity": .7, "minimum_balanced_accuracy_gain_over_baseline": .05}}
    train.write_json(tmp_path / "docs/semantic-model-protocol.json", protocol)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "read_simplified_labels", lambda *_: deepcopy(manifest))
    monkeypatch.setattr(train, "verified_baseline", lambda *_: {"metrics": {"balanced_accuracy": .4}, "report_sha256": "baseline"})
    monkeypatch.setattr(train, "encoder_recipe", lambda *_: {"synthetic": "recipe"})
    monkeypatch.setattr(train, "collect_features", lambda *_args, **_kwargs: ({row["video_sha256"]: answer() for row in source}, {"cached": 40}))
    monkeypatch.setattr(train, "select_configuration", lambda ids, *_: {"feature_family": "main", "regularization": 3.,
                                                                       "threshold": .9 if len(ids) == 40 else .5})
    monkeypatch.setattr(train, "fit_model", lambda *_: {"format": "synthetic"})
    monkeypatch.setattr(train, "predict_probabilities", lambda _model, matrix: np.full(len(matrix), .5))
    assert train.main(["--train-only"]) == 0
    model = json.loads((tmp_path / "models/case-detector-v3/model.json").read_text())
    report = json.loads((tmp_path / "docs/semantic-model-evaluation.json").read_text())
    assert model["threshold"] == .9 and model["approved_for_inference"] is False
    assert report["metrics"]["semantic"]["true_positive"] == 21
    assert report["metrics"]["semantic"]["false_positive"] == 19
    assert report["final_model_training_selection"]["threshold"] == .9
