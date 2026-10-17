"""Synthetic grouped protocols prevent duplicate videos crossing nested folds."""
from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest

from worker.grouped_validation import (choose_sensitivity_balanced_threshold,
                                       evaluate_oof_predictions, nested_group_folds,
                                       safe_binary_metrics, stratified_group_folds)


def groups(positive=21, negative=19):
    return [{"video_sha256": hashlib.sha256(f"{label}-{index}".encode()).hexdigest(),
             "binary_label": label, "video_path": f"private-{label}-{index}.mov"}
            for label, count in ((1, positive), (0, negative)) for index in range(count)]


def test_stratification_is_balanced_complete_disjoint_and_independent_of_input_order():
    source = groups()
    original = deepcopy(source)
    folds = stratified_group_folds(source)
    assert folds == stratified_group_folds(list(reversed(source)))
    assert folds != stratified_group_folds(source, seed=18)
    assert source == original
    all_ids = {group["video_sha256"] for group in source}
    labels = {group["video_sha256"]: group["binary_label"] for group in source}
    heldout = []
    class_counts = []
    for fold in folds:
        train, validation = set(fold["train_ids"]), set(fold["validation_ids"])
        assert train.isdisjoint(validation) and train | validation == all_ids
        assert len(train) == 32 and len(validation) == 8
        heldout.extend(validation)
        class_counts.append(Counter(labels[digest] for digest in validation))
    assert Counter(heldout) == Counter({digest: 1 for digest in all_ids})
    assert all(max(count[label] for count in class_counts) - min(count[label] for count in class_counts) <= 1 for label in (0, 1))


def test_nested_folds_never_reintroduce_outer_test_videos_in_inner_training_or_validation():
    source = groups()
    folds = nested_group_folds(source)
    assert folds == nested_group_folds(source)
    for fold in folds:
        train, test = set(fold["train_ids"]), set(fold["test_ids"])
        inner_validation = []
        assert len(fold["inner_folds"]) == 4
        for inner in fold["inner_folds"]:
            inner_train, validation = set(inner["train_ids"]), set(inner["validation_ids"])
            assert inner_train | validation == train and inner_train.isdisjoint(validation)
            assert not (inner_train & test or validation & test)
            inner_validation.extend(validation)
        assert Counter(inner_validation) == Counter({digest: 1 for digest in train})


@pytest.mark.parametrize("data", [[], groups(positive=4), groups(negative=4), groups() + [groups()[0]],
                                 [{"video_sha256": "not-a-sha", "binary_label": 1}],
                                 [{"video_sha256": "a" * 64, "binary_label": "1"}]])
def test_invalid_or_insufficient_group_metadata_cannot_form_outer_folds(data):
    with pytest.raises(ValueError):
        stratified_group_folds(data)


def test_infeasible_inner_class_support_fails_instead_of_reusing_outer_test_data():
    with pytest.raises(ValueError, match="Each binary class"):
        nested_group_folds(groups(positive=5, negative=5), outer_splits=2, inner_splits=4)


def test_metrics_expose_all_positive_failure_and_undefined_rates():
    result = safe_binary_metrics([0, 0, 1, 1], [.9, .9, .9, .9])
    assert result["true_positive"] == result["false_positive"] == 2
    assert result["recall"] == 1 and result["specificity"] == 0 and result["balanced_accuracy"] == .5
    single = safe_binary_metrics([1], [.9])
    assert single["recall"] == 1 and single["specificity"] is single["balanced_accuracy"] is None
    empty = safe_binary_metrics([], [])
    assert empty["samples"] == 0 and empty["accuracy"] is empty["balanced_accuracy"] is None


@pytest.mark.parametrize("labels, scores, threshold", [
    ([0, 1], [.2], .5), ([0, 2], [.2, .8], .5), ([0, 1], [float("nan"), .8], .5),
    ([0, 1], [-.1, .8], .5), ([0, 1], [.2, 1.1], .5), ([0, 1], [.2, .8], float("inf")),
    ([0, 1], [.2, .8], True),
])
def test_invalid_metrics_inputs_cannot_be_reported_as_success(labels, scores, threshold):
    with pytest.raises(ValueError):
        safe_binary_metrics(labels, scores, threshold)


def test_threshold_ties_prefer_both_class_sensitivities_over_an_all_positive_answer():
    labels, probabilities = [0, 0, 1, 1], [.8, .6, .7, .5]
    threshold = choose_sensitivity_balanced_threshold(labels, probabilities)
    result = safe_binary_metrics(labels, probabilities, threshold)
    assert .6 < threshold <= .7
    assert result["balanced_accuracy"] == result["recall"] == result["specificity"] == .5


def test_threshold_selection_retains_half_when_it_is_optimal_and_requires_both_classes():
    assert choose_sensitivity_balanced_threshold([0, 0, 1, 1], [.1, .4, .6, .9]) == .5
    for labels, scores in (([], []), ([0], [.1]), ([1], [.9])):
        with pytest.raises(ValueError, match="both binary classes"):
            choose_sensitivity_balanced_threshold(labels, scores)


def test_oof_metrics_use_each_inner_selected_threshold_without_retuning_on_outer_labels():
    source = groups(positive=1, negative=1)
    predictions = [{"video_sha256": source[0]["video_sha256"], "probability": .4, "threshold": .3},
                   {"video_sha256": source[1]["video_sha256"], "probability": .6, "threshold": .7}]
    original = deepcopy((source, predictions))
    result = evaluate_oof_predictions(source, predictions)
    assert result["balanced_accuracy"] == result["accuracy"] == 1
    assert safe_binary_metrics([1, 0], [.4, .6], .5)["balanced_accuracy"] == 0
    assert (source, predictions) == original
    assert "video_sha256" not in json.dumps(result) and "private" not in json.dumps(result)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "foreign", "invalid_threshold", "invalid_probability"])
def test_oof_predictions_require_exactly_one_valid_heldout_record_per_video(defect):
    source = groups(positive=1, negative=1)
    predictions = [{"video_sha256": group["video_sha256"], "probability": .8, "threshold": .5} for group in source]
    if defect == "missing":
        predictions.pop()
    elif defect == "duplicate":
        predictions.append(dict(predictions[0]))
    elif defect == "foreign":
        predictions[0]["video_sha256"] = "f" * 64
    elif defect == "invalid_threshold":
        predictions[0]["threshold"] = None
    else:
        predictions[0]["probability"] = float("nan")
    with pytest.raises(ValueError):
        evaluate_oof_predictions(source, predictions)
