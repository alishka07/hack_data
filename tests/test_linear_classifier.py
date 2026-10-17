"""Numeric, synthetic-only tests for the frozen-feature classifier."""
from copy import deepcopy
import json

import numpy as np
import pytest

from worker.linear_classifier import choose_threshold, fit_linear_classifier, predict_probabilities


def test_separable_data_loss_descends_and_artifact_round_trips_without_targets():
    features = np.array([[-4., 2.], [-3., 1.], [-2., 3.], [2., 3.], [3., 1.], [4., 2.]])
    targets = np.array([0, 0, 0, 1, 1, 1])
    model = fit_linear_classifier(features, targets)
    probabilities = predict_probabilities(model, features)
    np.testing.assert_array_equal(probabilities >= .5, targets)
    assert probabilities[0] < .2 and probabilities[-1] > .8
    assert model["final_loss"] < model["initial_loss"] - .2
    assert all(b <= a for a, b in zip(model["loss_history"], model["loss_history"][1:]))
    assert model["n_iter"] == len(model["loss_history"]) - 1
    assert model["feature_dim"] == 2 and model["class_weight"] == "balanced"
    assert not {"labels", "y", "targets", "hashes", "training_labels", "video_hashes"} & set(model)
    restored = json.loads(json.dumps(model, allow_nan=False))
    np.testing.assert_array_equal(predict_probabilities(restored, features), probabilities)


def test_balanced_sample_weights_do_not_follow_the_majority_with_no_signal():
    features = np.full((100, 3), 7.)
    targets = np.array([0] * 99 + [1])
    model = fit_linear_classifier(features, targets)
    np.testing.assert_allclose(predict_probabilities(model, features), .5, atol=1e-12)
    np.testing.assert_array_equal(model["weights"], [0., 0., 0.])
    assert model["constant_features"] == [True, True, True]
    assert model["converged"]


def test_intercept_is_unpenalized_under_imbalanced_separable_data():
    features = np.array([[0.]] * 39 + [[10.]])
    targets = np.array([0] * 39 + [1])
    model = fit_linear_classifier(features, targets, regularization=10, max_iter=2000)
    probabilities = predict_probabilities(model, features)
    weights = np.where(targets == 1, .5, .5 / 39)
    assert abs(float(weights @ (probabilities - targets))) < 1e-6
    assert abs(model["bias"]) > .1
    np.testing.assert_array_equal(probabilities >= .5, targets)


def test_constant_columns_and_large_finite_features_are_stable():
    features = np.array([[-1e308, 1e308, 0.], [-5e307, 1e308, 0.],
                         [5e307, 1e308, 0.], [1e308, 1e308, 0.]])
    targets = [0, 0, 1, 1]
    model = fit_linear_classifier(features, targets)
    assert model["constant_features"] == [False, True, True]
    assert model["scale"][1:] == [1., 1.]
    probabilities = predict_probabilities(model, features)
    assert np.isfinite(probabilities).all()
    np.testing.assert_array_equal(probabilities >= .5, targets)
    json.dumps(model, allow_nan=False)


def test_sigmoid_remains_finite_when_logits_would_overflow_naive_exp():
    model = {"feature_dim": 1, "mean": [0.], "scale": [1.], "weights": [1.], "bias": 0.}
    with np.errstate(over="raise"):
        probabilities = predict_probabilities(model, [[-1000.], [0.], [1000.]])
    np.testing.assert_array_equal(probabilities, [0., .5, 1.])


def test_high_dimensional_frozen_features_need_no_feature_hessian():
    generator = np.random.default_rng(17)
    features = generator.normal(size=(12, 256))
    targets = np.array([0] * 6 + [1] * 6)
    features[:, 0] += 4 * (2 * targets - 1)
    model = fit_linear_classifier(features, targets)
    np.testing.assert_array_equal(predict_probabilities(model, features) >= .5, targets)
    assert model["feature_dim"] == 256
    assert model["final_loss"] < model["initial_loss"]


def test_scaling_uses_training_data_and_is_unchanged_by_prediction_batch():
    features = np.array([[-3., 5.], [-1., 5.], [1., 5.], [3., 5.]])
    model = fit_linear_classifier(features, [0, 0, 1, 1])
    np.testing.assert_allclose(model["mean"], features.mean(axis=0))
    np.testing.assert_allclose(model["scale"], [features[:, 0].std(), 1.])
    one = predict_probabilities(model, [1., 5.])
    batch = predict_probabilities(model, [[1., 5.], [1e100, -1e100]])
    assert one.shape == (1,) and one[0] == batch[0]
    assert predict_probabilities(model, np.empty((0, 2))).shape == (0,)
    assert np.isfinite(batch).all()


def test_training_prediction_and_threshold_selection_preserve_their_inputs():
    features = np.array([[-2., 8.], [-1., 8.], [1., 8.], [2., 8.]])
    targets = np.array([0, 0, 1, 1])
    original_features, original_targets = features.copy(), targets.copy()
    model = fit_linear_classifier(features, targets)
    original_model = deepcopy(model)
    probabilities = predict_probabilities(model, features)
    original_probabilities = probabilities.copy()
    choose_threshold(targets, probabilities)
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(targets, original_targets)
    np.testing.assert_array_equal(probabilities, original_probabilities)
    assert model == original_model


def test_training_is_reproducible_and_label_reversal_reverses_predictions():
    features = [[-4., -1.], [-2., 2.], [-1., 0.], [1., 2.], [2., 0.], [4., -1.]]
    targets = np.array([0, 0, 0, 1, 1, 1])
    model = fit_linear_classifier(features, targets)
    assert fit_linear_classifier(features, targets) == model
    reversed_model = fit_linear_classifier(features, 1 - targets)
    np.testing.assert_allclose(predict_probabilities(reversed_model, features),
                               1 - predict_probabilities(model, features), atol=1e-8)


def test_threshold_maximizes_balanced_accuracy_and_retains_midpoint_when_suitable():
    assert choose_threshold([0, 0, 1, 1], [.1, .4, .6, .9]) == .5
    threshold = choose_threshold([0, 0, 1, 1], [.1, .2, .3, .4])
    np.testing.assert_array_equal(np.array([.1, .2, .3, .4]) >= threshold, [0, 0, 1, 1])


def test_threshold_breaks_balanced_accuracy_ties_in_favor_of_recall():
    # The scores rank both labels incorrectly. All-positive and all-negative
    # decisions tie at balanced accuracy .5; recall chooses all-positive.
    threshold = choose_threshold([0, 1], [.8, .2])
    assert threshold <= .2
    assert choose_threshold([0, 1], [.5, .5]) == .5


@pytest.mark.parametrize(("targets", "scores"), [([], []), ([0, 0], [.1, .9]), ([1], [.99])])
def test_threshold_defaults_to_half_without_both_classes(targets, scores):
    assert choose_threshold(targets, scores) == .5


@pytest.mark.parametrize(("features", "targets"), [
    ([], []), ([[], []], [0, 1]), ([0., 1.], [0, 1]),
    ([[0.], [float("nan")]], [0, 1]), ([[0.], [float("inf")]], [0, 1]),
    ([[0.], [1.]], [0]), ([[0.], [1.]], [[0], [1]]),
    ([[0.], [1.]], [0, 2]), ([[0.], [1.]], [0, float("nan")]),
    ([[0.], [1.]], [0, 0]), ([["0"], ["1"]], [0, 1]),
])
def test_invalid_training_data_is_rejected(features, targets):
    with pytest.raises(ValueError):
        fit_linear_classifier(features, targets)


@pytest.mark.parametrize("regularization", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_regularization_is_rejected(regularization):
    with pytest.raises(ValueError):
        fit_linear_classifier([[0.], [1.]], [0, 1], regularization=regularization)


@pytest.mark.parametrize("max_iter", [0, -1, True, 1.5])
def test_invalid_iteration_limits_are_rejected(max_iter):
    with pytest.raises(ValueError):
        fit_linear_classifier([[0.], [1.]], [0, 1], max_iter=max_iter)


def test_zero_regularization_and_small_iteration_budget_return_valid_artifacts():
    model = fit_linear_classifier([[-1.], [1.]], [0, 1], regularization=0, max_iter=1)
    assert model["n_iter"] == 1 and model["final_loss"] < model["initial_loss"]
    assert not model["converged"]
    assert np.isfinite(predict_probabilities(model, [[-1.], [1.]])).all()


@pytest.mark.parametrize("change", [
    {"feature_dim": True}, {"mean": []}, {"weights": [float("nan")]},
    {"scale": [0.]}, {"scale": [-1.]}, {"bias": float("inf")},
    {"constant_features": [1]},
])
def test_invalid_prediction_artifact_is_rejected(change):
    model = {"feature_dim": 1, "mean": [0.], "scale": [1.], "weights": [1.], "bias": 0., **change}
    with pytest.raises(ValueError):
        predict_probabilities(model, [[1.]])


def test_prediction_rejects_wrong_dimensions_and_nonfinite_values():
    model = fit_linear_classifier([[-1.], [1.]], [0, 1])
    with pytest.raises(ValueError):
        predict_probabilities(model, [[1., 2.]])
    with pytest.raises(ValueError):
        predict_probabilities(model, [[float("nan")]])


@pytest.mark.parametrize(("targets", "scores"), [
    ([0, 1], [.2]), ([0, 1], [.2, float("nan")]), ([0, 1], [-.1, .5]),
    ([0, 1], [.1, 1.1]), ([0, 2], [.1, .9]), ([0, 1], [[.1], [.9]]),
])
def test_invalid_threshold_inputs_are_rejected(targets, scores):
    with pytest.raises(ValueError):
        choose_threshold(targets, scores)
