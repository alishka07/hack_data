"""Synthetic window bags: no media, pretrained models or private annotations."""
from copy import deepcopy
import json

import numpy as np
import pytest

from worker.event_classifier import fit_event_classifier, predict_event_score, predict_event_windows


def training_bags():
    return ([np.array([[-3., -1.], [-2., 1.]]),
             np.array([[-4., 1.], [-2., -1.], [-3., 0.]]),
             np.array([[2., 0.], [-50., 80.], [-40., -60.]]),
             np.array([[-80., 50.], [3., 1.]])],
            [0, 0, 1, 1],
            [None, np.array([False, False, False]),
             np.array([True, False, False]), np.array([False, True])])


def test_weak_anchor_model_learns_and_aggregates_window_scores():
    bags, labels, masks = training_bags()
    model = fit_event_classifier(bags, labels, masks, regularization=.3, pca_components=8)
    scores = np.array([predict_event_score(model, bag) for bag in bags])
    np.testing.assert_array_equal(scores >= .5, labels)
    for bag in bags:
        windows = predict_event_windows(model, bag)
        assert windows.shape == (len(bag),)
        assert predict_event_score(model, bag) == float(windows.max())
    assert model["pca"]["n_components"] <= 2
    assert len(model["head"]["weights"]) == model["pca"]["n_components"]
    assert model["head"]["final_loss"] < model["head"]["initial_loss"]
    assert all(b <= a for a, b in zip(model["head"]["loss_history"], model["head"]["loss_history"][1:]))
    assert model["head"]["converged"]
    restored = json.loads(json.dumps(model, allow_nan=False))
    np.testing.assert_array_equal(predict_event_windows(restored, bags[0]),
                                  predict_event_windows(model, bags[0]))
    assert not {"labels", "anchor_masks", "hashes", "video_paths"} & set(model)


def test_unannotated_positive_windows_never_enter_pca_or_logistic_loss():
    bags, labels, masks = training_bags()
    original = fit_event_classifier(bags, labels, masks)
    changed = deepcopy(bags)
    changed[2][~masks[2]] = [[1e6, -1e7], [-1e8, 1e9]]
    changed[3][~masks[3]] = [[2e10, -3e11]]
    assert fit_event_classifier(changed, labels, masks) == original
    assert original["eligible_window_count"] == 7


def test_repeating_a_videos_windows_does_not_increase_its_training_weight():
    bags, labels, masks = training_bags()
    model = fit_event_classifier(bags, labels, masks)
    repeated = deepcopy(bags)
    repeated[0] = np.repeat(repeated[0], 40, axis=0)
    other = fit_event_classifier(repeated, labels, masks)
    np.testing.assert_allclose(model["pca"]["mean"], other["pca"]["mean"], atol=1e-12)
    np.testing.assert_allclose(model["pca"]["scale"], other["pca"]["scale"], atol=1e-12)
    for bag in bags:
        np.testing.assert_allclose(predict_event_windows(model, bag),
                                   predict_event_windows(other, bag), atol=1e-7)


def test_preprocessing_moments_balance_classes_and_videos_not_window_counts():
    bags = [np.array([[-4.], [-2.]]), np.array([[2.]]), np.array([[6.]])]
    labels, masks = [0, 1, 1], [None, [True], [True]]
    model = fit_event_classifier(bags, labels, masks, pca_components=8)
    # Negative video receives .5, each positive video .25; all four rows .25.
    np.testing.assert_allclose(model["pca"]["mean"], [.5])
    np.testing.assert_allclose(model["pca"]["scale"], [np.std([-4., -2., 2., 6.])])
    assert model["weighting"] == "balanced_classes_equal_total_per_video_within_class"


def test_constant_features_produce_a_finite_intercept_only_balanced_baseline():
    bags = [np.full((20, 4), 7.), np.full((2, 4), 7.), np.full((1, 4), 7.)]
    model = fit_event_classifier(bags, [0, 0, 1], [None, None, [True]], pca_components=8)
    assert model["pca"]["n_components"] == 0
    assert model["head"]["weights"] == []
    assert model["pca"]["constant_features"] == [True] * 4
    np.testing.assert_allclose(predict_event_windows(model, bags[0]), .5, atol=1e-12)
    json.dumps(model, allow_nan=False)


def test_pca_rank_is_bounded_for_more_features_than_training_windows():
    generator = np.random.default_rng(17)
    bags = [generator.normal(size=(2, 64)), generator.normal(size=(3, 64))]
    model = fit_event_classifier(bags, [0, 1], [None, [False, True, False]], pca_components=16)
    assert 0 < model["pca"]["n_components"] <= 2
    assert np.isfinite(predict_event_windows(model, bags[1])).all()


def test_prediction_does_not_refit_or_mutate_and_fits_are_reproducible():
    bags, labels, masks = training_bags()
    original_bags, original_masks = deepcopy(bags), deepcopy(masks)
    model = fit_event_classifier(bags, labels, masks)
    original_model = deepcopy(model)
    one = predict_event_windows(model, bags[0][:1])[0]
    batch = predict_event_windows(model, np.vstack((bags[0][:1], [1e100, -1e100])))
    assert one == pytest.approx(batch[0], abs=1e-12)
    assert np.isfinite(batch).all()
    assert model == original_model
    assert fit_event_classifier(bags, labels, masks) == model
    for actual, expected in zip(bags, original_bags):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(masks, original_masks):
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mask", [None, [False, False], [1, 0], [True]])
def test_positive_videos_require_a_valid_nonempty_boolean_anchor_mask(mask):
    with pytest.raises(ValueError):
        fit_event_classifier([[[0.], [1.]], [[2.], [3.]]], [0, 1], [None, mask])


@pytest.mark.parametrize(("bags", "labels", "masks"), [
    ([[[0.]], [[1.]]], [0, 0], [None, [True]]),
    ([[[0.]], [[1., 2.]]], [0, 1], [None, [True]]),
    ([[[0.]], [[float("nan")]]], [0, 1], [None, [True]]),
    ([[[0.]], []], [0, 1], [None, []]),
    ([[[0.]], [[1.]]], [0, 1], [None]),
])
def test_invalid_training_bags_are_rejected(bags, labels, masks):
    with pytest.raises(ValueError):
        fit_event_classifier(bags, labels, masks)


def test_prediction_rejects_incompatible_or_corrupt_artifacts():
    bags, labels, masks = training_bags()
    model = fit_event_classifier(bags, labels, masks)
    with pytest.raises(ValueError):
        predict_event_windows(model, [[1., 2., 3.]])
    bad = deepcopy(model)
    bad["pca"]["output_scale"][0] = 0
    with pytest.raises(ValueError):
        predict_event_windows(bad, bags[0])
    with pytest.raises(ValueError):
        predict_event_score(model, np.empty((0, 2)))
