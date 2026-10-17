import numpy as np
import pytest

from scripts.train_event_detector import select_configuration, weak_anchor_mask


def test_weak_points_select_nearby_windows_with_fallback_at_video_edges():
    windows = [{"center_sec": value} for value in (4, 8, 12, 16)]
    np.testing.assert_array_equal(weak_anchor_mask(windows, [0, 9, 18]), [True, True, False, True])
    # A second annotation does not label the unobserved intervening windows.
    np.testing.assert_array_equal(weak_anchor_mask(windows, [4, 16]), [True, False, False, True])
    for anchors in ([], [float("nan")], [-1]):
        with pytest.raises(ValueError):
            weak_anchor_mask(windows, anchors)


def test_inner_selection_rejects_folds_that_escape_outer_training_before_fitting():
    protocol = {"stronger_model": {"regularization_candidates": [3]}}
    folds = [{"train_ids": ["training"], "validation_ids": ["outer_test"]}]
    with pytest.raises(ValueError, match="escapes"):
        select_configuration("event", ["training", "validation"], folds, {}, {}, {}, {}, protocol)
