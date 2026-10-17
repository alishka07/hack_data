import numpy as np
import pytest

from worker.event_features import contextualize_windows


def test_context_features_preserve_appearance_and_center_only_the_current_video():
    frames = np.zeros((2, 512))
    frames[0, 0], frames[1, 1] = 2, 3
    result = contextualize_windows(frames)
    assert result.shape == (2, 1024)
    np.testing.assert_allclose(result[:, :512].sum(axis=1), 1)
    np.testing.assert_allclose(result[:, 512:].mean(axis=0), 0)
    np.testing.assert_allclose(contextualize_windows(frames * 4), result)
    np.testing.assert_array_equal(contextualize_windows(frames[:1])[:, 512:], 0)
    np.testing.assert_array_equal(frames[:, :2], [[2, 0], [0, 3]])


def test_zero_features_stay_finite_and_bad_shapes_are_rejected():
    np.testing.assert_array_equal(contextualize_windows(np.zeros((1, 512))), 0)
    for value in (np.zeros(512), np.zeros((0, 512)), np.ones((3, 511)), np.full((1, 512), np.nan)):
        with pytest.raises(ValueError):
            contextualize_windows(value)
