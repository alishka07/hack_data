"""Label-free context features shared by event training and video inference."""
from __future__ import annotations

import numpy as np


FEATURE_VERSION = "r3d18_l2_absolute_and_clip_centered_v1"


def contextualize_windows(features):
    """Retain appearance and local changes relative to this video's context.

    Normalization uses only the supplied video's pixels, with no fitted corpus
    statistics. A window bag has 512 frozen action features per row; the output
    concatenates the unit vector and its difference from the video's mean.
    """
    values = np.asarray(features, dtype=np.float64)
    if (values.ndim != 2 or not len(values) or values.shape[1] != 512
            or not np.isfinite(values).all()):
        raise ValueError("Action features must be a finite nonempty Nx512 matrix")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norm, 1e-12)
    result = np.concatenate((normalized, normalized - normalized.mean(axis=0)), axis=1)
    return result.astype(np.float32)
