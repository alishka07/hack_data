"""Synthetic action preprocessing, temporal alignment, and offline loader tests."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from worker import action_encoder as action
from worker.video import FrameRef, VideoInfo


def test_letterbox_keeps_visible_frame_edges_rgb_order_and_kinetics_normalization():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :20, 0] = 255
    image[:, 20:, 2] = 255
    before = image.copy()
    result = action.preprocess_rgb(image)
    assert result.shape == (3, 112, 112) and result.dtype == np.float32
    np.testing.assert_array_equal(image, before)
    np.testing.assert_allclose(result[:, 0, :], 0, atol=1e-6)
    np.testing.assert_allclose(result[:, 60, 0], (np.array([1, 0, 0]) - action._MEAN) / action._STD, atol=1e-6)
    np.testing.assert_allclose(result[:, 60, -1], (np.array([0, 0, 1]) - action._MEAN) / action._STD, atol=1e-6)


@pytest.mark.parametrize("image", [None, np.zeros((0, 1, 3), dtype=np.uint8), np.zeros((1, 1, 4)),
                                  np.ones((1, 1, 3)) * 255, np.ones((1, 1, 3)) * np.nan,
                                  np.ones((1, 1, 3), dtype=np.int16)])
def test_invalid_rgb_is_rejected(image):
    with pytest.raises(ValueError):
        action.preprocess_rgb(image)


def test_windows_cover_source_ends_with_fixed_stride_and_explicit_tail():
    assert action.window_plan(3) == [{"start_sec": 0., "end_sec": 3., "center_sec": 1.5}]
    assert [(row["start_sec"], row["end_sec"], row["center_sec"]) for row in action.window_plan(14)] == [
        (0., 8., 4.), (4., 12., 8.), (8., 14., 11.)]
    pts = tuple(index / 10 for index in range(140))
    windows = action.window_metadata(pts, 14.)
    assert windows[0]["sample_timestamps_sec"][0] == 0
    assert windows[-1]["sample_timestamps_sec"][-1] == 13.9
    assert all(len(row["input_indices"]) == 16 for row in windows)
    assert all(row["sample_timestamps_sec"] == [pts[index] for index in row["input_indices"]] for row in windows)


def test_short_clips_repeat_actual_source_frames_without_fabricating_timestamps():
    windows = action.window_metadata([0., .1], .2)
    assert len(windows) == 1
    assert set(windows[0]["sample_timestamps_sec"]) == {0., .1}
    assert windows[0]["input_indices"][0] == 0 and windows[0]["input_indices"][-1] == 1


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf"), 601])
def test_duration_bound_protects_sampler_work(duration):
    with pytest.raises(ValueError):
        action.window_plan(duration)


@pytest.mark.parametrize("timestamps", [[], [0., 0.], [1., 0.], [-1., 0.], [0., 9.],
                                       [False, True], [0., float("nan")]])
def test_foreign_nonmonotonic_and_nonfinite_timestamps_are_rejected(timestamps):
    with pytest.raises(ValueError):
        action.window_metadata(timestamps, 8.)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.calls, self.inputs = [], []
        self.invalid = None

    def load_state_dict(self, state, strict):
        self.calls.append(("state", state, strict))

    def to(self, device):
        self.calls.append(("device", device))

    def eval(self):
        self.calls.append(("eval",))

    def requires_grad_(self, enabled):
        self.calls.append(("grad", enabled))

    def __call__(self, values):
        self.inputs.append(values.copy())
        return FakeTensor(np.full((1, 512), values[:, :, -1].mean() - values[:, :, 0].mean(), dtype=np.float32)
                          if self.invalid is None else self.invalid)


def assets(tmp_path):
    weights = b"synthetic action weights"
    (tmp_path / action.WEIGHTS_FILENAME).write_bytes(weights)
    manifest = {"encoder": "r3d_18", "pretraining": "KINETICS400_V1", "feature_dimension": 512,
                "source_code_version": "torchvision==0.28.0+cpu", "preprocessing": action.PREPROCESS_VERSION,
                "weights_sha256": hashlib.sha256(weights).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.fixture
def fake_encoder(tmp_path, monkeypatch):
    assets(tmp_path)
    model, events = FakeModel(), []
    identity = object()
    torch = SimpleNamespace(set_num_threads=lambda n: events.append(("threads", n)),
                            load=lambda *args, **kwargs: events.append(("load", args, kwargs)) or {"weight": 1},
                            nn=SimpleNamespace(Identity=lambda: identity), inference_mode=nullcontext,
                            from_numpy=lambda values: values)
    torchvision = SimpleNamespace(__version__="0.28.0+cpu", models=SimpleNamespace(video=SimpleNamespace(
        r3d_18=lambda **kwargs: events.append(("build", kwargs)) or model)))
    monkeypatch.setattr(action.importlib, "import_module", lambda name: {"torch": torch, "torchvision": torchvision}[name])
    encoder = action.ActionEncoder(tmp_path)
    return encoder, model, events, identity


def test_offline_loader_hashes_assets_loads_strictly_and_removes_only_pretrained_head(fake_encoder):
    _, model, events, identity = fake_encoder
    assert events[0] == ("threads", 4)
    assert ("build", {"weights": None}) in events
    assert next(event for event in events if event[0] == "load")[2] == {"map_location": "cpu", "weights_only": True}
    assert model.calls == [("state", {"weight": 1}, True), ("device", "cpu"), ("eval",), ("grad", False)]
    assert model.fc is identity


def test_encoder_preserves_temporal_axis_and_maps_supplied_source_times(fake_encoder):
    encoder, model, _, _ = fake_encoder
    images = [np.full((112, 112, 3), value, dtype=np.uint8) for value in range(0, 180, 10)]
    features, windows = encoder.encode_windows(images, [index * .5 for index in range(18)], 9.)
    assert features.shape == (2, 512) and features.dtype == np.float32
    assert all(array.shape == (1, 3, 16, 112, 112) for array in model.inputs)
    assert np.all(features > 0)
    assert len(windows) == 2
    assert encoder.encode_clips([]).shape == (0, 512)
    with pytest.raises(ValueError):
        encoder.encode_windows(images[:1], [0., 1.], 2.)
    with pytest.raises(ValueError):
        encoder.encode_clips([images[:15]])


def test_invalid_encoder_features_never_become_training_data(fake_encoder):
    encoder, model, _, _ = fake_encoder
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for invalid in (np.zeros((1, 400)), np.full((1, 512), np.nan)):
        model.invalid = invalid
        with pytest.raises(ValueError):
            encoder.encode_clips([[image] * 16])


def test_corrupt_weights_are_rejected_before_any_torch_import(tmp_path, monkeypatch):
    assets(tmp_path)
    (tmp_path / action.WEIGHTS_FILENAME).write_bytes(b"corrupt")
    monkeypatch.setattr(action.importlib, "import_module", lambda name: pytest.fail("Corrupt weights must not load Torch"))
    with pytest.raises(ValueError, match="checksum"):
        action.ActionEncoder(tmp_path)


def test_single_decode_selects_all_unique_source_indices_and_returns_rgb(tmp_path, monkeypatch):
    info = VideoInfo(Path("synthetic.mov"), 10., 4, 2, tuple(index / 10 for index in range(100)), 42.)
    monkeypatch.setattr(action, "probe_video", lambda path, budget: info)
    monkeypatch.setenv("FORTE_FEATURE_TEMP_DIR", str(tmp_path))
    calls = []

    def extract(info, targets, directory, budget, height):
        calls.append((targets, height))
        return [FrameRef(info.pts.index(timestamp), timestamp, directory / f"{index}.jpg")
                for index, timestamp in enumerate(targets)]

    monkeypatch.setattr(action, "extract_frames", extract)
    monkeypatch.setattr(action.cv2, "imread", lambda path, flags: np.tile(np.array([1, 2, 3], dtype=np.uint8), (2, 4, 1)))
    images, timestamps, duration, indices = action.sample_video_windows(info.path)
    expected = sorted({index for window in action.window_metadata(info.pts, 10.) for index in window["input_indices"]})
    assert len(calls) == 1 and calls[0] == ([info.pts[index] for index in expected], 112)
    assert indices == expected and timestamps == [info.pts[index] for index in expected]
    assert images[0][0, 0].tolist() == [3, 2, 1] and duration == 10.


def test_clip_features_returns_global_source_indices_and_window_pts(fake_encoder, monkeypatch):
    encoder, _, _, _ = fake_encoder
    images = [np.full((2, 4, 3), value, dtype=np.uint8) for value in (0, 50, 100)]
    monkeypatch.setattr(action, "sample_video_windows", lambda path: (images, [0., 1., 2.], 2.1, [5, 15, 30]))
    features, windows = encoder.clip_features(Path("synthetic.mov"))
    assert features.shape == (1, 512)
    assert windows[0]["source_indices"][0] == 5 and windows[0]["source_indices"][-1] == 30
    assert windows[0]["sample_timestamps_sec"][0] == 0. and windows[0]["sample_timestamps_sec"][-1] == 2.
