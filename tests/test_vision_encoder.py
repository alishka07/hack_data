"""Offline visual feature contracts using synthetic arrays and a fake Torch API."""
from contextlib import nullcontext
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from worker import vision_encoder as vision
from worker.video import AnalysisError, FrameRef, VideoInfo


def test_full_frame_letterbox_retains_both_edges_and_uses_rgb_imagenet_normalization():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, :40, 0] = 255
    image[:, 40:, 2] = 255
    previous = image.copy()
    encoded = vision.preprocess_rgb(image)
    assert encoded.shape == (3, 224, 224) and encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    np.testing.assert_array_equal(image, previous)
    np.testing.assert_allclose(encoded[:, 0, :], 0, atol=1e-6)
    np.testing.assert_allclose(encoded[:, -1, :], 0, atol=1e-6)
    np.testing.assert_allclose(encoded[:, 100, 0], (np.array([1, 0, 0]) - vision._MEAN) / vision._STD, atol=1e-6)
    np.testing.assert_allclose(encoded[:, 100, -1], (np.array([0, 0, 1]) - vision._MEAN) / vision._STD, atol=1e-6)
    np.testing.assert_allclose(encoded, vision.preprocess_rgb(image.astype(np.float32) / 255), atol=1e-6)


@pytest.mark.parametrize("image", [None, np.zeros((0, 10, 3), dtype=np.uint8),
                                  np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 4), dtype=np.uint8),
                                  np.zeros((10, 10, 3), dtype=np.int16), np.ones((1, 1, 3)) * 255,
                                  np.ones((1, 1, 3)) * np.nan, np.ones((1, 1, 3)) * np.inf,
                                  np.ones((1, 1, 3)) * -.1])
def test_preprocessing_rejects_ambiguous_or_invalid_rgb_data(image):
    with pytest.raises(ValueError):
        vision.preprocess_rgb(image)


def test_temporal_pooling_has_known_statistics_and_preserves_time_direction():
    features = np.tile(np.array([[1.], [3.], [5.]], dtype=np.float32), (1, 384))
    original = features.copy()
    pooled = vision.pool_features(features)
    assert pooled.shape == (1152,) and pooled.dtype == np.float32
    np.testing.assert_allclose(pooled[:384], 3)
    np.testing.assert_allclose(pooled[384:768], np.sqrt(8 / 3), rtol=1e-6)
    np.testing.assert_allclose(pooled[768:], 4)
    reversed_pool = vision.pool_features(features[::-1])
    np.testing.assert_allclose(reversed_pool[:768], pooled[:768])
    np.testing.assert_allclose(reversed_pool[768:], -pooled[768:])
    np.testing.assert_array_equal(features, original)
    single = vision.pool_features(features[:1])
    np.testing.assert_array_equal(single[384:], 0)


@pytest.mark.parametrize("features", [np.empty((0, 384)), np.zeros((2, 383)), np.zeros(384),
                                     np.full((1, 384), np.nan), np.full((1, 384), np.inf),
                                     np.full((1, 384), "1"), np.ones((1, 384), dtype=complex),
                                     np.full((1, 384), np.finfo(np.float64).max)])
def test_pooling_rejects_missing_wrong_size_and_nonfinite_features(features):
    with pytest.raises(ValueError):
        vision.pool_features(features)


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
        self.batches, self.calls = [], []
        self.invalid_output = None

    def load_state_dict(self, state, strict):
        self.calls.append(("weights", state, strict))

    def to(self, device):
        self.calls.append(("device", device))

    def eval(self):
        self.calls.append(("eval",))

    def requires_grad_(self, enabled):
        self.calls.append(("grad", enabled))

    def __call__(self, batch):
        self.batches.append(batch.copy())
        values = np.repeat(batch.mean(axis=(1, 2, 3))[:, None], 384, axis=1)
        return FakeTensor(values if self.invalid_output is None else self.invalid_output)


@pytest.fixture
def fake_encoder(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    (tmp_path / "source/hubconf.py").write_text("# Synthetic source marker", encoding="utf-8")
    (tmp_path / "dinov2_vits14_pretrain.pth").write_bytes(b"synthetic")
    model, events = FakeModel(), []

    def load_hub(*args, **kwargs):
        events.append(("hub", args, kwargs))
        return model

    def load_weights(*args, **kwargs):
        events.append(("load", args, kwargs))
        return {"synthetic_weight": 1}

    fake = SimpleNamespace(set_num_threads=lambda threads: events.append(("threads", threads)),
                           hub=SimpleNamespace(load=load_hub), load=load_weights,
                           inference_mode=nullcontext, from_numpy=lambda batch: batch)
    monkeypatch.setattr(vision.importlib, "import_module", lambda name: fake if name == "torch" else None)
    return vision.DinoEncoder(tmp_path), model, events


def test_encoder_loads_only_local_code_cpu_safe_weights_and_frozen_evaluation(fake_encoder):
    encoder, model, events = fake_encoder
    assert events[0] == ("threads", 4)
    hub = next(event for event in events if event[0] == "hub")
    assert hub[1][1] == "dinov2_vits14"
    assert hub[2] == {"source": "local", "pretrained": False}
    load = next(event for event in events if event[0] == "load")
    assert load[2] == {"map_location": "cpu", "weights_only": True}
    assert model.calls == [("weights", {"synthetic_weight": 1}, True),
                           ("device", "cpu"), ("eval",), ("grad", False)]
    images = [np.full((224, 224, 3), value, dtype=np.uint8) for value in (0, 40, 80, 120, 160)]
    features = encoder.encode_rgb(images)
    assert features.shape == (5, 384) and features.dtype == np.float32
    assert [len(batch) for batch in model.batches] == [4, 1]
    assert np.all(np.diff(features[:, 0]) > 0)
    assert encoder.encode_rgb([]).shape == (0, 384)
    assert encoder.encode_rgb(images[0]).shape == (1, 384)


@pytest.mark.parametrize("output", [np.zeros((1, 385)), np.full((1, 384), np.nan)])
def test_encoder_rejects_invalid_model_outputs(fake_encoder, output):
    encoder, model, _ = fake_encoder
    model.invalid_output = output
    with pytest.raises(ValueError):
        encoder.encode_rgb([np.zeros((10, 20, 3), dtype=np.uint8)])


def test_missing_assets_fail_before_attempting_to_import_torch(tmp_path, monkeypatch):
    def forbidden_import(name):
        pytest.fail("Missing assets should not import Torch")
    monkeypatch.setattr(vision.importlib, "import_module", forbidden_import)
    with pytest.raises(FileNotFoundError):
        vision.DinoEncoder(tmp_path)
    for threads in (0, True, 1.5):
        with pytest.raises(ValueError):
            vision.DinoEncoder(tmp_path, threads)


def test_clip_features_uses_the_same_rgb_encoding_and_temporal_pooling(fake_encoder, monkeypatch):
    encoder, _, _ = fake_encoder
    rgb = [np.zeros((10, 20, 3), dtype=np.uint8), np.full((10, 20, 3), 255, dtype=np.uint8)]
    monkeypatch.setattr(vision, "sample_video", lambda path, max_frames: (rgb, [0., 2.], 2.1))
    pooled, timestamps, per_frame = encoder.clip_features(Path("unused.mov"), max_frames=2)
    assert timestamps == [0., 2.]
    np.testing.assert_array_equal(pooled, vision.pool_features(per_frame))
    assert per_frame.shape == (2, 384)


def test_sampling_uses_actual_vfr_source_indices_and_returns_rgb(monkeypatch):
    info = VideoInfo(Path("synthetic.mov"), 10.1, 4, 2, (0., .1, .3, 8., 10.), 42.)
    monkeypatch.setattr(vision, "probe_video", lambda path, budget: info)
    calls = []

    def extract(info, targets, directory, budget, height):
        calls.append((targets, height))
        return [FrameRef(info.pts.index(t), t, directory / f"{index}.jpg") for index, t in enumerate(targets)]

    monkeypatch.setattr(vision, "extract_frames", extract)
    monkeypatch.setattr(vision.cv2, "imread", lambda path, flags: np.tile(np.array([1, 2, 3], dtype=np.uint8), (2, 4, 1)))
    images, timestamps, duration = vision.sample_video(Path("synthetic.mov"), 5)
    assert timestamps == [0., .3, 8., 10.]
    assert calls == [([0., .3, 8., 10.], 224)]
    assert duration == 10.1
    assert images[0][0, 0].tolist() == [3, 2, 1]


def test_uniform_sampler_includes_first_and_last_decoded_source_pts(tmp_path):
    video = tmp_path / "synthetic.mov"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=80x48:r=6:d=2", "-an", "-c:v", "libx264", "-y", str(video)], check=True)
    images, timestamps, duration = vision.sample_video(video, max_frames=4)
    assert len(images) == len(timestamps) == 4
    np.testing.assert_allclose(timestamps, [0., 4 / 6, 7 / 6, 11 / 6], atol=1e-5)
    assert duration == pytest.approx(2.)
    assert all(image.dtype == np.uint8 and image.shape[2] == 3 for image in images)


@pytest.mark.parametrize("maximum", [0, 1, True, 2.5, 513])
def test_sampling_limits_are_bounded_and_require_room_for_both_source_ends(maximum):
    with pytest.raises(ValueError):
        vision.sample_video(Path("unused.mov"), maximum)
