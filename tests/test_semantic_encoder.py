"""Synthetic single-request semantic extraction; no model or private media calls."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from worker import semantic_encoder as semantic
from worker.semantic_features import SEMANTIC_KEYS, SEMANTIC_PROMPT
from worker.video import AnalysisError, Budget, FrameRef, VideoInfo


class FakeRuntime:
    def __init__(self, *, invalid=False, fail=False):
        self.calls = []
        self.invalid, self.fail = invalid, fail

    def ask(self, prompt, frames, schema, budget, max_tokens):
        self.calls.append({"prompt": prompt, "frames": list(frames), "schema": schema,
                           "budget": budget, "max_tokens": max_tokens,
                           "dimensions": [cv2.imread(str(frame.path)).shape[:2] for frame in frames]})
        if self.fail:
            raise AnalysisError("model_request_failed", "Synthetic local model failure")
        answer = {key: "unknown" for key in SEMANTIC_KEYS}
        if self.invalid:
            answer[SEMANTIC_KEYS[0]] = "violation"
        return answer


@pytest.fixture
def source(monkeypatch):
    info = VideoInfo(Path("private-label-never-in-prompt.mov"), 10.1, 2000, 448,
                     tuple(index / 10 for index in range(100)), 42.)
    monkeypatch.setattr(semantic, "probe_video", lambda path, budget: info)
    calls = []

    def extract(info, targets, folder, budget, height):
        calls.append((targets, height))
        folder.mkdir(parents=True)
        refs = []
        for position, timestamp in enumerate(targets):
            filename = folder / f"{position}.jpg"
            assert cv2.imwrite(str(filename), np.zeros((448, 2000, 3), dtype=np.uint8))
            refs.append(FrameRef(info.pts.index(timestamp), timestamp, filename))
        return refs

    monkeypatch.setattr(semantic, "extract_frames", extract)
    return info, calls


def test_one_fixed_request_uses_actual_uniform_pts_and_caps_wide_images(tmp_path, source):
    info, extraction_calls = source
    runtime = FakeRuntime()
    budget = Budget(10**12)
    encoder = semantic.SemanticEncoder(tmp_path, tmp_path, runtime)
    answer, metadata = encoder.clip_features(info.path, budget)
    assert len(runtime.calls) == len(extraction_calls) == 1
    call = runtime.calls[0]
    assert call["prompt"] == SEMANTIC_PROMPT
    assert info.path.name not in call["prompt"]
    assert call["max_tokens"] == 768 and call["budget"] is budget
    assert len(call["frames"]) == 8
    assert metadata["source_indices"][0] == 0 and metadata["source_indices"][-1] == 99
    assert metadata["timestamps"] == [info.pts[index] for index in metadata["source_indices"]]
    assert extraction_calls == [(metadata["timestamps"], 448)]
    assert all(max(shape) <= 960 for shape in call["dimensions"])
    assert all(shape == (215, 960) for shape in call["dimensions"])
    assert metadata["frame_dimensions"] == [{"height": 215, "width": 960}] * 8
    assert metadata["duration_sec"] == 10.1
    assert set(answer) == set(SEMANTIC_KEYS)
    assert metadata["recipe"] == semantic.semantic_recipe()
    assert list((tmp_path / "semantic-tmp").iterdir()) == []
    assert all(not frame.path.exists() for frame in call["frames"])


def test_recipe_captures_prompt_schema_sampling_and_generation_limits():
    recipe = semantic.semantic_recipe()
    assert len(recipe["prompt_sha256"]) == len(recipe["schema_sha256"]) == 64
    assert recipe["frame_count"] == 8 and recipe["frame_height"] == 448 and recipe["max_side"] == 960
    assert recipe["max_tokens"] == 768 and recipe["temperature"] == 0


@pytest.mark.parametrize("runtime", [FakeRuntime(invalid=True), FakeRuntime(fail=True)])
def test_invalid_or_failed_observation_never_returns_clean_features_and_cleans_private_frames(tmp_path, source, runtime):
    with pytest.raises(AnalysisError):
        semantic.SemanticEncoder(tmp_path, tmp_path, runtime).clip_features(source[0].path)
    assert len(runtime.calls) == 1
    assert list((tmp_path / "semantic-tmp").iterdir()) == []


def test_invalid_pts_are_rejected_before_extraction_or_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "probe_video", lambda path, budget: VideoInfo(path, 1., 10, 10, (0., float("nan")), 0.))
    runtime = FakeRuntime()
    with pytest.raises(AnalysisError) as error:
        semantic.SemanticEncoder(tmp_path, tmp_path, runtime).clip_features(Path("unused.mov"))
    assert error.value.code == "invalid_timestamps" and not runtime.calls
