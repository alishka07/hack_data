"""Hybrid boundaries: original pixels/PTS, unconditional coverage, failed jobs.

All frames and model answers here are synthetic. No private data or real local
model service is needed, and no HTTP/socket operation is performed.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from worker import pipeline as pipeline_module
from worker.pipeline import Pipeline, screening_plan
from worker.scene import collect_scene, detail_views
from worker.video import AnalysisError, Budget, FrameCache, FrameRef, VideoInfo, nearest_index, window_targets


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_optional_model_assets(monkeypatch):
    monkeypatch.setenv("FORTE_POSE_BACKEND", "disabled")
    monkeypatch.setenv("FORTE_OBJECT_BACKEND", "disabled")


def test_unseen_scene_region_crops_original_pixels_and_retains_source_pts(tmp_path):
    pts = (0., .17, .61, 1.19)
    info = VideoInfo(tmp_path / "synthetic.mov", 1.2, 300, 200, pts, 83.)
    cache = FrameCache(info, tmp_path / "frames", Budget(time.time() + 30))
    frames = []
    for index, timestamp in enumerate(pts):
        image = np.zeros((200, 300, 3), np.uint8)
        image[:, 150:] = [40, 150, 240]
        path = tmp_path / f"original-{index}.jpg"
        assert cv2.imwrite(str(path), image)
        frames.append(FrameRef(index, timestamp, path))
    cache.seed(frames, 0)
    # The proposed counter is away from any fixed center/lower-camera default.
    scene = collect_scene([{"zone": "counter", "status": "supported", "bbox": [.62, .25, .91, .75],
                            "frame_indices": [2]}], frames)
    result = detail_views(cache, frames, scene, .61)
    assert all(frame in result for frame in frames)
    crops = [frame for frame in result if frame.crop_bbox is not None]
    assert [frame.index for frame in crops] == [1, 2]
    for crop in crops:
        assert crop.timestamp == pts[crop.index]
        assert crop.crop_bbox == (.62, .25, .91, .75)
        assert crop.view == "counter detail; full scene supplied separately"
        pixels = cv2.imread(str(crop.path))
        assert pixels.shape[:2] == (100, 87)
        assert pixels.mean(axis=(0, 1)) == pytest.approx([40, 150, 240], abs=3)
    assert len(cache.crops) == 2
    for maximum in (4, 5, 8):
        bounded = detail_views(cache, frames, scene, .61, max_frames=maximum)
        assert len(bounded) <= maximum
        assert any(frame.crop_bbox is not None for frame in bounded)
    assert len(detail_views(cache, frames, collect_scene([], frames), .61)) == len(frames)


def test_uniform_plan_survives_trigger_flood_and_every_trigger_has_a_source_view():
    pts = tuple(round(i / 10, 1) for i in range(201))
    info = VideoInfo(Path("synthetic.mov"), 20., 640, 360, pts, 5.)
    triggers = [{"start_sec": pts[i], "end_sec": pts[i], "timecode_sec": pts[i],
                 "reason": "synthetic_trigger", "source_indices": [i]} for i in range(1, 200)]
    windows = screening_plan(info, triggers)
    assert all(window in windows for window in window_targets(info.duration))
    inspected = [pts[nearest_index(pts, target)] for window in windows for target in window]
    assert inspected[0] == pts[0]
    assert max(inspected) == pts[-1]
    assert all(min(abs(target - trigger["timecode_sec"]) for target in inspected) <= .25 for trigger in triggers)
    assert len(windows) > len(window_targets(info.duration))


def test_failed_trigger_flood_never_returns_completed_clean_verdict(tmp_path, monkeypatch):
    from worker import temporal
    source = VideoInfo(Path("synthetic.mov"), 6., 320, 192,
                       tuple(round(i / 10, 1) for i in range(61)), 80.)
    clock = {"now": 0.}
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock["now"])
    monkeypatch.setattr(pipeline_module, "probe_video", lambda path, budget: source)

    class MemoryCache:
        def __init__(self, info, directory, budget):
            self.frames = {}

        def get(self, targets, height=640):
            result = []
            for target in targets:
                index = nearest_index(source.pts, target)
                frame = FrameRef(index, source.pts[index], tmp_path / "unused.jpg")
                self.frames[(index, height)] = frame
                result.append(frame)
            return result

    monkeypatch.setattr(pipeline_module, "FrameCache", MemoryCache)
    monkeypatch.setattr(temporal, "temporal_proposals", lambda *args: {"status": "disabled", "proposals": []})
    monkeypatch.setattr(pipeline_module, "analyze_perception", lambda *args: {
        "detections": [], "proposals": [{"timecode_sec": 1.2, "start_sec": 1., "end_sec": 1.4,
                                          "source_indices": [12], "reason": "synthetic_trigger"}],
        "coverage": {"sample_rate": 5., "frames_analyzed": 31, "ending_included": True,
                     "complete": True, "max_gap_sec": .2, "unknowns": []},
        "backend": "synthetic", "scene": {}, "motion": [], "visibility_gaps": [], "timings": {}})

    class ExpiringRuntime:
        calls = 0

        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            self.calls += 1
            clock["now"] = 101.
            answer = {"visibility": "clear", "visible_facts": [],
                      "transaction": {"kind": "unknown", "phase": "unknown", "frame_indices": []},
                      "candidates": []}
            if "scene_regions" in schema["properties"]:
                answer["scene_regions"] = []
            return answer

    runtime = ExpiringRuntime()
    with pytest.raises(AnalysisError) as error:
        Pipeline(ROOT, tmp_path, runtime).run({"id": "expiry", "video_path": "synthetic.mov", "deadline_at": 100.},
                                             lambda *_: None)
    assert error.value.code == "deadline_exceeded"
    assert runtime.calls == 1
    assert not (tmp_path / "jobs/expiry/timings.json").exists()


def test_runtime_payload_preserves_crop_identity_and_outside_unknown(tmp_path):
    from worker.runtime import LocalRuntime
    path = tmp_path / "synthetic.jpg"
    assert cv2.imwrite(str(path), np.zeros((20, 30, 3), np.uint8))
    frame = FrameRef(42, 1.407, path, "counter detail", (.62, .25, .91, .75))
    recorded = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}).encode()

    class Opener:
        def open(self, request, timeout):
            recorded.update(json.loads(request.data))
            return Response()

    runtime = LocalRuntime(ROOT, tmp_path)
    runtime.opener = Opener()
    schema = {"type": "object", "required": ["ok"], "additionalProperties": False,
              "properties": {"ok": {"type": "boolean"}}}
    assert runtime.ask("Synthetic task", [frame], schema, Budget(time.time() + 30)) == {"ok": True}
    content = recorded["messages"][0]["content"]
    assert "source index 42" in content[1]["text"]
    assert "elapsed source time 1.407" in content[1]["text"]
    assert "counter detail" in content[1]["text"]
    assert "outside this crop is unknown" in content[1]["text"]
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_counter_geometry_is_not_reused_after_a_camera_cut():
    frames = [FrameRef(0, 0., Path("unused-0.jpg")), FrameRef(1, .2, Path("unused-1.jpg")),
              FrameRef(2, .6, Path("unused-2.jpg")), FrameRef(3, .8, Path("unused-3.jpg"))]
    scene = collect_scene([{"zone": "counter", "status": "supported", "bbox": [.1, .2, .7, .8],
                            "frame_indices": [0]}], frames)
    scene["scene_change_timestamps"] = [.4]

    class NoCrop:
        def crop(self, *args):
            pytest.fail("A counter box from the previous camera segment must not crop later frames")

    assert detail_views(NoCrop(), frames, scene, .6) == frames


def test_zone_transition_discovered_after_first_screen_gets_its_own_screen(tmp_path, monkeypatch):
    from worker import temporal
    source = VideoInfo(Path("synthetic.mov"), 6., 320, 192,
                       tuple(round(i / 10, 1) for i in range(61)), 80.)
    image_path = tmp_path / "source.jpg"
    assert cv2.imwrite(str(image_path), np.full((192, 320, 3), 100, np.uint8))
    monkeypatch.setattr(pipeline_module, "probe_video", lambda path, budget: source)

    class MemoryCache:
        def __init__(self, info, directory, budget):
            self.frames = {}

        def get(self, targets, height=640):
            result = []
            for target in targets:
                index = nearest_index(source.pts, target)
                frame = FrameRef(index, source.pts[index], image_path)
                self.frames[(index, height)] = frame
                result.append(frame)
            return result

    monkeypatch.setattr(pipeline_module, "FrameCache", MemoryCache)
    monkeypatch.setattr(temporal, "temporal_proposals", lambda *args: {"status": "disabled", "proposals": []})

    def local_perception(info, cache, budget, directory):
        frames = cache.get([source.pts[i] for i in range(0, 61, 2)], 320)
        detections = [{"id": f"person-{frame.index}", "source_index": frame.index,
                       "timestamp_sec": frame.timestamp, "kind": "person", "track_id": "track-1",
                       "continuity": "new" if frame.index == 0 else "adjacent_detection", "confidence": .8,
                       "label": "Visible person", "bbox": [.15, .1, .45, .9] if frame.index < 12 else [.35, .1, .65, .9]}
                      for frame in frames]
        return {"detections": detections, "proposals": [], "backend": "synthetic",
                "coverage": {"sample_rate": 5., "frames_analyzed": 31, "ending_included": True,
                             "complete": True, "max_gap_sec": .2, "unknowns": []},
                "motion": [], "visibility_gaps": [], "scene": {}, "timings": {}}

    monkeypatch.setattr(pipeline_module, "analyze_perception", local_perception)

    class SceneRuntime:
        def __init__(self):
            self.frames = []

        def ask(self, prompt, frames, schema, budget, max_tokens=512):
            self.frames.append(frames)
            answer = {"visibility": "clear", "visible_facts": [],
                      "transaction": {"kind": "unknown", "phase": "unknown", "frame_indices": []}, "candidates": []}
            if "scene_regions" in schema["properties"]:
                answer["scene_regions"] = [{"zone": "counter", "status": "supported",
                                             "bbox": [.4, .1, .9, .9], "frame_indices": [0]}]
            return answer

    runtime = SceneRuntime()
    _, details, _ = Pipeline(ROOT, tmp_path, runtime).run(
        {"id": "zone", "video_path": "synthetic.mov", "deadline_at": time.time() + 30}, lambda *_: None)
    assert len(runtime.frames) > len(window_targets(source.duration))
    assert any(any(frame.timestamp == 1.2 for frame in request) for request in runtime.frames[1:])
    assert details["coverage"]["proposal_windows"] >= 1
    assert details["coverage"]["windows_completed"] == details["coverage"]["windows_planned"] == len(runtime.frames)
    assert details["coverage"]["uniform_windows"] == 1
