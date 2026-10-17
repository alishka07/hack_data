"""Perception safety/PTS boundaries use synthetic local pixels, not labels."""
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from worker import perception
from worker.perception import (IoUTracker, PerceptionConfig, YOLOXPersonDetector,
                               analyze_perception, decode_yolox, normalize_scene, sample_targets)
from worker.video import AnalysisError, Budget, FrameRef, VideoInfo


@pytest.fixture(autouse=True)
def isolate_optional_models(monkeypatch):
    # These tests target deterministic detection/PTS fixtures. Real optional
    # models are exercised in test_perception_models and explicit smoke runs.
    monkeypatch.setenv("FORTE_POSE_BACKEND", "disabled")
    monkeypatch.setenv("FORTE_OBJECT_BACKEND", "disabled")
    monkeypatch.setenv("FORTE_TRACKER", "adjacent_iou")


def info(pts=(0., .2, .4, .6), duration=.65):
    return VideoInfo(Path("local.mov"), duration, 320, 192, pts, 7.)


def test_sample_schedule_includes_final_irregular_source_pts_without_duplicates():
    source = info((0., .03, .22, .43, .81, .97), 1.)
    targets = sample_targets(source, 5)
    assert targets[0] == 0
    assert targets[-1] == .97
    assert targets == sorted(set(targets))
    assert set(targets) <= set(source.pts)
    assert len(targets) == 5


def test_tracker_disappearance_is_unknown_gap_and_reappearance_has_new_id():
    tracker = IoUTracker(max_gap=1.)
    box = {"bbox": [.1, .1, .4, .9], "confidence": .8}
    first, _ = tracker.update([box], FrameRef(0, 0., Path("0.jpg")))
    second, gaps = tracker.update([], FrameRef(5, .2, Path("1.jpg")))
    third, _ = tracker.update([box], FrameRef(10, .4, Path("2.jpg")))
    assert second == []
    assert gaps == [{"start_sec": 0., "end_sec": .2, "source_indices": [0, 5],
                     "reason": "track_detection_gap", "track_id": first[0]["track_id"]}]
    assert first[0]["track_id"] != third[0]["track_id"]
    assert "absence" not in str(gaps)


def test_tracker_keeps_id_only_for_adjacent_geometric_support_and_expires_clock_gap():
    tracker = IoUTracker(max_gap=.8)
    box = {"bbox": [.1, .1, .4, .9], "confidence": .8}
    first, _ = tracker.update([box], FrameRef(0, 0., Path("0.jpg")))
    second, _ = tracker.update([box], FrameRef(5, .2, Path("1.jpg")))
    third, gaps = tracker.update([box], FrameRef(90, 4., Path("2.jpg")))
    assert first[0]["track_id"] == second[0]["track_id"]
    assert first[0]["track_id"] != third[0]["track_id"]
    assert gaps[0]["start_sec"] == .2 and gaps[0]["end_sec"] == 4.


def test_raw_yolox_decode_maps_letterboxed_boxes_and_only_person_score():
    # 128px input => (16^2 + 8^2 + 4^2) anchors.
    raw = np.zeros((1, 336, 85), dtype=np.float32)
    # At stride8/grid(0,0), center16,32 and size16,32; source128x64.
    raw[0, 0, :6] = [2., 4., np.log(2.), np.log(4.), .9, .8]
    # A high-confidence non-person anchor must not create a person box.
    raw[0, 10, 4] = 1.
    raw[0, 10, 6] = 1.
    result = decode_yolox(raw, (128, 128), (128, 64), 1., .35)
    assert len(result) == 1
    assert result[0]["bbox"] == pytest.approx([.125, .125, .375, .375])
    assert result[0]["confidence"] == pytest.approx(.72)


@pytest.mark.parametrize("change", ["shape", "nan", "invalid_probability"])
def test_incompatible_model_outputs_fail_instead_of_empty_detections(change):
    raw = np.zeros((1, 336, 85), dtype=np.float32)
    if change == "shape":
        raw = raw[:, :10]
    elif change == "nan":
        raw[0, 0, 0] = np.nan
    else:
        raw[0, 0, 5] = 10.
    with pytest.raises(AnalysisError, match="person model") as error:
        decode_yolox(raw, (128, 128), (128, 64), 1., .35)
    assert error.value.code == "person_output_invalid"


@pytest.mark.parametrize("setting,value", [("FORTE_PERCEPTION_FPS", "nan"), ("FORTE_PERCEPTION_FPS", "0"),
                                            ("FORTE_PERCEPTION_BACKEND", "bytetrack"),
                                            ("FORTE_PERCEPTION_HEIGHT", "no")])
def test_bad_detector_configuration_is_explicit_failure(monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    with pytest.raises(AnalysisError) as error:
        PerceptionConfig.from_env()
    assert error.value.code == "perception_config_invalid"


def test_explicit_missing_model_does_not_silently_fallback(tmp_path):
    with pytest.raises(AnalysisError) as error:
        perception.build_detector(PerceptionConfig(model_path=tmp_path / "missing.onnx"))
    assert error.value.code == "person_asset_invalid"


def test_untrusted_model_hash_is_checked_before_loading_native_onnx(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a reviewed model")
    manifest = model.with_suffix(".json")
    manifest.write_text('{"format":"yolox-raw-coco-v1","license":"Apache-2.0",'
                        '"input_size":[416,416],"sha256":"' + "0" * 64 + '","source_url":"https://example.test/model"}')
    with pytest.raises(AnalysisError) as error:
        YOLOXPersonDetector(model, manifest)
    assert error.value.code == "person_asset_invalid"


def test_scene_has_no_camera_defaults_and_rejects_foreign_frame_hypotheses():
    source = info()
    assert normalize_scene(None, source)["unknown"] == list(perception.ZONE_NAMES)
    zones = [{"name": "counter", "bbox": [.1, .2, .9, .5], "source_indices": [2]},
             {"name": "storage", "bbox": [.2, .6, .4, .9], "source_indices": [999]}]
    scene = normalize_scene({"zones": zones}, source)
    assert [zone["name"] for zone in scene["zones"]] == ["counter"]
    assert scene["evidence"] == [{"source_index": 2, "timestamp_sec": .4}]
    assert "storage" in scene["unknown"]


def test_later_scene_assignment_uses_anonymous_adjacent_tracks_without_cross_camera_geometry():
    source = info()
    report = {"detections": [{"id": "a", "kind": "person", "track_id": "track-1", "bbox": [.1, .1, .3, .9],
                               "timestamp_sec": 0., "source_index": 0, "continuity": "new"},
                              {"id": "b", "kind": "person", "track_id": "track-1", "bbox": [.3, .1, .7, .9],
                               "timestamp_sec": .2, "source_index": 1, "continuity": "adjacent_detection"},
                              {"id": "c", "kind": "person", "track_id": "track-1", "bbox": [.3, .1, .7, .9],
                               "timestamp_sec": .4, "source_index": 2, "continuity": "new"}],
              "proposals": [], "motion": [{"timestamp_sec": .4, "scene_change_suspected": True}]}
    scene = {"zones": [{"zone": "counter", "status": "supported", "bbox": [.4, .1, .8, .9],
                        "refs": [{"source_index": 0, "timestamp_sec": 0.}]}]}
    mapped = perception.apply_scene(report, scene, source)
    assert report["proposals"] == []  # pure update
    assert [o["zones"] for o in mapped["track_observations"]] == [[], ["counter"], []]
    assert len(mapped["proposals"]) == 1
    assert mapped["proposals"][0]["status"] == "unconfirmed_trigger"
    assert mapped["proposals"][0]["source_indices"] == [0, 1]
    assert mapped["scene"]["invalidated_by_scene_change"]


class FakeCache:
    def __init__(self, frames):
        self.frames, self.calls = frames, []

    def get(self, targets, height=320):
        self.calls.append((targets, height))
        return self.frames


class SequenceDetector:
    name = "synthetic_detector"
    provenance = {"person_detection_available": True}
    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def detect(self, image):
        return next(self.sequence)


def synthetic_cache(tmp_path, colors):
    source = info(tuple(i * .2 for i in range(len(colors))), len(colors) * .2)
    frames = []
    for index, color in enumerate(colors):
        path = tmp_path / f"{index}.jpg"
        assert cv2.imwrite(str(path), np.full((192, 320, 3), color, np.uint8))
        frames.append(FrameRef(index, source.pts[index], path))
    return source, FakeCache(frames)


def test_analyzer_binds_pts_and_preserves_ending_with_gap_unknowns(tmp_path, monkeypatch):
    source, cache = synthetic_cache(tmp_path, [20, 20, 20, 20])
    box = {"bbox": [.1, .1, .4, .9], "confidence": .8}
    monkeypatch.setattr(perception, "build_detector", lambda config: SequenceDetector([[box], [], [box], [box]]))
    result = analyze_perception(source, cache, Budget(time.time() + 30), tmp_path / "out")
    assert len(cache.calls) == 1
    assert result["coverage"]["ending_included"] is True
    assert result["coverage"]["last_timestamp_sec"] == source.pts[-1]
    assert all(d["timestamp_sec"] == source.pts[d["source_index"]] for d in result["detections"])
    assert result["detections"][0]["track_id"] != result["detections"][1]["track_id"]
    assert all(d["role"] == "unknown" for d in result["detections"])
    gap = next(g for g in result["visibility_gaps"] if g["reason"] == "track_detection_gap")
    assert gap["refs"] == [{"source_index": 0, "timestamp_sec": 0.}, {"source_index": 1, "timestamp_sec": .2}]
    assert result["proposals"] == []
    assert (tmp_path / "out" / "perception.json").is_file()


def test_scene_cut_invalidates_geometry_and_motion_proposals_remain_unconfirmed(tmp_path, monkeypatch):
    source, cache = synthetic_cache(tmp_path, [0, 255, 255])
    monkeypatch.setattr(perception, "build_detector", lambda config: SequenceDetector([[], [], []]))
    scene = {"zones": [{"name": "counter", "bbox": [.1, .2, .9, .5], "source_indices": [0]}]}
    result = analyze_perception(source, cache, Budget(time.time() + 30), tmp_path / "out", scene=scene)
    assert result["scene"]["invalidated_by_scene_change"]
    assert result["scene"]["unknown"] == list(perception.ZONE_NAMES)
    assert result["proposals"][0]["reason"] == "scene_change"
    assert result["proposals"][0]["status"] == "unconfirmed_trigger"
    assert result["proposals"][0]["timecode_sec"] == .2
    assert result["proposals"][0]["source_indices"] == [0, 1]
    assert "verdict" not in result


def test_unreadable_frame_and_deadline_fail_without_completed_perception(tmp_path, monkeypatch):
    source = info((0.,), .1)
    cache = FakeCache([FrameRef(0, 0., tmp_path / "missing.jpg")])
    monkeypatch.setattr(perception, "build_detector", lambda config: SequenceDetector([[]]))
    for budget, expected in ((Budget(time.time() + 20), "perception_frame_invalid"),
                             (Budget(time.time() - 1), "deadline_exceeded")):
        with pytest.raises(AnalysisError) as error:
            analyze_perception(source, cache, budget, tmp_path / "out")
        assert error.value.code == expected
        assert not (tmp_path / "out" / "perception.json").exists()


def test_perception_stage_overrun_does_not_write_completed_report(tmp_path, monkeypatch):
    source, cache = synthetic_cache(tmp_path, [20])
    clock = [0.]
    class SlowDetector(SequenceDetector):
        def detect(self, image):
            clock[0] = 46.
            return []
    monkeypatch.setattr(perception, "build_detector", lambda config: SlowDetector([[]]))
    monkeypatch.setattr(perception.time, "monotonic", lambda: clock[0])
    with pytest.raises(AnalysisError) as error:
        analyze_perception(source, cache, Budget(time.time() + 300), tmp_path / "out")
    assert error.value.code == "perception_timeout"
    assert not (tmp_path / "out" / "perception.json").exists()
