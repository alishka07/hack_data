"""Model contracts, source linkage and genuine ByteTrack recovery fixtures."""
from pathlib import Path
import io
import json
import time

import cv2
import numpy as np
import pytest

from worker import perception, perception_models as models
from worker.perception_models import ByteTrackAdapter, decode_simcc, normalize_owlv2_result, pose_affine
from worker.video import AnalysisError, Budget, FrameRef, VideoInfo


@pytest.fixture(autouse=True)
def isolate_installed_assets(monkeypatch):
    monkeypatch.setenv("FORTE_POSE_BACKEND", "disabled")
    monkeypatch.setenv("FORTE_OBJECT_BACKEND", "disabled")


def test_simcc_decode_uses_affine_source_coordinates_and_marks_unseen_wrists():
    image = np.zeros((512, 384, 3), dtype=np.uint8)
    tensor, center, scale = pose_affine(image, [.25, .25, .75, .75])
    assert tensor.shape == (1, 3, 256, 192)
    x, y = np.zeros((1, 17, 384), np.float32), np.zeros((1, 17, 512), np.float32)
    x[:, :, 192], y[:, :, 256] = .7, .8
    x[:, 9, :] = 0  # no left-wrist support
    result = decode_simcc([x, y], (192, 256), center, scale, (384, 512))
    assert result[10]["name"] == "right_wrist"
    assert result[10]["x"] == pytest.approx(.5)
    assert result[10]["y"] == pytest.approx(.5)
    assert result[10]["confidence"] == pytest.approx(.7)
    assert result[9]["status"] == "not_visible"
    assert result[9]["x"] is None and result[9]["y"] is None


@pytest.mark.parametrize("outputs", [[np.zeros((1, 26, 384)), np.zeros((1, 26, 512))],
                                    [np.zeros((1, 17, 384))],
                                    [np.full((1, 17, 384), np.nan), np.zeros((1, 17, 512))]])
def test_simcc_rejects_incompatible_or_nonfinite_model_outputs(outputs):
    with pytest.raises(AnalysisError) as error:
        decode_simcc(outputs, (192, 256), [192, 256], [192, 256], (384, 512))
    assert error.value.code == "pose_output_invalid"


def test_object_output_is_normalized_fixed_vocabulary_and_nms_preserves_different_classes():
    output = {"boxes": np.asarray([[40, 20, 120, 60], [42, 20, 120, 60], [40, 20, 120, 60]]),
              "scores": np.asarray([.8, .7, .6]), "labels": np.asarray([0, 0, 2])}
    objects = normalize_owlv2_result(output, (100, 200))
    assert [obj["kind"] for obj in objects] == ["cash", "phone"]
    assert objects[0]["bbox"] == pytest.approx([.2, .2, .6, .6])
    assert "verdict" not in str(objects)
    assert "violation" not in str(objects)
    output["labels"][0] = 999
    with pytest.raises(AnalysisError) as error:
        normalize_owlv2_result(output, (100, 200))
    assert error.value.code == "object_output_invalid"


def test_computer_query_localizes_hardware_without_asserting_display_or_lock_state():
    kinds = [kind for kind, _ in models.OBJECT_QUERIES]
    # Existing class indices remain stable when hardware support is extended.
    assert kinds[:7] == ["cash", "bag", "phone", "drawer", "safe", "counter", "window"]
    assert kinds.count("computer") == 1
    index = kinds.index("computer")
    assert models.OBJECT_QUERIES[index][1] == "a laptop computer or desktop computer monitor"
    output = {"boxes": np.asarray([[40, 20, 120, 60]]),
              "scores": np.asarray([.8]), "labels": np.asarray([index])}
    computer, = normalize_owlv2_result(output, (100, 200))
    assert computer["kind"] == computer["label"] == "computer"
    assert computer["bbox"] == pytest.approx([.2, .2, .6, .6])
    assert set(computer) == {"bbox", "confidence", "kind", "label", "query"}


def test_missing_optional_assets_report_unavailable_without_loading_or_download(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing assets must not load or download models")
    monkeypatch.setattr(models, "RTMPoseONNX", forbidden)
    monkeypatch.setattr(models, "OWLv2Local", forbidden)
    for kind in ("pose", "objects"):
        backend, capability = models.optional_backend(kind, "auto", tmp_path / "absent", .3)
        assert backend is None
        assert capability["status"] == "unavailable"
        assert capability["model_download_during_inference"] is False


def test_explicit_optional_backend_missing_assets_fails_instead_of_fallback(tmp_path):
    with pytest.raises(AnalysisError) as error:
        models.optional_backend("pose", "rtmpose", tmp_path / "absent.onnx", .3)
    assert error.value.code == "perception_asset_invalid"


def test_object_subprocess_caps_reads_and_timeout_updates_to_remaining_stage(monkeypatch):
    worker = object.__new__(models.OWLv2Subprocess)
    clock = [10.]
    monkeypatch.setattr(models.time, "monotonic", lambda: clock[0])
    class Messages:
        timeout = None
        def get(self, timeout):
            self.timeout = timeout
            return '{"status":"ok","detections":[]}'
    worker.messages = Messages()
    worker.deadline = 10.25
    assert worker._receive(90)["status"] == "ok"
    assert worker.messages.timeout == pytest.approx(.25)
    worker.set_timeout(.1)
    assert worker.deadline == pytest.approx(10.1)
    worker.set_timeout(30.)
    assert worker.deadline == pytest.approx(10.1)  # cannot extend accepted job budget
    clock[0] = 11.
    with pytest.raises(AnalysisError) as error:
        worker._receive(90)
    assert error.value.code == "object_inference_timeout"


def test_object_startup_has_deadline_and_offline_cpu_environment(monkeypatch):
    calls = {}
    class Process:
        stdin, stdout = io.StringIO(), io.StringIO()
        returncode = None
        def __init__(self, arguments, **kwargs):
            calls.update({"arguments": arguments, **kwargs})
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = 0
        def wait(self, timeout):
            return self.returncode
    class Thread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(models.subprocess, "Popen", Process)
    monkeypatch.setattr(models.threading, "Thread", Thread)
    monkeypatch.setattr(models.time, "monotonic", lambda: 10.)
    def receive(worker, timeout):
        assert worker.deadline == pytest.approx(10.05)
        return {"status": "ready", "provenance": {"device": "cpu"}}
    monkeypatch.setattr(models.OWLv2Subprocess, "_receive", receive)
    worker = models.OWLv2Subprocess(Path("local-model"), Path(models.sys.executable), timeout=.05)
    assert calls["env"]["HF_HUB_OFFLINE"] == "1"
    assert calls["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert calls["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls["env"]["FORTE_OBJECT_ISOLATED_WORKER"] == "1"
    assert calls["cwd"] == str(models.ROOT)
    worker.close()
    assert worker.process is None


def test_real_bytetrack_keeps_low_score_association_and_never_emits_prediction_as_visible():
    try:
        tracker = ByteTrackAdapter()
    except AnalysisError as exc:
        if exc.code == "tracking_backend_unavailable":
            pytest.skip("Optional real ByteTrack dependencies not installed")
        raise
    strong = {"bbox": [.1, .1, .4, .9], "confidence": .9}
    weak = {"bbox": [.1, .1, .4, .9], "confidence": .18}
    # Real library confirms a spawned track on its next successful observation.
    tracker.update([strong], FrameRef(0, 0., Path("unused")))
    second, _ = tracker.update([strong], FrameRef(1, .2, Path("unused")))
    recovered, _ = tracker.update([weak], FrameRef(2, .4, Path("unused")))
    assert second[0]["track_id"] == recovered[0]["track_id"]
    assert recovered[0]["confidence"] == .18
    assert recovered[0]["continuity"] == "adjacent_detection"
    missing, gaps = tracker.update([], FrameRef(3, .6, Path("unused")))
    assert missing == []
    assert gaps[0]["reason"] == "track_detection_gap"
    resumed, _ = tracker.update([strong], FrameRef(4, .8, Path("unused")))
    assert resumed[0]["track_id"] == recovered[0]["track_id"]
    assert resumed[0]["continuity"] == "after_unknown_gap"
    tracker.reset()
    new, _ = tracker.update([weak], FrameRef(5, 1., Path("unused")))
    assert new == []  # low confidence cannot spawn new people


def test_cadence_source_linkage_and_models_receive_pixels_only(tmp_path, monkeypatch):
    # An adversarial filename/rule label must never become model input text.
    source = VideoInfo(tmp_path / "violation_GT_phone.mov", .8, 320, 192, (0., .2, .4, .6), 5.)
    frames = []
    for index in range(4):
        path = tmp_path / f"GT_cash_{index}.jpg"
        cv2.imwrite(str(path), np.full((192, 320, 3), 50, np.uint8))
        frames.append(FrameRef(index, source.pts[index], path))
    class Cache:
        def get(self, targets, height):
            return frames
    class Detector:
        name = "fixture"
        provenance = {"person_detection_available": True}
        def detect(self, pixels):
            assert isinstance(pixels, np.ndarray)
            return [{"bbox": [.1, .1, .4, .9], "confidence": .8}]
    class Pose:
        name = "fixture_pose"
        closed = False
        def estimate(self, pixels, persons):
            assert isinstance(pixels, np.ndarray)
            assert set(persons[0]) <= {"id", "label", "kind", "confidence", "bbox", "timestamp_sec", "source_index",
                                     "track_id", "continuity", "backend", "role", "association_status", "refs"}
            assert "GT" not in json.dumps(persons)
            return [{"person_detection_id": person["id"], "track_id": person["track_id"], "keypoints": []} for person in persons]
        def close(self):
            self.closed = True
    class Objects:
        name = "fixture_objects"
        closed = False
        calls = 0
        def detect(self, pixels):
            assert isinstance(pixels, np.ndarray)
            self.calls += 1
            return [{"kind": "phone", "label": "phone", "confidence": .6, "bbox": [.2, .2, .3, .3]}]
        def close(self):
            self.closed = True
    pose, objects = Pose(), Objects()
    monkeypatch.setenv("FORTE_TRACKER", "adjacent_iou")
    monkeypatch.setattr(perception, "build_detector", lambda config: Detector())
    monkeypatch.setattr(perception, "optional_backend", lambda kind, *args: (pose if kind == "pose" else objects,
                                                                          {"status": "enabled", "enabled": True}))
    report = perception.analyze_perception(source, Cache(), Budget(time.time() + 30), tmp_path / "result")
    assert objects.calls == 1
    assert [p["source_index"] for p in report["poses"]] == [0, 3]
    assert all(p["refs"] == [{"source_index": p["source_index"], "timestamp_sec": source.pts[p["source_index"]]}]
               for p in report["poses"])
    assert len(report["frame_samples"]) == 4
    assert len(report["object_samples"]) == 1
    assert report["poses"][0]["source_index"] == report["object_samples"][0]["source_index"]
    assert pose.closed and objects.closed
    assert report["capabilities"]["scheduling"]["gpu_memory_allocated_by_perception"] is False
