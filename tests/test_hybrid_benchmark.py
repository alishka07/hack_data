"""The local benchmark never sends reference labels or guesses missing metrics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.check_hybrid_local import CheckError, loopback_url, select_samples, summary, validate_details
from worker.video import VideoInfo


@pytest.mark.parametrize("value", ["https://127.0.0.1", "http://example.com", "http://127.0.0.1/private",
                                 "http://user:secret@127.0.0.1", "http://127.0.0.1?secret=x"])
def test_remote_or_credential_urls_are_rejected(value):
    with pytest.raises(CheckError):
        loopback_url(value)


def test_selection_is_duration_based_and_balanced_by_unique_source(tmp_path):
    rows = []
    durations = {}
    for i, (duration, binary) in enumerate(((15, False), (4, True), (9, False), (30, True))):
        path = tmp_path / f"synthetic-{i}.mov"
        path.write_bytes(b"synthetic")
        rows.append({"sha256": f"{i:064x}", "binary": binary, "video_path": str(path)})
        durations[path] = duration
    selection = {"longest_sample": rows[-1], "validation_samples": rows}
    def inspect(path, budget):
        return VideoInfo(path, durations[path], 64, 64, (0., 1.), 0.)
    assert select_samples(selection, "shortest", inspect)[0][1].duration == 4
    assert select_samples(selection, "longest", inspect)[0][1].duration == 30
    assert [info.duration for row, info in select_samples(selection, "balanced", inspect)] == [9, 4]


def test_summary_does_not_turn_failure_or_missing_review_into_clean_prediction():
    result = summary([
        {"status": "done", "reference_binary": False, "verdict": "violation", "finding_count": 2,
         "duration_sec": 120., "upload_to_result_sec": 30., "checks_passed": True, "under_300_sec": True},
        {"status": "failed", "reference_binary": True, "duration_sec": 30., "upload_to_result_sec": 300.,
         "checks_passed": True, "under_300_sec": False},
    ])
    assert result["failure_rate"] == .5
    assert result["review_rate_completed_known"] is None
    assert result["exploratory_binary"]["fp"] == 1
    assert result["exploratory_binary"]["fn"] == 0
    assert result["exploratory_binary"]["denominator"] == 1
    assert result["exploratory_binary"]["emitted_findings_per_negative_hour"] == 60
    assert result["all_under_300_sec"] is False


def details_fixture():
    return {"details": {"versions": {"pipeline": "3.0"},
        "coverage": {"windows_completed": 1, "windows_planned": 1,
            "perception": {"complete": True, "ending_included": True, "first_timestamp_sec": 0.,
                           "last_timestamp_sec": 1., "frames_analyzed": 2}},
        "evidence": [{"filename": "source.jpg", "timestamp_sec": 1.}],
        "detections": [{"source_index": 1, "timestamp_sec": 1., "bbox": [.1, .2, .7, .9],
                        "evidence_frame": {"filename": "source.jpg", "timestamp_sec": 1.}}],
        "review_queue": [], "case_decision": {"status": "review_required"}},
        "timings": {"vlm_sec": 3., "decode_and_cache_sec": 2.}}


def fake_client():
    _, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
    class Client:
        def get(self, url, headers=None):
            return SimpleNamespace(status_code=200 if headers else 401, content=encoded.tobytes())
    return Client()


def test_declared_jpeg_auth_boxes_pts_and_coverage_are_checked():
    info = VideoInfo(Path("synthetic.mov"), 2., 16, 16, (0., 1.), 0.)
    result = validate_details(fake_client(), {"X-API-Key": "synthetic"}, "synthetic", details_fixture(), info, "3.0", 3)
    assert result["detection_count"] == result["declared_jpegs_validated"] == 1
    assert result["evidence_authentication_checked"] is True
    assert result["review_required"] is True
    assert result["stage_targets"]["perception_sec"]["within_target"] is None


@pytest.mark.parametrize("change", [
    lambda d: d["details"]["versions"].update(pipeline="2.4"),
    lambda d: d["details"]["coverage"].update(windows_completed=0),
    lambda d: d["details"]["coverage"]["perception"].update(ending_included=False),
    lambda d: d["details"]["detections"][0].update(timestamp_sec=.5),
    lambda d: d["details"]["detections"][0].update(bbox=[.1, .2, 1.1, .9]),
    lambda d: d["details"]["detections"][0]["evidence_frame"].update(filename="undeclared.jpg"),
])
def test_invalid_details_cannot_pass_technical_contract(change):
    fixture = details_fixture()
    change(fixture)
    info = VideoInfo(Path("synthetic.mov"), 2., 16, 16, (0., 1.), 0.)
    with pytest.raises(CheckError):
        validate_details(fake_client(), {"X-API-Key": "synthetic"}, "synthetic", fixture, info, "3.0", 3)
