"""Synthetic object localization contracts; no private media or inference."""
from copy import deepcopy
from pathlib import Path

import pytest

from worker.localization import (KINDS, collect_detections, deduplicate_detections,
                                 detection_schema, localization_prompt)
from worker.runtime import _validate_json_schema
from worker.video import AnalysisError, FrameRef


def frames():
    return [FrameRef(30, 1., Path("synthetic-1.jpg")),
            FrameRef(60, 2., Path("synthetic-2.jpg"))]


def detection(**fields):
    return {"frame_index": 0, "label": "Visible item", "kind": "document",
            "confidence": .8, "bbox": [.1, .2, .5, .8], **fields}


def collect(*items, source=None, window_id=3):
    return collect_detections({"detections": list(items)}, frames() if source is None else source, window_id)


def test_schema_bounds_objects_and_coordinates_and_prompt_keeps_annotations_separate():
    schema = detection_schema(2)
    assert schema["maxItems"] == 12
    assert schema["items"]["properties"]["frame_index"]["maximum"] == 1
    assert schema["items"]["properties"]["kind"]["enum"] == list(KINDS)
    _validate_json_schema([detection()], schema)
    for invalid in ([detection()] * 13, [detection(bbox=[0, 0, 1])],
                    [detection(frame_index=2)], [detection(bbox=[0, 0, 1, 1.1])]):
        with pytest.raises(AnalysisError):
            _validate_json_schema(invalid, schema)
    for count in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            detection_schema(count)
    prompt = localization_prompt()
    assert all(kind in prompt for kind in KINDS)
    assert "one best-visible supplied frame per object" in prompt
    assert "do not prove a breach" in prompt
    assert "without identities" in prompt
    assert len(prompt) < 1000


def test_collection_binds_only_supplied_source_metadata_without_mutating_input():
    answer = {"detections": [detection(frame_index=1, label=" Document ", bbox=[0, 0, 1, 1])]}
    previous = deepcopy(answer)
    result = collect_detections(answer, frames(), 7)
    assert answer == previous
    assert result == [{"id": "detection-7-0", "label": "Document", "kind": "document",
                       "confidence": .8, "timestamp_sec": 2., "source_index": 60,
                       "bbox": [0., 0., 1., 1.]}]
    result[0]["bbox"][0] = .2
    assert answer == previous


def test_missing_and_empty_detections_are_backward_compatible_and_zero_is_not_a_detection():
    assert collect_detections({}, frames(), 0) == []
    assert collect() == []
    assert collect(detection(confidence=0)) == []
    result = collect(detection(confidence=0), detection(frame_index=1))
    assert result[0]["id"] == "detection-3-1"
    with pytest.raises(AnalysisError):
        collect(detection(confidence=0, bbox=[0, 0, 0, 0]))


@pytest.mark.parametrize("box", [None, "0,0,1,1", [], [0, 0, 1], [0, 0, 1, 1, 1],
                                 [0, 0, 0, 1], [0, 0, 1, 0], [.8, .2, .1, .9],
                                 [0, -.1, 1, 1], [0, 0, 1.1, 1], [False, 0, 1, 1],
                                 [0, 0, float("nan"), 1], [0, 0, 1, float("inf")]])
def test_invalid_boxes_fail_at_collection_boundary(box):
    with pytest.raises(AnalysisError) as error:
        collect(detection(bbox=box))
    assert error.value.code == "model_schema_invalid"


@pytest.mark.parametrize("fields", [{"frame_index": -1}, {"frame_index": 2}, {"frame_index": True},
                                    {"frame_index": 0.}, {"label": " "}, {"label": "x" * 61},
                                    {"kind": "unknown_kind"}, {"confidence": True}, {"confidence": -.1},
                                    {"confidence": 1.1}, {"confidence": float("nan")},
                                    {"confidence": float("inf")}, {"source_index": 999}])
def test_invalid_fields_and_foreign_source_references_are_rejected(fields):
    with pytest.raises(AnalysisError) as error:
        collect(detection(**fields))
    assert error.value.code == "model_schema_invalid"


@pytest.mark.parametrize("source", [[], [object()], [FrameRef(-1, 1., Path("unused.jpg"))],
                                   [FrameRef(True, 1., Path("unused.jpg"))],
                                   [FrameRef(1, float("nan"), Path("unused.jpg"))],
                                   [FrameRef(1, -1., Path("unused.jpg"))],
                                   [FrameRef(1, 1., Path("unused.jpg")), FrameRef(1, 2., Path("unused.jpg"))]])
def test_invalid_host_source_metadata_cannot_become_annotation(source):
    with pytest.raises(AnalysisError):
        collect(detection(), source=source)


def test_collection_rejects_malformed_container_missing_fields_and_window_ids():
    for answer in (None, [], {"detections": None}, {"detections": {}},
                   {"detections": [None]}, {"detections": [detection()] * 13},
                   {"detections": [{key: value for key, value in detection().items() if key != "bbox"}]}):
        with pytest.raises(AnalysisError):
            collect_detections(answer, frames(), 0)
    for window_id in (-1, True, 1.5):
        with pytest.raises(AnalysisError):
            collect(detection(), window_id=window_id)


def test_dedupe_keeps_strongest_overlapping_duplicate_without_mutation():
    supplied = collect(detection(confidence=.5), detection(confidence=.9, bbox=[.11, .2, .51, .8]))
    previous = deepcopy(supplied)
    result = deduplicate_detections(supplied)
    assert result == [supplied[1]]
    assert supplied == previous
    result[0]["bbox"][0] = .3
    assert supplied == previous


def test_dedupe_never_merges_other_source_frames_kinds_or_distinct_nearby_objects():
    supplied = collect(detection(), detection(frame_index=1), detection(kind="cash"),
                       detection(bbox=[.4, .2, .8, .8]))
    assert deduplicate_detections(supplied) == supplied
    # Equal presentation times do not make different source indices identical.
    supplied[1]["timestamp_sec"] = supplied[0]["timestamp_sec"]
    assert deduplicate_detections(supplied) == supplied


def test_dedupe_requires_exact_source_timestamp_and_keeps_stable_input_order():
    supplied = collect(detection(confidence=.2), detection(frame_index=1, confidence=.9))
    assert deduplicate_detections(supplied) == supplied
    repeated = deepcopy(supplied[0])
    repeated["timestamp_sec"] += 1
    assert deduplicate_detections([supplied[0], repeated]) == [supplied[0], repeated]
    assert deduplicate_detections([]) == []
