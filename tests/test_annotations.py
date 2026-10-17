"""Synthetic reviews exercise provenance, honest unknowns and metric denominators."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.review_annotations import main
from worker.annotations import (AnnotationError, annotation_template, import_annotation,
                                load_annotation, local_file, proposal_metrics,
                                split_group_ids, validate_annotation, write_exclusive_json)
from worker.video import VideoInfo


def source(tmp_path, name="synthetic"):
    path = tmp_path / (name + ".mov")
    path.write_bytes(name.encode())
    info = VideoInfo(path, 4., 640, 480, (0., .5, 1., 1.5, 2., 2.5, 3., 3.5), 42.)
    digest = hashlib.sha256(name.encode()).hexdigest()
    return info, digest


def reviewed(tmp_path, name="synthetic", *, positive=True):
    info, digest = source(tmp_path, name)
    annotation = annotation_template(info, digest)
    annotation["groups"].update(camera_id="camera_a", session_id=name)
    annotation["review"].update(status="reviewed", reviewer="Synthetic reviewer",
                                reviewed_at="2026-10-17T10:00:00+05:00", human_review_attested=True,
                                coverage="complete", coverage_intervals=[{"start_sec": 0., "end_sec": 4.}],
                                video_verdict="violation" if positive else "no_violation")
    if positive:
        annotation["events"].append({
            "event_id": "event_1", "type": "synthetic_action", "start_sec": 1., "end_sec": 2.,
            "status": "supported", "visible_facts": [{"name": "movement", "state": "supported",
                "description": "Synthetic movement", "frame_refs": [{"source_index": 3, "timestamp_sec": 1.5}]}],
            "prerequisites": [{"name": "context", "state": "unknown", "description": "Synthetic context unknown",
                               "frame_refs": []}], "exceptions": [], "uncertainty": ["Synthetic uncertainty"],
            "evidence_refs": [{"source_index": 3, "timestamp_sec": 1.5, "relevance": "unknown"}]})
    return info, digest, annotation


def test_template_has_no_fabricated_label_or_review(tmp_path):
    info, digest = source(tmp_path)
    annotation = annotation_template(info, digest)
    assert annotation["events"] == []
    assert annotation["review"]["video_verdict"] == "unknown"
    assert annotation["review"]["human_review_attested"] is False
    assert validate_annotation(annotation, info, digest)["reviewed"] is False
    with pytest.raises(AnnotationError, match="human-reviewed"):
        validate_annotation(annotation, info, digest, require_reviewed=True)


def test_reviewed_import_is_idempotent_never_replaces_other_annotations(tmp_path):
    info, digest, annotation = reviewed(tmp_path)
    destination = import_annotation(annotation, info, digest, tmp_path / "imports",
                                    attesting_reviewer="Synthetic reviewer")
    assert destination.name == digest + ".json"
    before = destination.read_bytes()
    assert import_annotation(annotation, info, digest, destination.parent,
                             attesting_reviewer="Synthetic reviewer") == destination
    changed = deepcopy(annotation)
    changed["events"][0]["uncertainty"].append("Additional uncertainty")
    with pytest.raises(AnnotationError, match="nothing was replaced"):
        import_annotation(changed, info, digest, destination.parent, attesting_reviewer="Synthetic reviewer")
    assert destination.read_bytes() == before
    assert not list(destination.parent.glob("*.tmp"))


def test_import_requires_explicit_matching_reviewer(tmp_path):
    info, digest, annotation = reviewed(tmp_path)
    with pytest.raises(AnnotationError, match="matching reviewer's"):
        import_annotation(annotation, info, digest, tmp_path / "imports", attesting_reviewer="Another reviewer")
    assert not (tmp_path / "imports").exists()


@pytest.mark.parametrize("field,value", [("source_index", 4), ("timestamp_sec", 1.5001),
                                         ("source_index", True), ("timestamp_sec", float("nan"))])
def test_evidence_must_bind_exact_source_index_and_pts(tmp_path, field, value):
    info, digest, annotation = reviewed(tmp_path)
    annotation["events"][0]["evidence_refs"][0][field] = value
    with pytest.raises(AnnotationError, match="frame|timestamp"):
        validate_annotation(annotation, info, digest)


def test_facts_and_source_identity_are_also_checked(tmp_path):
    info, digest, annotation = reviewed(tmp_path)
    with pytest.raises(AnnotationError, match="identity"):
        validate_annotation(annotation, info, "0" * 64)
    annotation["events"][0]["visible_facts"][0]["frame_refs"][0]["timestamp_sec"] = 2.
    with pytest.raises(AnnotationError, match="source PTS"):
        validate_annotation(annotation, info, digest)


@pytest.mark.parametrize("change", [
    lambda a: a["review"].update(human_review_attested=False),
    lambda a: a["review"].update(reviewed_at="2026-10-17T10:00:00"),
    lambda a: a["groups"].update(session_id=None),
    lambda a: a["review"].update(reviewer="=SUM(1)"),
    lambda a: a["review"].update(coverage_intervals=[{"start_sec": 0., "end_sec": 2.}]),
    lambda a: a["events"][0].update(evidence_refs=[]),
    lambda a: a["events"][0]["visible_facts"][0].update(frame_refs=[]),
    lambda a: a["events"][0].update(type="unknown"),
])
def test_reviewed_metadata_and_supported_evidence_cannot_be_implicit(tmp_path, change):
    info, digest, annotation = reviewed(tmp_path)
    change(annotation)
    with pytest.raises(AnnotationError):
        validate_annotation(annotation, info, digest, require_reviewed=True)


def test_negative_verdict_requires_complete_resolved_review(tmp_path):
    info, digest, annotation = reviewed(tmp_path, positive=False)
    assert validate_annotation(annotation, info, digest)["supported_events"] == 0
    annotation["review"]["unknowns"] = ["Visibility gap"]
    with pytest.raises(AnnotationError, match="complete, resolved"):
        validate_annotation(annotation, info, digest)


def test_adjacent_review_intervals_cover_an_event_and_catalog_is_explicit(tmp_path):
    info, digest, annotation = reviewed(tmp_path)
    annotation["review"]["coverage_intervals"] = [{"start_sec": 0., "end_sec": 1.5},
                                                 {"start_sec": 1.5, "end_sec": 4.}]
    assert validate_annotation(annotation, info, digest, allowed_types={"synthetic_action"})
    with pytest.raises(AnnotationError, match="catalog"):
        validate_annotation(annotation, info, digest, allowed_types={"other_action"})


@pytest.mark.parametrize("payload", ['{"schema_version":1,"schema_version":2}', '{"value":NaN}',
                                    '[1]', '{"value":Infinity}'])
def test_unsafe_or_ambiguous_json_is_rejected(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(AnnotationError):
        load_annotation(path)


@pytest.mark.parametrize("path", ["https://example.invalid/video.mov", "file:///C:/video.mov", "//server/video.mov",
                                 "\\\\server\\video.mov"])
def test_source_reader_rejects_network_locations_without_fetching(path):
    with pytest.raises(AnnotationError, match="local file"):
        local_file(path)


def test_group_components_join_transitive_sessions_related_clips_and_duplicates(tmp_path):
    annotations = [reviewed(tmp_path, name)[2] for name in ("a", "b", "c", "d")]
    annotations[1]["groups"]["session_id"] = "a"
    annotations[1]["groups"]["related_group_id"] = "related"
    annotations[2]["groups"]["related_group_id"] = "related"
    mapping = split_group_ids(annotations + [deepcopy(annotations[0])])
    assert len(mapping) == 4 and len(set(mapping.values())) == 2
    assert mapping == split_group_ids(list(reversed(annotations)))


def test_metrics_keep_unknown_failures_and_human_relevance_separate(tmp_path):
    _, pos, positive = reviewed(tmp_path, "positive")
    _, neg, negative = reviewed(tmp_path, "negative", positive=False)
    _, failed_hash, failed = reviewed(tmp_path, "failed")
    _, unknown_hash, unknown = reviewed(tmp_path, "unknown", positive=False)
    unknown["review"]["video_verdict"] = "unknown"
    result = proposal_metrics([positive, negative, failed, unknown], [
        {"video_sha256": pos, "status": "done", "verdict": "violation", "review_required": True,
         "proposals": [{"start_sec": .5, "end_sec": 2.5}], "latency_sec": 12.,
         "violations": [{"type": "synthetic_action", "timecode_sec": 1.5,
                         "evidence_frames": [{"timestamp_sec": 1.5}, {"timestamp_sec": 3.}]}]},
        {"video_sha256": neg, "status": "done", "verdict": "no_violation"},
        {"video_sha256": failed_hash, "status": "failed"},
        {"video_sha256": unknown_hash, "status": "done", "verdict": "violation"},
    ])
    assert result["counts"]["tp"] == result["counts"]["tn"] == 1
    assert result["counts"]["fn"] == result["counts"]["fp"] == 0
    assert result["video_confusion_denominator"] == 2
    assert result["video_precision"] == result["video_recall"] == result["video_specificity"] == 1.
    assert result["proposal_recall"] == result["event_localization_recall"] == .5
    assert result["evidence_temporal_relevance"] == .5
    assert result["human_evidence_relevance"] is None
    assert result["failure_rate"] == result["review_rate"] == .25
    assert result["latency_measured_sources"] == 1
    assert result["max_peak_gpu_memory_mib"] is None


def test_human_relevance_only_uses_exact_explicit_labels(tmp_path):
    _, digest, annotation = reviewed(tmp_path)
    annotation["events"][0]["evidence_refs"][0]["relevance"] = "irrelevant"
    prediction = {"video_sha256": digest, "status": "done", "verdict": "violation",
                  "violations": [{"type": "synthetic_action", "timecode_sec": 1.5,
                                  "evidence_frames": [{"timestamp_sec": 1.5}, {"timestamp_sec": 1.6}]}]}
    result = proposal_metrics([annotation], [prediction])
    assert result["evidence_temporal_relevance"] == 1.
    assert result["human_evidence_relevance"] == 0.
    assert result["counts"]["human_labeled_frames"] == 1


def test_one_finding_does_not_match_multiple_annotated_events(tmp_path):
    _, digest, annotation = reviewed(tmp_path)
    second = deepcopy(annotation["events"][0])
    second["event_id"] = "event_2"
    annotation["events"].append(second)
    result = proposal_metrics([annotation], [{"video_sha256": digest, "status": "done", "verdict": "violation",
        "violations": [{"type": "synthetic_action", "timecode_sec": 1.5}]}])
    assert result["event_localization_recall"] == .5
    assert result["event_localization_precision"] == 1.


def test_missing_predictions_and_unreviewed_inputs_do_not_become_clean_results(tmp_path):
    info, digest, annotation = reviewed(tmp_path)
    result = proposal_metrics([annotation], [])
    assert result["missing_predictions"] == 1
    assert result["video_recall"] is None
    assert result["proposal_recall"] is None
    with pytest.raises(AnnotationError, match="human-reviewed"):
        proposal_metrics([annotation_template(info, digest)], [])
    with pytest.raises(AnnotationError, match="exactly one"):
        proposal_metrics([annotation, annotation], [])


def test_unlisted_categories_and_unreviewed_intervals_are_not_event_negatives(tmp_path):
    _, digest, annotation = reviewed(tmp_path)
    prediction = {"video_sha256": digest, "status": "done", "verdict": "violation", "violations": [
        {"type": "synthetic_action", "timecode_sec": 1.5},
        {"type": "unlisted_action", "timecode_sec": 1.5},
        {"type": "synthetic_action", "timecode_sec": 3.5}]}
    result = proposal_metrics([annotation], [prediction], tolerance_sec=0)
    assert result["counts"]["emitted_events"] == 3
    assert result["counts"]["assessable_emitted_events"] == 1
    assert result["counts"]["unassessed_emitted_events"] == 2
    assert result["event_localization_precision"] == 1.


def test_point_annotation_proposal_recall_uses_containment(tmp_path):
    _, digest, annotation = reviewed(tmp_path)
    annotation["events"][0].update(start_sec=1.5, end_sec=1.5)
    result = proposal_metrics([annotation], [{"video_sha256": digest, "status": "done", "verdict": "no_violation",
                                              "proposals": [{"start_sec": 1., "end_sec": 2.}]}])
    assert result["proposal_recall"] == 1.


def test_cli_init_and_validate_do_not_print_private_paths_or_scene_text(tmp_path, monkeypatch, capsys):
    import scripts.review_annotations as cli
    info, digest = source(tmp_path, "PRIVATE_SOURCE")
    monkeypatch.setattr(cli, "inspect_source", lambda path: (info, digest))
    output = tmp_path / "draft.json"
    assert main(["init", "--video", str(info.path), "--output", str(output)]) == 0
    assert output.with_suffix(".pts.json").is_file()
    assert main(["validate", "--video", str(info.path), "--annotation", str(output)]) == 0
    assert main(["import", "--video", str(info.path), "--annotation", str(output),
                 "--attest-reviewed", "someone", "--directory", str(tmp_path / "imports")]) == 2
    captured = capsys.readouterr()
    assert "PRIVATE_SOURCE" not in captured.out + captured.err
    assert "draft_created" in captured.out and '"status": "rejected"' in captured.err


def test_atomic_json_refuses_unrelated_file_and_removes_temporary(tmp_path):
    target = tmp_path / "existing.json"
    target.write_text("Unrelated content")
    with pytest.raises(AnnotationError, match="nothing was replaced"):
        write_exclusive_json(target, {"a": 1})
    assert target.read_text() == "Unrelated content"
    assert not list(tmp_path.glob("*.tmp"))
