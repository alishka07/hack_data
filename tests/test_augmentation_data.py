"""No filename, perceptual similarity or candidate category can invent labels."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from openpyxl import Workbook
import pytest

from worker.augmentation_data import (assign_supervision, augmentation_family, export_supervised_jsonl,
                                      load_reviewed_provenance, rank_source_candidates, read_explicit_workbook)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def source_records():
    sources, rows = [], []
    for index in range(24):
        path = str(Path(f"synthetic-{index}.mov").resolve())
        sources.append({"path": path, "sha256": digest(index), "decode_status": "sampled", "duration_sec_estimate": 20.0})
        rows.append({"row": index + 2, "video_path": path, "binary_label": index % 2,
                     "weak_anchors_sec": [4.0] if index % 2 else [], "issues": []})
    return sources, {"rows": rows}


def clip_record(name="cashier_01__flipped", family="cashier_01", seed="clip"):
    return {"path": str(Path(name + ".mp4").resolve()), "sha256": digest(seed), "family": family,
            "decode_status": "sampled", "duration_sec_estimate": 20.0, "source_candidates": []}


def test_authoritative_column_c_and_exact_case_matching_do_not_use_category_or_number(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "case-01.mov").write_bytes(b"synthetic")
    book = Workbook()
    sheet = book.active
    sheet.append(["номер кейса", "таймстемп", "нарушение / без нарушения", "возможные нарушения"])
    sheet.append(["case-01", "00:03", "без нарушения", "some_candidate"])
    sheet.append(["cashier_01", "00:04", "нарушение", "some_candidate"])
    sheet.append(["case-01", "00:05", None, "some_candidate"])
    path = tmp_path / "labels.xlsx"
    book.save(path)
    before = path.read_bytes()
    result = read_explicit_workbook(path, folder)
    assert path.read_bytes() == before
    assert result["rows"][0]["binary_label"] == 0
    assert result["rows"][0]["weak_anchors_sec"] == [3.0]
    assert result["rows"][1]["binary_label"] is None
    assert result["rows"][2]["binary_label"] is None


def test_formula_labels_are_quarantined_instead_of_using_cached_or_descriptive_values(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "one.mov").write_bytes(b"synthetic")
    book = Workbook()
    book.active.append(["file", "timestamp", "нарушение / без нарушения"])
    book.active.append(["one", None, "=1"])
    path = tmp_path / "labels.xlsx"
    book.save(path)
    row = read_explicit_workbook(path, folder)["rows"][0]
    assert row["binary_label"] is None and "missing_or_formula_input" in row["issues"]


def test_filename_families_group_variants_but_do_not_map_to_source_labels():
    sources, labels = source_records()
    clip = clip_record()
    clip["source_candidates"] = [{"source_sha256": sources[1]["sha256"], "strong_candidate": True}]
    result = assign_supervision(sources, [clip], labels, {})
    assert augmentation_family(Path(clip["path"])) == ("cashier_01", "flipped")
    assert clip["binary_label"] is None and clip["split"] == "quarantine"
    assert clip["verified_source_sha256"] is None
    assert Counter(group["split"] for group in result["source_groups"]) == {"train": 8, "validation": 8, "test": 8}


def test_exact_duplicate_sources_with_conflicting_labels_are_not_exportable():
    sources, labels = source_records()
    sources[1]["sha256"] = sources[0]["sha256"]
    result = assign_supervision(sources, [], labels, {})
    group = next(group for group in result["source_groups"] if group["video_sha256"] == sources[0]["sha256"])
    assert group["binary_label"] is None and group["split"] == "quarantine"


def test_reviewed_family_links_union_transitively_before_split_even_with_mixed_labels():
    sources, labels = source_records()
    clips = [clip_record(f"a__v{i}", "a" if i < 2 else "b", f"clip{i}") for i in range(4)]
    source_indices = [0, 1, 1, 2]
    reviewed = {clip["sha256"]: {"source_sha256": sources[index]["sha256"]} for clip, index in zip(clips, source_indices)}
    result = assign_supervision(sources, clips, labels, reviewed)
    linked = [group for group in result["source_groups"] if group["video_sha256"] in {sources[index]["sha256"] for index in source_indices}]
    assert len({group["leakage_group_id"] for group in linked}) == 1
    assert len({group["split"] for group in linked}) == 1
    assert linked[0]["split"] != "quarantine"
    assert all(clip["split"] == linked[0]["split"] for clip in clips)
    reversed_result = assign_supervision(list(reversed(sources)), deepcopy(list(reversed(clips))), labels, reviewed)
    assert {g["video_sha256"]: g["split"] for g in result["source_groups"]} == {g["video_sha256"]: g["split"] for g in reversed_result["source_groups"]}


def test_tiny_corpus_cannot_silently_become_all_training():
    sources, labels = source_records()
    result = assign_supervision(sources[:2], [], labels, {})
    assert all(group["split"] == "quarantine" for group in result["source_groups"])


def test_export_balances_source_weights_and_separates_clean_holdouts_from_robustness(tmp_path):
    sources, labels = source_records()
    initial = assign_supervision(sources, [], labels, {})
    training = next(group for group in initial["source_groups"] if group["split"] == "train")
    validation = next(group for group in initial["source_groups"] if group["split"] == "validation")
    clips = [clip_record("a__bright", "a", "first"), clip_record("a__dim", "a", "second"), clip_record("b__flip", "b", "third")]
    reviewed = {clips[0]["sha256"]: {"source_sha256": training["video_sha256"]},
                clips[1]["sha256"]: {"source_sha256": training["video_sha256"]},
                clips[2]["sha256"]: {"source_sha256": validation["video_sha256"]}}
    manifest = assign_supervision(sources, clips, labels, reviewed)
    counts = export_supervised_jsonl(manifest, tmp_path)
    assert counts == {"train": 10, "validation": 8, "test": 8}
    records = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    weights = Counter()
    for record in records:
        weights[record["source_group_id"]] += record["sample_weight"]
        assert record["supervision"] == "video_binary_only"
        assert "category" not in record and "assistant" not in record
    assert all(weight == pytest.approx(1) for weight in weights.values())
    robustness = [json.loads(line) for line in (tmp_path / "robustness-validation.jsonl").read_text().splitlines()]
    assert len(robustness) == 1 and robustness[0]["source_group_id"] == validation["video_sha256"]


def test_byte_identity_never_duplicates_or_overwrites_the_original_export(tmp_path):
    sources, labels = source_records()
    clip = clip_record()
    clip["sha256"] = sources[0]["sha256"]
    result = assign_supervision(sources, [clip], labels, {})
    assert clip["provenance_status"] == "byte_identity" and clip["binary_label"] == 0
    counts = export_supervised_jsonl(result, tmp_path)
    assert sum(counts.values()) == 24
    assert all(json.loads(line)["origin"] == "original" for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines())


def test_reviewed_provenance_requires_attestation_existing_hashes_and_full_timeline(tmp_path):
    sources, _ = source_records()
    clips = [clip_record()]
    link = {"clip_sha256": clips[0]["sha256"], "source_sha256": sources[0]["sha256"], "full_video": True,
            "label_preserving": True, "time_offset_sec": 0, "time_scale": 1, "horizontal_flip": True}
    review = {"schema_version": "forte-reviewed-augmentation-provenance/1", "human_review_attested": True,
              "reviewer": "Synthetic test reviewer", "reviewed_at": "2026-10-18T00:00:00+05:00", "links": [link]}
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(review))
    assert load_reviewed_provenance(path, clips, sources)[clips[0]["sha256"]]["source_sha256"] == sources[0]["sha256"]
    for mutation in ({"human_review_attested": False}, {"reviewed_at": "2026-10-18"}, {"reviewer": None}):
        path.write_text(json.dumps({**review, **mutation}))
        with pytest.raises(ValueError):
            load_reviewed_provenance(path, clips, sources)
    for mutation in ({"full_video": False}, {"time_scale": 0.5}, {"source_sha256": digest("foreign")}, {"label_preserving": False}):
        path.write_text(json.dumps({**review, "links": [{**link, **mutation}]}))
        with pytest.raises(ValueError):
            load_reviewed_provenance(path, clips, sources)


def test_perceptual_matching_handles_flip_and_never_promotes_second_ranked_source():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(5, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    altered = vectors * 0.985
    signature = {"normal": -vectors, "flipped": vectors, "hashes": [0] * 5, "flipped_hashes": [15] * 5}
    sources = [{"sha256": digest(0), "duration_sec_estimate": 20.0}, {"sha256": digest(1), "duration_sec_estimate": 20.1}]
    signatures = {digest(0): {"normal": vectors, "hashes": [15] * 5}, digest(1): {"normal": altered, "hashes": [15] * 5}}
    ranked = rank_source_candidates({"duration_sec_estimate": 20.0}, signature, sources, signatures)
    assert ranked[0]["source_sha256"] == digest(0) and ranked[0]["orientation"] == "flipped"
    assert ranked[0]["strong_candidate"] is False  # near-identical rival needs review
    assert ranked[1]["strong_candidate"] is False
    assert "binary_label" not in ranked[0]
    single = rank_source_candidates({"duration_sec_estimate": 20.0}, signature, sources[:1], signatures)
    assert single[0]["strong_candidate"] is True
