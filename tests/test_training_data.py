"""Synthetic training manifests cannot turn candidate categories into positives."""
from collections import Counter
from copy import deepcopy
import hashlib

from openpyxl import Workbook
import pytest

from worker.training_data import deterministic_split, read_simplified_labels


HEADERS = ["номер кейса", "таймстемп", "нарушение / без нарушения", "возможные нарушения"]


def source(tmp_path, rows, videos):
    folder = tmp_path / "videos"
    folder.mkdir()
    for stem, content in videos.items():
        (folder / (stem + ".mov")).write_bytes(content)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(HEADERS)
    for row in rows:
        sheet.append(row)
    path = tmp_path / "synthetic-labels.xlsx"
    workbook.save(path)
    workbook.close()
    return path, folder


def test_exact_stems_hash_duplicates_and_weak_anchors_preserve_the_source(tmp_path):
    labels, videos = source(tmp_path, [
        ["case-a", "00:02", "нарушение", "synthetic_event"],
        ["case-copy", "00:04", "нарушение", "synthetic_event"],
        ["case-b", None, "без нарушения", None],
    ], {"case-a": b"same video", "case-copy": b"same video", "case-b": b"negative video"})
    before = labels.read_bytes()
    result = read_simplified_labels(labels, videos)
    assert labels.read_bytes() == before
    assert result["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert result["counts"]["unique_video_groups"] == 2 and result["counts"]["duplicate_hash_groups"] == 1
    positive = next(group for group in result["groups"] if group["binary_label"])
    assert positive["anchors_sec"] == [2., 4.] and positive["source_rows"] == [2, 3]
    assert len(positive["video_paths"]) == 2 and positive["categories"] == ["synthetic_event"]
    assert positive["category_targets"] == {"synthetic_event": 1}


def test_negative_possible_categories_and_timestamps_never_create_positive_labels(tmp_path):
    labels, videos = source(tmp_path, [
        ["negative", "Candidate considered at 10 sec", "без нарушения", "synthetic_control"],
        ["positive-unknown", None, "нарушение", None],
    ], {"negative": b"negative", "positive-unknown": b"positive"})
    result = read_simplified_labels(labels, videos)
    negative = next(group for group in result["groups"] if group["binary_label"] == 0)
    positive = next(group for group in result["groups"] if group["binary_label"] == 1)
    assert negative["categories"] == [] and negative["observed_categories"] == ["synthetic_control"]
    assert negative["category_targets"] == {"synthetic_control": 0} and negative["category_mask"] is True
    assert negative["anchors_sec"] == [10.]
    assert positive["category_mask"] is False and positive["category_targets"] == {}
    assert positive["anchors_sec"] == []


def test_duplicate_video_binary_conflict_fails_instead_of_choosing_a_row(tmp_path):
    labels, videos = source(tmp_path, [["one", None, "нарушение", None], ["two", None, "без нарушения", None]],
                            {"one": b"identical", "two": b"identical"})
    with pytest.raises(ValueError, match="conflicting binary"):
        read_simplified_labels(labels, videos)


@pytest.mark.parametrize("case", ["case-1", "Case-01", "1", "case-01.mov"])
def test_case_matching_never_uses_fuzzy_numbers_casefolding_or_filename_fallback(tmp_path, case):
    labels, videos = source(tmp_path, [[case, None, "без нарушения", None]], {"case-01": b"video"})
    with pytest.raises(ValueError, match="exact MOV stem"):
        read_simplified_labels(labels, videos)


@pytest.mark.parametrize("timestamp", ["not a timestamp", "00:99", "-1", -1,
                                      pytest.param("9" * 400, id="nonfinite_numeric_text")])
def test_invalid_timestamps_fail_without_exposing_private_annotation_text(tmp_path, timestamp):
    labels, videos = source(tmp_path, [["case", timestamp, "нарушение", None]], {"case": b"video"})
    with pytest.raises(ValueError, match="invalid weak timestamp"):
        read_simplified_labels(labels, videos)


@pytest.mark.parametrize("column", range(4))
def test_formulas_in_any_label_column_are_rejected(tmp_path, column):
    row = ["case", None, "без нарушения", None]
    row[column] = "=1+1"
    labels, videos = source(tmp_path, [row], {"case": b"video"})
    with pytest.raises(ValueError, match="Formula cells"):
        read_simplified_labels(labels, videos)


@pytest.mark.parametrize("binary", [None, "unknown", 2])
def test_binary_labels_must_be_explicit_and_never_inferred_from_timestamps(tmp_path, binary):
    labels, videos = source(tmp_path, [["case", "00:01", binary, None]], {"case": b"video"})
    with pytest.raises(ValueError, match="explicit binary"):
        read_simplified_labels(labels, videos)


def test_multiple_sparse_category_annotations_are_combined_within_the_same_video_group(tmp_path):
    labels, videos = source(tmp_path, [["one", "00:01", "нарушение", "synthetic_first; synthetic_second"],
                                       ["two", "00:02", "нарушение", None]],
                            {"one": b"identical", "two": b"identical"})
    group = read_simplified_labels(labels, videos)["groups"][0]
    assert group["categories"] == ["synthetic_first", "synthetic_second"]
    assert group["category_targets"] == {"synthetic_first": 1, "synthetic_second": 1}


def test_empty_labels_fail_instead_of_returning_a_trainable_empty_manifest(tmp_path):
    labels, videos = source(tmp_path, [], {"unlabeled": b"video"})
    with pytest.raises(ValueError, match="no labeled videos"):
        read_simplified_labels(labels, videos)


def groups(positive=21, negative=19):
    return [{"video_sha256": hashlib.sha256(f"{label}-{index}".encode()).hexdigest(),
             "binary_label": label, "source_rows": [index + 2], "video_paths": [f"private-{label}-{index}.mov"]}
            for label, count in ((1, positive), (0, negative)) for index in range(count)]


def test_stratified_split_is_deterministic_disjoint_complete_and_input_order_independent():
    original = groups()
    before = deepcopy(original)
    result = deterministic_split(original)
    assert result == deterministic_split(list(reversed(original)))
    assert original == before
    assert {name: Counter(group["binary_label"] for group in chosen) for name, chosen in result.items()} == {
        "train": {1: 13, 0: 11}, "validation": {1: 4, 0: 4}, "test": {1: 4, 0: 4}}
    sets = [{group["video_sha256"] for group in chosen} for chosen in result.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set.union(*sets)) == len(original)
    result["train"][0]["source_rows"].append(999)
    assert original == before
    assert deterministic_split(original, seed=18) != deterministic_split(original, seed=17)


@pytest.mark.parametrize("data", [groups(positive=8), groups(negative=8), [], groups() + [groups()[0]]])
def test_missing_binary_class_support_or_duplicate_hash_records_cannot_form_holdouts(data):
    with pytest.raises(ValueError):
        deterministic_split(data)
