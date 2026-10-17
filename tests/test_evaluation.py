"""Synthetic hash/label evaluation checks; no private footage or descriptions."""
import json
from pathlib import Path
import sqlite3

from openpyxl import Workbook

from scripts.evaluate_local import evaluate, match_points, point_times, prepare_validation, read_reference


def test_hash_dedup_blank_negative_latest_done_and_timestamp_disagreement(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    for name, content in {"negative.mov": b"synthetic negative", "negative-copy.mov": b"synthetic negative",
                          "positive.mov": b"synthetic positive", "positive-copy.mov": b"synthetic positive"}.items():
        (videos / name).write_bytes(content)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "labels"
    for row in [("file", "time"), ("negative.mov", None), ("negative-copy.mov", None),
                ("positive.mov", "00:10"), ("positive-copy.mov", "00:20")]:
        sheet.append(row)
    labels = tmp_path / "labels.xlsx"
    workbook.save(labels)
    database = tmp_path / "jobs.sqlite"
    # Upload names intentionally differ from workbook names; matching must use bytes.
    upload_negative = tmp_path / "arbitrary-upload-1.mov"
    upload_positive = tmp_path / "arbitrary-upload-2.mov"
    upload_negative.write_bytes(b"synthetic negative")
    upload_positive.write_bytes(b"synthetic positive")
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE jobs(id TEXT,status TEXT,video_path TEXT,created_at REAL,result_json TEXT)")
        clean = json.dumps({"verdict": "no_violation", "violations": []})
        flagged = json.dumps({"verdict": "violation", "violations": [{"timecode_sec": 10}]})
        connection.executemany("INSERT INTO jobs VALUES(?,?,?,?,?)", [
            ("negative-old", "done", str(upload_negative), 1, flagged),
            ("negative-new", "done", str(upload_negative), 2, clean),
            ("negative-running", "processing", str(upload_negative), 3, None),
            ("positive", "done", str(upload_positive), 1, flagged),
        ])
    report = evaluate(labels, videos, database)
    assert report["reference"]["matched_unique_hash_groups"] == 2
    assert report["reference"]["duplicate_hash_groups"] == 2
    assert report["reference"]["disagreeing_timestamp_groups"] == 1
    assert report["coverage"]["completed_groups"] == 2
    assert report["exploratory_binary"]["confusion"] == {"true_positive": 1, "false_positive": 0, "true_negative": 1, "false_negative": 0}
    assert report["exploratory_localization"]["completed_groups_excluded_for_disagreement"] == 1
    assert report["exploratory_localization"]["reference_points"] == 0
    # Aggregate report must never contain source names, job IDs or generated descriptions.
    serialized = json.dumps(report)
    assert "negative.mov" not in serialized and "negative-new" not in serialized


def test_point_matching_is_one_to_one_and_blank_means_negative():
    assert point_times(None) == point_times(" ") == ()
    assert point_times("00:10; 01:05") == (10, 65)
    assert match_points((10, 11), (10.5,), 3) == (0.5,)
    assert len(match_points((10, 20), (11, 19, 40), 3)) == 2


def make_reference_and_jobs(tmp_path, rows, contents=None):
    videos = tmp_path / "videos"
    videos.mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "labels"
    sheet.append(["наименование файла", "таймстемп", "нарушение / без нарушения"])
    for name, timestamp, verdict in rows:
        sheet.append([name, timestamp, verdict])
        (videos / name).write_bytes((contents or {}).get(name, name.encode()))
    labels = tmp_path / "labels.xlsx"
    workbook.save(labels)
    database = tmp_path / "jobs.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE jobs(id TEXT,status TEXT,video_path TEXT,created_at REAL,result_json TEXT)")
        for index, path in enumerate(videos.iterdir()):
            connection.execute("INSERT INTO jobs VALUES(?,?,?,?,?)", (str(index), "done", str(path), index,
                               json.dumps({"verdict": "no_violation", "violations": []})))
    return labels, videos, database


def test_explicit_verdicts_do_not_turn_missing_points_into_negatives(tmp_path):
    labels, videos, database = make_reference_and_jobs(tmp_path, [
        ("positive-missing.mov", None, "нарушение"),
        ("positive-point.mov", "Synthetic timestamp: 1 мин 05 сек.", "нарушение"),
        ("negative.mov", None, "без нарушения"),
        ("fallback.mov", None, None),
        ("unknown.mov", None, "unrecognized-private-token"),
        ("conflict.mov", "Synthetic timestamp: 10 sec", "без нарушения"),
    ])
    report = evaluate(labels, videos, database)
    assert report["reference"]["explicit_verdict_column"] == "C"
    assert report["reference"]["positive_rows"] == report["reference"]["negative_rows"] == 2
    assert report["reference"]["row_counts"]["invalid_binary_rows"] == 2
    assert report["exploratory_binary"]["confusion"] == {"true_positive": 0, "false_positive": 0, "true_negative": 2, "false_negative": 2}
    assert report["exploratory_binary"]["completed_groups_excluded_for_label_issues"] == 2
    assert report["exploratory_localization"]["completed_groups_excluded_for_missing_timestamps"] == 1
    assert report["exploratory_localization"]["reference_points"] == 1
    assert point_times("Synthetic timestamp: 1 мин 05 сек.") == (65,)
    assert "unrecognized-private-token" not in json.dumps(report)
    assert "Synthetic timestamp" not in json.dumps(report)


def test_conflicting_duplicate_binary_labels_are_excluded(tmp_path):
    labels, videos, database = make_reference_and_jobs(tmp_path, [
        ("a.mov", None, "нарушение"), ("b.mov", None, "без нарушения"),
    ], {"a.mov": b"same video", "b.mov": b"same video"})
    report = evaluate(labels, videos, database)
    assert report["reference"]["matched_unique_hash_groups"] == 1
    assert report["reference"]["conflicting_binary_groups"] == 1
    assert report["exploratory_binary"]["evaluated_groups"] == 0
    assert report["exploratory_binary"]["completed_groups_excluded_for_label_issues"] == 1


def test_reference_snapshot_hash_does_not_follow_a_later_file_change(tmp_path):
    labels, videos, _ = make_reference_and_jobs(tmp_path, [("a.mov", None, "без нарушения")])
    original = labels.read_bytes()
    def hashing_with_source_change(path):
        labels.write_bytes(b"a later unrelated source change")
        return "synthetic-hash"
    _, summary, _ = read_reference(labels, videos, hashing_with_source_change)
    import hashlib
    assert summary["workbook_sha256"] == hashlib.sha256(original).hexdigest()


def test_candidate_timing_is_separate_from_legal_verdict_and_scene_text(tmp_path):
    labels, videos, database = make_reference_and_jobs(tmp_path, [("a.mov", "Synthetic timestamp: 10 sec", "нарушение")])
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE jobs ADD COLUMN details_json TEXT")
        connection.execute("UPDATE jobs SET details_json=?", (json.dumps({"versions": {"pipeline": "2.0"},
                           "candidate_events": [{"timecode_sec": 11, "disposition": "review", "type_id": "synthetic"}],
                           "private_scene_description": "must never appear in aggregate outputs"}),))
    report = evaluate(labels, videos, database)
    assert report["exploratory_binary"]["confusion"]["false_negative"] == 1
    assert report["exploratory_localization"]["matched_points"] == 0
    assert report["exploratory_candidate_localization"]["matched_points"] == 1
    assert report["exploratory_candidate_localization"]["recall"] == 1
    assert report["coverage"]["completed_pipeline_versions"] == {"2.0": 1}
    assert "must never appear" not in json.dumps(report)


def test_exact_pipeline_filter_uses_latest_done_within_that_version(tmp_path):
    labels, videos, database = make_reference_and_jobs(tmp_path, [("a.mov", "10 sec", "нарушение")])
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE jobs ADD COLUMN details_json TEXT")
        connection.execute("UPDATE jobs SET details_json=?", (json.dumps({"versions": {"pipeline": "2.0"}}),))
        connection.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)", ("newer-other-version", "done", str(videos / "a.mov"), 99,
                           json.dumps({"verdict": "violation", "violations": [{"timecode_sec": 10}]}),
                           json.dumps({"versions": {"pipeline": "1.0"}})))
    report = evaluate(labels, videos, database, pipeline_version="2.0")
    assert report["exploratory_binary"]["confusion"]["false_negative"] == 1
    assert report["coverage"]["completed_pipeline_versions"] == {"2.0": 1}
    assert evaluate(labels, videos, database)["exploratory_binary"]["confusion"]["true_positive"] == 1
    assert evaluate(labels, videos, database, pipeline_version="3.0")["coverage"]["completed_groups"] == 0


def test_deterministic_balanced_validation_keeps_disagreement_in_development(tmp_path, monkeypatch):
    rows = [(f"p-{index}.mov", "10 sec", "нарушение") for index in range(21)]
    rows += [(f"n-{index}.mov", None, "без нарушения") for index in range(19)]
    rows += [("p-copy.mov", "20 sec", "нарушение")]
    labels, videos, _ = make_reference_and_jobs(tmp_path, rows, {"p-0.mov": b"duplicated", "p-copy.mov": b"duplicated"})
    from types import SimpleNamespace
    monkeypatch.setattr("scripts.evaluate_local.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout=b'{"format":{"duration":"120"}}'))
    directory = tmp_path / "audit"
    summary = prepare_validation(labels, videos, directory)
    assert summary["total_groups"] == 40
    assert summary["development_groups"] == 32
    assert summary["validation_groups"] == 8
    assert summary["validation_positive_groups"] == summary["validation_negative_groups"] == 4
    assert summary["timestamp_disagreement_groups_development_only"] == 1
    first = (directory / "groups.json").read_bytes()
    prepare_validation(labels, videos, directory)
    assert first == (directory / "groups.json").read_bytes()
    selection = json.loads((directory / "validation-selection.json").read_text())
    assert sum(sample["binary"] for sample in selection["validation_samples"]) == 2
