"""The repeatable metrics command reads saved JobStore reviews without writes."""
import hashlib
import json
from pathlib import Path
import sqlite3

from app.db import JobStore
from app.reviews import default_review_state, review_cases
from scripts.evaluate_review_quality import evaluate_saved_reviews, main


def test_cli_reads_actual_reviews_and_keeps_database_and_model_unchanged(tmp_path, capsys):
    database = tmp_path / "jobs.sqlite"
    store = JobStore(database)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source = uploads / "private-video.mov"
    source.write_bytes(b"synthetic footage bytes")
    details = {"case_decision": {"cases": [{"type_id": "personal_belongings", "status": "confirmed",
                                           "timecode_sec": 10, "probability": .8,
                                           "probability_scope": "joint_checks", "observed_action": "private scene prose"}]}}
    result = {"verdict": "violation", "violations": []}
    store.create_job("private-job-id", source, "private-video.mov")
    store.complete_job("private-job-id", result, details)
    case_id = review_cases(details, result)[0]["case_id"]
    state = {**default_review_state(), "reviewer": "private-reviewer", "source_group": "original",
             "camera_id": "camera", "session_id": "session", "coverage_complete": True,
             "decisions": [{"case_id": case_id, "decision": "dismissed", "comment": "private reviewer comment"}]}
    store.save_review("private-job-id", state)
    before_job, before_review = store.get_job("private-job-id"), store.get_review("private-job-id")
    # Complete a checkpoint first so the raw database snapshot is stable.
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    output = tmp_path / "metrics.json"
    assert main(["--db", str(database), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["totals"]["automatic_decision"]["false_positive"] == 1
    assert report["totals"]["automatic_decision"]["precision"] == 0
    assert report["records"]["with_review"] == 1
    encoded = output.read_text(encoding="utf-8")
    for private in ("private-video", "private-job-id", "private-reviewer", "private scene", "private reviewer", str(source)):
        assert private not in encoded
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before_hash
    assert store.get_job("private-job-id") == before_job
    assert store.get_review("private-job-id") == before_review
    assert store.get_source_sha256("private-job-id") == ""  # Legacy fallback hash is not cached into SQLite.
    assert "no training or calibration performed" in capsys.readouterr().out


def test_legacy_database_has_no_imaginary_feedback_and_is_not_migrated(tmp_path):
    database = tmp_path / "old.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE jobs(id TEXT,status TEXT,created_at REAL,video_path TEXT,result_json TEXT,details_json TEXT)")
        connection.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)", ("old", "done", 1, "absent.mov",
                           json.dumps({"verdict": "violation", "violations": [{"type": "screen", "timecode_sec": 10}]}), "{}"))
    report = evaluate_saved_reviews(database)
    assert report["status"] == "no_reviewed_labels"
    assert report["totals"]["automatic_decision"]["precision"] is None
    assert report["totals"]["candidate_adjudication"]["unreviewed"] == 1
    with sqlite3.connect(database) as connection:
        assert {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"jobs"}
