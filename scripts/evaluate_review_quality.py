"""Read-only, repeatable aggregate metrics for locally saved human reviews.

Only the requested JSON output is written. This never starts inference, imports
API settings/keys, changes SQLite, fits a calibrator, or trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.review_metrics import build_review_quality
from app.reviews import default_review_state, review_cases


def _object(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _fingerprint(job: dict, uploads: Path) -> str:
    if job.get("sha256"):
        return job["sha256"]
    # Same fallback as the UI adapter, without writing the resulting hash back.
    try:
        source = Path(job["video_path"]).resolve()
        if source.parent != uploads.resolve() or not source.is_file():
            return ""
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, KeyError, TypeError):
        return ""


def load_review_records(database: Path, uploads_dir: Path | None = None) -> list[dict]:
    """Read the JobStore schema in one read-only snapshot, including legacy jobs."""
    database = Path(database).resolve()
    uploads = Path(uploads_dir).resolve() if uploads_dir else database.parent / "uploads"
    records = []
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        review_column = "r.state_json" if "job_reviews" in tables else "NULL AS state_json"
        source_column = "s.sha256" if "job_sources" in tables else "NULL AS sha256"
        joins = (" LEFT JOIN job_reviews r ON r.job_id=j.id" if "job_reviews" in tables else "")
        joins += " LEFT JOIN job_sources s ON s.job_id=j.id" if "job_sources" in tables else ""
        query = (f"SELECT j.*, {review_column}, {source_column} FROM jobs j" + joins
                 + " WHERE j.status='done' ORDER BY j.created_at DESC,j.id DESC")
        for row in connection.execute(query):
            job = dict(row)
            details, result = _object(job.get("details_json")), _object(job.get("result_json"))
            state = _object(job.get("state_json")) or default_review_state()
            records.append({"job_id": job["id"], "source_fingerprint": _fingerprint(job, uploads),
                            "model_cases": review_cases(details, result), "review_state": state,
                            "created_at": job["created_at"], "review_updated_at": state.get("updated_at")})
    return records


def evaluate_saved_reviews(database: Path, uploads_dir: Path | None = None) -> dict:
    return build_review_quality(load_review_records(database, uploads_dir))


def main(argv: list[str] | None = None) -> int:
    data_dir = Path(os.environ.get("FORTE_DATA_DIR", ROOT / "data"))
    parser = argparse.ArgumentParser(description="Aggregate human review quality without changing the local database.")
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("FORTE_DB_PATH", data_dir / "jobs.sqlite")),
                        help="Existing local JobStore SQLite database")
    parser.add_argument("--uploads-dir", type=Path, default=None,
                        help="Optional uploads directory for legacy jobs without cached source hashes")
    parser.add_argument("--output", type=Path, required=True, help="Aggregate JSON output file; contains no media paths or comments")
    args = parser.parse_args(argv)
    database, output = args.db.resolve(), args.output.resolve()
    if output == database or str(output) in {str(database) + "-wal", str(database) + "-shm"}:
        parser.error("Output must be separate from the database and its sidecar files")
    try:
        report = evaluate_saved_reviews(database, args.uploads_dir)
    except (sqlite3.Error, OSError) as error:
        parser.error("Could not read the existing local database: " + str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    counts = report["records"]
    print("Evaluated analyses: " + str(counts["evaluated"]) + "; with human review: " + str(counts["with_review"])
          + "; complete event coverage: " + str(counts["eligible_complete_coverage"]) + ".")
    if report["status"] == "no_reviewed_labels":
        print("No decisive human labels: quality rates are unavailable (null), not zero accuracy.")
    else:
        print("Exploratory reviewed-subset metrics only; no training or calibration performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
