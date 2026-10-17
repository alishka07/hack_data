"""Small durable job queue with atomic admission and worker leases."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator


class BusyError(RuntimeError):
    """An upload or another live worker already holds the single processing slot."""


class JobStore:
    def __init__(self, db_path: str | Path, worker_stale_sec: int = 30):
        self.db_path = Path(db_path)
        self.worker_stale_sec = worker_stale_sec
        self.init()

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('processing','done','failed')),
                    internal_stage TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    deadline_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    owner TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    result_json TEXT,
                    details_json TEXT,
                    timings_json TEXT,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS worker_lease (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    owner TEXT NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    ready INTEGER NOT NULL,
                    detail TEXT NOT NULL DEFAULT ''
                );
            """)

    def create_job(self, job_id: str, video_path: str | Path, original_name: str,
                   timeout_sec: int = 300) -> dict[str, Any]:
        now = time.time()
        with self._connection(write=True) as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE status='processing' LIMIT 1").fetchone():
                raise BusyError("The local worker is already processing a video")
            connection.execute(
                "INSERT INTO jobs (id,status,internal_stage,video_path,original_name,created_at,deadline_at,updated_at) "
                "VALUES (?,'processing','queued',?,?,?,?,?)",
                (job_id, str(video_path), original_name, now, now + timeout_sec, now),
            )
            return dict(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def active_job(self) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE status='processing' LIMIT 1").fetchone()
            return dict(row) if row else None

    def claim_next(self, owner: str) -> dict[str, Any] | None:
        now = time.time()
        with self._connection(write=True) as connection:
            lease = connection.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            if (not lease or lease["owner"] != owner or not lease["ready"]
                    or now - lease["heartbeat_at"] > self.worker_stale_sec):
                return None
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='processing' AND internal_stage='queued' AND deadline_at>? "
                "ORDER BY created_at LIMIT 1", (now,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE jobs SET owner=?,internal_stage='starting',updated_at=? WHERE id=? AND internal_stage='queued'",
                (owner, now, row["id"]),
            )
            return dict(connection.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def heartbeat_worker(self, owner: str, ready: bool, detail: str = "") -> None:
        now = time.time()
        with self._connection(write=True) as connection:
            previous = connection.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            if previous and previous["owner"] != owner and now - previous["heartbeat_at"] <= self.worker_stale_sec:
                raise BusyError("Another live worker owns the local inference lease")
            connection.execute(
                "INSERT INTO worker_lease(singleton,owner,heartbeat_at,ready,detail) VALUES(1,?,?,?,?) "
                "ON CONFLICT(singleton) DO UPDATE SET owner=excluded.owner, heartbeat_at=excluded.heartbeat_at, "
                "ready=excluded.ready, detail=excluded.detail",
                (owner, now, int(ready), detail),
            )

    def worker_state(self) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            if not row:
                return None
            result = dict(row)
            result["ready"] = bool(result["ready"])
            return result

    def release_worker(self, owner: str) -> bool:
        """Release only the caller's lease after its model process has stopped."""
        with self._connection(write=True) as connection:
            cursor = connection.execute("DELETE FROM worker_lease WHERE singleton=1 AND owner=?", (owner,))
            return bool(cursor.rowcount)

    def update_progress(self, job_id: str, stage: str, progress: float, message: str = "") -> None:
        with self._connection(write=True) as connection:
            connection.execute(
                "UPDATE jobs SET internal_stage=?,progress=?,message=?,updated_at=? WHERE id=? AND status='processing'",
                (stage, max(0.0, min(float(progress), 1.0)), message, time.time(), job_id),
            )

    def complete_job(self, job_id: str, result: dict, details: dict | None = None,
                     timings: dict | None = None) -> bool:
        now = time.time()
        encoded = [json.dumps(value, ensure_ascii=False, allow_nan=False) for value in (result, details or {}, timings or {})]
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status='done',internal_stage='done',progress=1,message='',result_json=?, "
                "details_json=?,timings_json=?,updated_at=? WHERE id=? AND status='processing' AND deadline_at>?",
                (*encoded, now, job_id, now),
            )
            return bool(cursor.rowcount)

    def fail_job(self, job_id: str, error_code: str, message: str) -> bool:
        with self._connection(write=True) as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status='failed',internal_stage='failed',error_code=?,error_message=?,updated_at=? "
                "WHERE id=? AND status='processing'", (error_code, message, time.time(), job_id),
            )
            return bool(cursor.rowcount)

    def recover_stale_jobs(self, max_age_sec: int = 30) -> int:
        """Fail expired work and work orphaned by a dead/replaced worker; protect live leases."""
        now = time.time()
        recovered = 0
        with self._connection(write=True) as connection:
            lease = connection.execute("SELECT * FROM worker_lease WHERE singleton=1").fetchone()
            lease_live = bool(lease and now - lease["heartbeat_at"] <= max_age_sec)
            for row in connection.execute("SELECT * FROM jobs WHERE status='processing'").fetchall():
                deadline_passed = now >= row["deadline_at"]
                live_owner = bool(lease_live and (row["owner"] is None or row["owner"] == lease["owner"]))
                orphaned = not live_owner and now - row["updated_at"] > max_age_sec
                if deadline_passed or orphaned:
                    code = "deadline_exceeded" if deadline_passed else "worker_lost"
                    message = "The processing deadline expired" if deadline_passed else "The processing worker stopped responding"
                    connection.execute(
                        "UPDATE jobs SET status='failed',internal_stage='failed',error_code=?,error_message=?,updated_at=? WHERE id=?",
                        (code, message, now, row["id"]),
                    )
                    recovered += 1
        return recovered
