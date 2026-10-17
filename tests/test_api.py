"""API checks use synthetic payloads only; private dataset content is never loaded."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.db import BusyError, JobStore
from app.main import create_app
from app.settings import Settings, load_settings


@pytest.fixture
def service(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path, api_key="synthetic-test-key", max_upload_bytes=1024)
    store = JobStore(settings.db_path)
    store.heartbeat_worker("test-worker", True)
    monkeypatch.setattr("app.main.probe_video", lambda *_: {"format": {"duration": "2"}})
    with TestClient(create_app(settings, store)) as client:
        yield client, store, settings


def _upload(client, **kwargs):
    return client.post("/api/v1/analyze", headers={"X-API-Key": "synthetic-test-key"},
                       files={"file": ("example.mov", b"synthetic-media", "video/quicktime")}, **kwargs)


def test_auth_is_required_for_jobs_upload_and_private_files(service):
    client, _, _ = service
    job_id = str(uuid4())
    for path in (f"/api/v1/jobs/{job_id}", f"/api/ui/jobs/{job_id}", f"/api/ui/jobs/{job_id}/video",
                 f"/api/ui/jobs/{job_id}/evidence/image.jpg", "/api/ui/rules", "/api/ui/catalog",
                 "/api/ui/active-job"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/v1/analyze", files={"file": ("example.mov", b"x")}).status_code == 401
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert "synthetic-test-key" not in health.text


def test_admission_status_and_exact_jury_result(service):
    client, store, _ = service
    response = _upload(client)
    assert response.status_code == 202
    assert set(response.json()) == {"job_id"}
    job_id = response.json()["job_id"]
    headers = {"X-API-Key": "synthetic-test-key"}
    assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).json() == {"job_id": job_id, "status": "processing"}
    claimed = store.claim_next("test-worker")
    assert claimed["id"] == job_id
    assert store.claim_next("test-worker") is None
    assert store.complete_job(job_id, {"verdict": "no_violation", "violations": [], "private": "not-public"},
                              {"review_notes": ["synthetic"]}, {"total_sec": 1})
    assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).json() == {
        "job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []}
    detail = client.get(f"/api/ui/jobs/{job_id}", headers=headers).json()
    assert detail["timings"] == {"total_sec": 1}
    assert detail["details"]["review_notes"] == ["synthetic"]
    assert detail["progress"] == 1
    assert _upload(client).status_code == 202


def test_busy_rejection_does_not_create_second_job_or_keep_upload(service):
    client, store, settings = service
    assert _upload(client).status_code == 202
    response = _upload(client)
    assert response.status_code == 429
    assert "job_id" not in response.json()
    assert response.headers["retry-after"] == "5"
    assert len(list((settings.data_dir / "uploads").glob("*.mov"))) == 1
    with pytest.raises(BusyError):
        store.create_job(str(uuid4()), "not-created.mov", "not-created.mov")


def test_active_job_discovery_is_minimal_and_not_cached(service):
    client, store, _ = service
    headers = {"X-API-Key": "synthetic-test-key"}
    response = client.get("/api/ui/active-job", headers=headers)
    assert response.json() == {"job_id": None}
    assert response.headers["cache-control"] == "no-store"
    job_id = _upload(client).json()["job_id"]
    response = client.get("/api/ui/active-job", headers=headers)
    assert response.json() == {"job_id": job_id, "status": "processing", "stage": "queued", "progress": 0}
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/api/v1/health").json()["busy"] is True
    assert job_id not in client.get("/api/v1/health").text
    store.claim_next("test-worker")
    store.update_progress(job_id, "verifying", 0.63, "private internal detail")
    assert client.get("/api/ui/active-job", headers=headers).json() == {
        "job_id": job_id, "status": "processing", "stage": "verifying", "progress": 0.63}


@pytest.mark.parametrize("terminal_status", ["done", "failed"])
def test_active_job_discovery_excludes_finished_jobs(service, terminal_status):
    client, store, _ = service
    job_id = _upload(client).json()["job_id"]
    if terminal_status == "done":
        store.complete_job(job_id, {"verdict": "no_violation", "violations": []})
    else:
        store.fail_job(job_id, "synthetic_error", "Synthetic failure")
    assert client.get("/api/ui/active-job", headers={"X-API-Key": "synthetic-test-key"}).json() == {"job_id": None}
    assert client.get("/api/v1/health").json()["busy"] is False


def test_active_job_discovery_recovers_orphans_but_preserves_live_analysis(service):
    client, store, _ = service
    headers = {"X-API-Key": "synthetic-test-key"}
    job_id = _upload(client).json()["job_id"]
    store.claim_next("test-worker")
    store.update_progress(job_id, "verifying", 0.5)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE jobs SET updated_at=? WHERE id=?", (time.time() - 100, job_id))
    assert client.get("/api/ui/active-job", headers=headers).json()["job_id"] == job_id
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE worker_lease SET heartbeat_at=?", (time.time() - 100,))
    assert client.get("/api/ui/active-job", headers=headers).json() == {"job_id": None}
    assert store.get_job(job_id)["error_code"] == "worker_lost"


def test_active_job_discovery_recovers_expired_analysis(service):
    client, store, _ = service
    job_id = _upload(client).json()["job_id"]
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE jobs SET deadline_at=? WHERE id=?", (time.time() - 1, job_id))
    assert client.get("/api/ui/active-job", headers={"X-API-Key": "synthetic-test-key"}).json() == {"job_id": None}
    assert store.get_job(job_id)["error_code"] == "deadline_exceeded"


def test_unready_worker_does_not_accept_video(service):
    client, store, settings = service
    store.heartbeat_worker("test-worker", False, "warming up")
    response = _upload(client)
    assert response.status_code == 503
    assert list((settings.data_dir / "uploads").iterdir()) == []
    assert store.active_job() is None


def test_invalid_oversized_and_unsupported_uploads_are_removed(service, monkeypatch):
    client, store, settings = service
    headers = {"X-API-Key": "synthetic-test-key"}
    assert client.post("/api/v1/analyze", headers=headers,
                       files={"file": ("bad.mp4", b"x")}).status_code == 415
    assert client.post("/api/v1/analyze", headers=headers,
                       files={"file": ("empty.mov", b"")}).status_code == 400
    assert client.post("/api/v1/analyze", headers=headers,
                       files={"file": ("large.mov", b"x" * 1025)}).status_code == 413
    assert client.post("/api/v1/analyze", headers=headers,
                       files={"file": ("larger.mov", b"x" * 70000)}).status_code == 413
    def invalid(*_):
        raise HTTPException(400, "Synthetic invalid stream")
    monkeypatch.setattr("app.main.probe_video", invalid)
    assert _upload(client).status_code == 400
    assert list((settings.data_dir / "uploads").iterdir()) == []
    assert store.active_job() is None


def test_chunked_upload_cannot_bypass_request_body_limit(service):
    client, store, settings = service
    boundary = "synthetic-boundary"
    prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"example.mov\"\r\n"
              "Content-Type: video/quicktime\r\n\r\n").encode()
    content = iter([prefix, b"x" * 70000, f"\r\n--{boundary}--\r\n".encode()])
    response = client.post("/api/v1/analyze", content=content, headers={
        "X-API-Key": "synthetic-test-key", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413
    assert store.active_job() is None
    assert list((settings.data_dir / "uploads").iterdir()) == []


def test_worker_failure_during_validation_removes_unaccepted_file(service, monkeypatch):
    client, store, settings = service
    def worker_stopped(*_):
        store.heartbeat_worker("test-worker", False)
    monkeypatch.setattr("app.main.probe_video", worker_stopped)
    assert _upload(client).status_code == 503
    assert store.active_job() is None
    assert list((settings.data_dir / "uploads").iterdir()) == []


def test_atomic_admission_rejection_removes_uploaded_file(service, monkeypatch):
    client, store, settings = service
    def busy(*_):
        raise BusyError("Synthetic concurrent admission")
    monkeypatch.setattr(store, "create_job", busy)
    assert _upload(client).status_code == 429
    assert store.active_job() is None
    assert list((settings.data_dir / "uploads").iterdir()) == []


def test_failed_contract_and_unknown_job(service):
    client, store, _ = service
    job_id = _upload(client).json()["job_id"]
    store.fail_job(job_id, "synthetic_error", "Synthetic technical failure")
    headers = {"X-API-Key": "synthetic-test-key"}
    assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).json() == {"job_id": job_id, "status": "failed"}
    assert client.get(f"/api/ui/jobs/{job_id}", headers=headers).json()["error"]["code"] == "synthetic_error"
    assert client.get(f"/api/v1/jobs/{uuid4()}", headers=headers).status_code == 404
    assert client.get("/api/v1/jobs/not-a-uuid", headers=headers).status_code == 404


def test_evidence_and_video_downloads_are_scoped(service):
    client, store, settings = service
    job_id = _upload(client).json()["job_id"]
    directory = settings.data_dir / "evidence" / job_id
    directory.mkdir(parents=True)
    (directory / "frame.jpg").write_bytes(b"synthetic-jpeg")
    (directory / "undeclared.jpg").write_bytes(b"not-in-result")
    store.complete_job(job_id, {"verdict": "no_violation", "violations": []},
                       {"findings": [{"evidence_frames": [{"filename": "frame.jpg"}]}]})
    headers = {"X-API-Key": "synthetic-test-key"}
    prefix = f"/api/ui/jobs/{job_id}"
    assert client.get(prefix + "/video", headers=headers).content == b"synthetic-media"
    assert client.get(prefix + "/evidence/frame.jpg", headers=headers).content == b"synthetic-jpeg"
    assert client.get(prefix + "/evidence/undeclared.jpg", headers=headers).status_code == 404
    assert client.get(prefix + "/evidence/api-key.txt", headers=headers).status_code == 404
    assert client.get(prefix + "/evidence/%2e%2e%5capi-key.txt", headers=headers).status_code == 404


def test_worker_leases_and_deadline_recovery(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.heartbeat_worker("owner", True)
    with pytest.raises(BusyError):
        store.heartbeat_worker("other", True)
    job_id = str(uuid4())
    store.create_job(job_id, "synthetic.mov", "synthetic.mov")
    assert store.claim_next("other") is None
    assert store.claim_next("owner")["owner"] == "owner"
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE jobs SET updated_at=? WHERE id=?", (time.time() - 100, job_id))
    assert store.recover_stale_jobs() == 0  # A live lease protects long inference.
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE worker_lease SET heartbeat_at=?", (time.time() - 100,))
    assert store.recover_stale_jobs() == 1
    assert store.get_job(job_id)["error_code"] == "worker_lost"
    assert not store.complete_job(job_id, {"verdict": "no_violation", "violations": []})
    next_id = str(uuid4())
    store.create_job(next_id, "synthetic.mov", "synthetic.mov")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE jobs SET deadline_at=? WHERE id=?", (time.time() - 1, next_id))
    assert store.recover_stale_jobs() == 1
    assert store.get_job(next_id)["error_code"] == "deadline_exceeded"


def test_only_owner_can_release_worker_lease(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    store.heartbeat_worker("owner", False, "Loading model")
    assert not store.release_worker("other")
    with pytest.raises(BusyError):
        store.heartbeat_worker("other", True)
    assert store.release_worker("owner")
    assert store.worker_state() is None
    store.heartbeat_worker("other", True)
    assert store.worker_state()["owner"] == "other"


def test_rules_and_static_content_are_local(service):
    client, _, settings = service
    (settings.root / "rules").mkdir()
    (settings.root / "rules" / "registry.json").write_text(json.dumps({"rules": []}), encoding="utf-8")
    assert client.get("/api/ui/rules", headers={"X-API-Key": "synthetic-test-key"}).json() == {"rules": []}
    assert client.get("/").status_code == 200


def test_catalog_is_authenticated_and_requires_valid_local_structure(service):
    client, _, settings = service
    headers = {"X-API-Key": "synthetic-test-key"}
    assert client.get("/api/ui/catalog", headers=headers).status_code == 503
    directory = settings.root / "rules"
    directory.mkdir()
    path = directory / "catalog.json"
    path.write_text("invalid json", encoding="utf-8")
    assert client.get("/api/ui/catalog", headers=headers).status_code == 503
    path.write_text(json.dumps({"version": "synthetic", "items": {}}), encoding="utf-8")
    assert client.get("/api/ui/catalog", headers=headers).status_code == 503
    value = {"version": "synthetic", "items": [
        {"id": "event", "label": "Synthetic event", "scoring_kind": "event_occurrence", "initial_score": 0},
        {"id": "procedure", "label": "Synthetic procedure", "scoring_kind": "required_procedure", "initial_score": 100},
    ]}
    path.write_text(json.dumps(value), encoding="utf-8")
    response = client.get("/api/ui/catalog", headers=headers)
    assert response.status_code == 200
    assert response.json() == value
    assert response.headers["cache-control"] == "no-store"


def test_assessment_scores_stay_in_ui_details_and_do_not_change_jury_verdict(service):
    client, store, _ = service
    job_id = _upload(client).json()["job_id"]
    assessments = [
        {"item_id": "event", "label": "Synthetic review-only event", "scoring_kind": "event_occurrence",
         "initial_score": 0, "score": 100, "applicability": "applicable", "assessment": "observed",
         "evidence_refs": [], "confirmed_finding": False},
        {"item_id": "procedure", "label": "Synthetic unassessed procedure", "scoring_kind": "required_procedure",
         "initial_score": 100, "score": 100, "applicability": "unknown", "assessment": "unassessed",
         "evidence_refs": [], "confirmed_finding": False},
        {"item_id": "excluded", "label": "Synthetic N/A procedure", "scoring_kind": "required_procedure",
         "initial_score": 100, "score": None, "applicability": "not_applicable", "assessment": "observed",
         "evidence_refs": [], "confirmed_finding": False},
    ]
    store.complete_job(job_id, {"verdict": "no_violation", "violations": []},
                       {"catalog_version": "synthetic", "assessments": assessments})
    headers = {"X-API-Key": "synthetic-test-key"}
    assert client.get(f"/api/v1/jobs/{job_id}", headers=headers).json() == {
        "job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []}
    assert client.get(f"/api/ui/jobs/{job_id}", headers=headers).json()["details"]["assessments"] == assessments


def test_root_html_assets_are_served_locally(service):
    client, _, settings = service
    web = settings.root / "web"
    web.mkdir()
    (web / "index.html").write_text('<link rel="stylesheet" href="/style.css"><script src="/app.js"></script>')
    (web / "style.css").write_text("body { color: black; }")
    (web / "app.js").write_text("'use strict';")
    assert client.get("/").status_code == 200
    css = client.get("/style.css")
    script = client.get("/app.js")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert script.status_code == 200 and "javascript" in script.headers["content-type"]
    assert client.get("/api-key.txt").status_code == 404


def test_workplace_policy_source_is_local_and_separate_from_regulations(service):
    client, _, settings = service
    assert client.get("/policies/workplace-eating").status_code == 404
    directory = settings.root / "rules" / "sources"
    directory.mkdir(parents=True)
    text = "User-provided workplace policy: Cashiers must not eat at the work table during work."
    (directory / "workplace-eating.md").write_text(text, encoding="utf-8")
    response = client.get("/policies/workplace-eating")
    assert response.status_code == 200 and response.text == text
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/policies/api-key.txt").status_code == 404


def test_settings_save_key_and_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("FORTE_ROOT", str(tmp_path))
    monkeypatch.delenv("FORTE_API_KEY", raising=False)
    first = load_settings()
    second = load_settings()
    assert len(first.api_key) >= 32
    assert first.api_key == second.api_key
    assert (tmp_path / "data" / "api-key.txt").is_file()
    monkeypatch.setenv("FORTE_API_KEY", "explicit-test-key")
    monkeypatch.setenv("FORTE_PORT", "8765")
    assert load_settings().api_key == "explicit-test-key"
    assert load_settings().port == 8765
