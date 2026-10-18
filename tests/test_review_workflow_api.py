"""Reviewer persistence/security checks use only generated media bytes and cases."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.db import JobStore, ReviewConflictError
from app.main import create_app
from app.reviews import default_review_state, review_cases
from app.settings import Settings


HEADERS = {"X-API-Key": "review-test-key"}


@pytest.fixture
def service(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path, api_key="review-test-key")
    store = JobStore(settings.db_path)
    monkeypatch.setattr("app.main.load_catalog", lambda *_: {"items": [{"id": "personal_belongings_in_cash_unit"},
                                                                    {"id": "manual_rule"}]})
    with TestClient(create_app(settings, store)) as client:
        yield client, store, settings


def completed(service, name="example.mov", duration=30.):
    _, store, settings = service
    identifier = str(uuid4())
    path = settings.data_dir / "uploads" / f"{identifier}.mov"
    path.write_bytes(b"synthetic-media-only")
    store.create_job(identifier, path, name)
    cases = [{"type_id": "personal_belongings_in_cash_unit", "episode_id": f"window-{index}",
              "timecode_sec": float(index * 8), "status": "review_required", "requires_review": True,
              "probability": .7, "observed_action": "An item is visible on the table"} for index in range(2)]
    details = {"case_decision": {"cases": cases}, "coverage": {"duration_sec": duration}}
    result = {"verdict": "no_violation", "violations": []}
    assert store.complete_job(identifier, result, details)
    return identifier, result, details


def fetched(service, identifier):
    return service[0].get(f"/api/ui/jobs/{identifier}", headers=HEADERS).json()


def state_for(payload):
    state = deepcopy(payload["review_state"])
    state.update(reviewer="Проверяющий", camera_id="camera-1", session_id="shift-1", source_group="original-1")
    return state


def test_saved_job_get_has_stable_cases_and_lazy_private_source_identity(service):
    identifier, result, details = completed(service)
    before = service[1].get_job(identifier)
    first, second = fetched(service, identifier), fetched(service, identifier)
    assert first["review_state"] == default_review_state()
    assert first["review_cases"] == second["review_cases"]
    assert len({row["case_id"] for row in first["review_cases"]}) == 2
    assert first["source_fingerprint"] == hashlib.sha256(b"synthetic-media-only").hexdigest()
    assert str(service[2].data_dir) not in json.dumps(first)
    assert service[1].get_job(identifier) == before
    assert first["result"] == result and first["details"] == details


def test_stable_ids_ignore_text_probability_changes_and_distinguish_duplicate_identity():
    case = {"type_id": "type", "episode_id": "episode", "timecode_sec": 1, "probability": .4}
    original = review_cases({"case_decision": {"cases": [case, case]}})
    changed = review_cases({"case_decision": {"cases": [{**case, "timecode_sec": 1., "probability": .9,
                                                         "observed_action": "updated"}, case]}})
    assert [row["case_id"] for row in original] == [row["case_id"] for row in changed]
    assert original[0]["case_id"] != original[1]["case_id"]
    assert "case_id" not in case


def test_human_group_decisions_missed_event_and_coverage_persist_without_changing_model(service):
    identifier, result, details = completed(service)
    state = state_for(fetched(service, identifier))
    state.update(coverage_complete=True, decisions=[{"case_id": case["case_id"], "decision": "dismissed",
                "comment": "При просмотре не подтверждено", "start_sec": 2., "end_sec": 10.}
                for case in fetched(service, identifier)["review_cases"]],
                 missed_events=[{"event_id": str(uuid4()), "type_id": "manual_rule", "start_sec": 20.,
                                 "end_sec": 23., "comment": "Добавлено проверяющим"}])
    response = service[0].put(f"/api/ui/jobs/{identifier}/review", headers=HEADERS, json=state)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    saved = response.json()
    assert saved["revision"] == 1 and saved["updated_at"] > 0
    assert fetched(service, identifier)["review_state"] == saved
    assert JobStore(service[2].db_path).get_review(identifier) == saved
    raw = service[1].get_job(identifier)
    assert json.loads(raw["result_json"]) == result and json.loads(raw["details_json"]) == details
    assert service[0].get(f"/api/v1/jobs/{identifier}", headers=HEADERS).json() == {
        "job_id": identifier, "status": "done", **result}
    # Whole-state replacement can clear a mistaken annotation without losing source decisions.
    cleared = {**saved, "decisions": [], "missed_events": [], "coverage_complete": False}
    assert service[0].put(f"/api/ui/jobs/{identifier}/review", headers=HEADERS, json=cleared).json()["revision"] == 2


def test_stale_revision_never_overwrites_another_review(service):
    identifier, _, _ = completed(service)
    state = state_for(fetched(service, identifier))
    path = f"/api/ui/jobs/{identifier}/review"
    assert service[0].put(path, headers=HEADERS, json=state).status_code == 200
    stale = service[0].put(path, headers=HEADERS, json={**state, "reviewer": "Other"})
    assert stale.status_code == 409
    assert fetched(service, identifier)["review_state"]["reviewer"] == "Проверяющий"


def test_concurrent_store_writes_compare_revision_atomically(service):
    identifier, _, _ = completed(service)
    state = state_for(fetched(service, identifier))
    def write():
        try:
            return service[1].save_review(identifier, state)["revision"]
        except ReviewConflictError:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: write(), range(2)))
    assert sorted(map(str, outcomes)) == ["1", "conflict"]


@pytest.mark.parametrize("change", [
    {"revision": True}, {"revision": "0"}, {"revision": -1}, {"revision": 2**53},
    {"reviewer": None}, {"reviewer": "x" * 121}, {"coverage_complete": "true"},
    {"camera_id": "\u0000"}, {"arbitrary": "field"}, {"decisions": {}}, {"missed_events": {}},
    {"decisions": [{"case_id": "unknown", "decision": "confirmed"}]},
    {"missed_events": [{"event_id": str(uuid4()), "type_id": "unknown", "start_sec": 1, "end_sec": 2}]},
    {"missed_events": [{"event_id": "../../escape", "type_id": "manual_rule", "start_sec": 1, "end_sec": 2}]},
])
def test_invalid_state_does_not_create_a_revision(service, change):
    identifier, _, _ = completed(service)
    state = {**state_for(fetched(service, identifier)), **change}
    response = service[0].put(f"/api/ui/jobs/{identifier}/review", headers=HEADERS, json=state)
    assert response.status_code == 422, response.text
    assert fetched(service, identifier)["review_state"]["revision"] == 0


@pytest.mark.parametrize("start,end", [(True, 5), ("2", 5), (-1, 5), (6, 5), (2, 31), (2, None),
                                         (None, 2), (float("nan"), 5), (2, float("inf"))])
def test_invalid_interval_is_rejected_for_known_case(service, start, end):
    identifier, _, _ = completed(service)
    payload = fetched(service, identifier)
    state = state_for(payload)
    state["decisions"] = [{"case_id": payload["review_cases"][0]["case_id"], "decision": "confirmed",
                           "start_sec": start, "end_sec": end}]
    response = service[0].put(f"/api/ui/jobs/{identifier}/review", headers={**HEADERS, "Content-Type": "application/json"},
                              content=json.dumps(state))
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("decision", [[], {}, True, "not_supported"])
def test_invalid_decision_cannot_trigger_server_error(service, decision):
    identifier, _, _ = completed(service)
    payload = fetched(service, identifier)
    state = state_for(payload)
    state["decisions"] = [{"case_id": payload["review_cases"][0]["case_id"], "decision": decision}]
    assert service[0].put(f"/api/ui/jobs/{identifier}/review", headers=HEADERS, json=state).status_code == 422


def test_unknown_case_duplicate_case_and_missing_reviewer_are_rejected(service):
    identifier, _, _ = completed(service)
    payload = fetched(service, identifier)
    state = state_for(payload)
    choice = {"case_id": payload["review_cases"][0]["case_id"], "decision": "uncertain"}
    for changes in ({"decisions": [choice, choice]}, {"decisions": [choice], "reviewer": " "},
                    {"coverage_complete": True, "reviewer": ""}):
        assert service[0].put(f"/api/ui/jobs/{identifier}/review", headers=HEADERS,
                             json={**state, **changes}).status_code == 422


def test_missing_duration_allows_decisions_but_blocks_corrected_intervals_and_coverage(service):
    identifier, _, _ = completed(service, duration=None)
    payload = fetched(service, identifier)
    state = state_for(payload)
    state["decisions"] = [{"case_id": payload["review_cases"][0]["case_id"], "decision": "uncertain"}]
    path = f"/api/ui/jobs/{identifier}/review"
    assert service[0].put(path, headers=HEADERS, json={**state, "coverage_complete": True}).status_code == 422
    assert service[0].put(path, headers=HEADERS, json=state).status_code == 200


def test_all_new_endpoints_require_auth_and_missing_jobs_stay_404(service):
    identifier = str(uuid4())
    for path in ("/api/ui/history", "/api/ui/quality", f"/api/ui/jobs/{identifier}/report.html"):
        assert service[0].get(path).status_code == 401
    assert service[0].put(f"/api/ui/jobs/{identifier}/review", json={}).status_code == 401
    assert service[0].put(f"/api/ui/jobs/{identifier}/review", headers=HEADERS, json={}).status_code == 404
    assert service[0].get(f"/api/ui/jobs/{identifier}/report.html", headers=HEADERS).status_code == 404


@pytest.mark.parametrize("status", ["processing", "failed"])
def test_review_and_report_only_accept_completed_jobs(service, status):
    identifier = str(uuid4())
    service[1].create_job(identifier, "synthetic.mov", "name.mov")
    if status == "failed":
        service[1].fail_job(identifier, "synthetic_error", "Synthetic failure")
    assert service[0].put(f"/api/ui/jobs/{identifier}/review", headers=HEADERS, json={}).status_code == 409
    assert service[0].get(f"/api/ui/jobs/{identifier}/report.html", headers=HEADERS).status_code == 409


def test_history_search_pagination_literal_wildcards_and_review_summary(service):
    first, _, _ = completed(service, "cash_100%.mov")
    second, _, _ = completed(service, "cash-other.mov")
    state = state_for(fetched(service, first))
    service[0].put(f"/api/ui/jobs/{first}/review", headers=HEADERS, json=state)
    response = service[0].get("/api/ui/history?query=cash&limit=1&offset=0", headers=HEADERS)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    page = response.json()
    assert page["total"] == 2 and page["limit"] == 1 and page["offset"] == 0
    assert page["items"][0]["job_id"] == second
    second_page = service[0].get("/api/ui/history?query=cash&limit=1&offset=1", headers=HEADERS).json()
    assert second_page["items"][0]["job_id"] == first
    assert second_page["items"][0]["review_revision"] == 1
    literal = service[0].get("/api/ui/history", params={"query": "_100%"}, headers=HEADERS).json()
    assert literal["total"] == 1 and literal["items"][0]["job_id"] == first
    assert service[0].get("/api/ui/history", params={"query": "' OR 1=1--"}, headers=HEADERS).json()["total"] == 0
    for params in ({"limit": 0}, {"limit": 101}, {"offset": -1}, {"query": "x" * 201}):
        assert service[0].get("/api/ui/history", params=params, headers=HEADERS).status_code == 422


def test_malformed_and_oversized_review_bodies_are_rejected(service):
    identifier, _, _ = completed(service)
    path = f"/api/ui/jobs/{identifier}/review"
    for raw in ("{incomplete", "[]", "null"):
        assert service[0].put(path, headers={**HEADERS, "Content-Type": "application/json"}, content=raw).status_code == 422
    assert service[0].put(path, headers=HEADERS, content="x" * (256 * 1024 + 1)).status_code == 413
    assert fetched(service, identifier)["review_state"]["revision"] == 0


def test_huge_integer_timestamp_is_invalid_instead_of_a_server_error(service):
    identifier, _, _ = completed(service)
    payload = fetched(service, identifier)
    state = state_for(payload)
    state["decisions"] = [{"case_id": payload["review_cases"][0]["case_id"], "decision": "confirmed",
                           "start_sec": 0, "end_sec": 10**400}]
    response = service[0].put(f"/api/ui/jobs/{identifier}/review", headers=HEADERS, json=state)
    assert response.status_code == 422
    assert fetched(service, identifier)["review_state"]["revision"] == 0


def test_deeply_nested_json_is_invalid_instead_of_a_server_error(service):
    identifier, _, _ = completed(service)
    raw = "[" * 1500 + "0" + "]" * 1500
    response = service[0].put(f"/api/ui/jobs/{identifier}/review",
                              headers={**HEADERS, "Content-Type": "application/json"}, content=raw)
    assert response.status_code == 422
    assert fetched(service, identifier)["review_state"]["revision"] == 0


def test_report_adapter_is_downloadable_private_html_and_uses_exact_evidence_directory(service, monkeypatch):
    identifier, _, _ = completed(service, '<script>alert("filename")</script>.mov')
    args = []
    def build_report(*values):
        args.extend(values)
        return "<!doctype html><p>Safe report</p>"
    import sys
    monkeypatch.setitem(sys.modules, "app.report", SimpleNamespace(build_report=build_report))
    response = service[0].get(f"/api/ui/jobs/{identifier}/report.html", headers=HEADERS)
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert response.headers["content-disposition"] == f'attachment; filename="review-{identifier}.html"'
    assert response.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert args[0]["id"] == identifier and args[3] == service[2].data_dir / "evidence" / identifier


def test_report_rejects_symlinked_job_evidence_directory_outside_evidence_root(service, monkeypatch):
    identifier, _, _ = completed(service)
    root = service[2].data_dir / "evidence"
    root.mkdir()
    outside = service[2].root / "outside-private-directory"
    outside.mkdir()
    try:
        (root / identifier).symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows without symlink privilege: model only this boundary's resolve
        # result, leaving the rest of the real filesystem and HTTP path intact.
        original_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            return outside if path == root / identifier else original_resolve(path, *args, **kwargs)
        monkeypatch.setattr(Path, "resolve", resolve)
    response = service[0].get(f"/api/ui/jobs/{identifier}/report.html", headers=HEADERS)
    assert response.status_code == 404
    assert str(outside) not in response.text


def test_quality_adapter_preserves_human_state_and_stable_model_case_ids(service, monkeypatch):
    identifier, _, _ = completed(service)
    import sys
    captured = []
    def build_review_quality(records):
        captured.extend(records)
        return {"synthetic_quality": True}
    monkeypatch.setitem(sys.modules, "app.review_metrics", SimpleNamespace(build_review_quality=build_review_quality))
    response = service[0].get("/api/ui/quality", headers=HEADERS)
    assert response.status_code == 200 and response.json() == {"synthetic_quality": True}
    record = captured[0]
    assert record["job_id"] == identifier and record["source_fingerprint"]
    assert all(row["case_id"].startswith("case_") for row in record["model_cases"])
    assert record["review_state"] == default_review_state()
