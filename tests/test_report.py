"""Exports use synthetic evidence and must remain independent of the live service."""
from copy import deepcopy
from html.parser import HTMLParser
import base64
import json

import cv2
import numpy as np

from app.report import build_report
from app.review_presentation import justification
from app.reviews import default_review_state, review_cases


class Document(HTMLParser):
    def __init__(self, value):
        super().__init__()
        self.tags = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def _check(name, times, status="yes"):
    return {"stage": "action", "name": name, "status": status,
            "evidence_refs": [{"source_index": int(time * 30), "timestamp_sec": time} for time in times]}


def _case(time=0, **changes):
    return {"type_id": "personal_belongings_in_cash_unit", "status": "review_required", "requires_review": True,
            "episode_id": "window-" + str(time), "timecode_sec": time, "probability": None,
            "probability_scope": "action_only", "observed_action": "Unsupported theft or intention must stay hidden.",
            "checks": [_check("non_cash_object_visually_supported", [time]),
                       _check("object_cash_area_relationship_visible", [time])], **changes}


def _job(details, **changes):
    return {"id": "11111111-1111-4111-8111-111111111111", "original_name": "synthetic.mov", "created_at": 1792300000,
            "result_json": json.dumps({"verdict": "no_violation", "violations": []}), **changes}


def _details(cases):
    return {"case_decision": {"cases": cases}, "coverage": {"duration_sec": 30, "verification_complete": True}}


def _jpeg():
    success, encoded = cv2.imencode(".jpg", np.full((24, 32, 3), 120, dtype=np.uint8))
    assert success
    return encoded.tobytes()


def test_grouped_span_original_evidence_and_separate_human_decision(tmp_path):
    details = _details([_case(time) for time in (0, 8, 13, 17)])
    original = deepcopy(details)
    state = default_review_state()
    state.update(reviewer="Проверяющий", revision=2, coverage_complete=True)
    state["decisions"] = [{"case_id": case["case_id"], "decision": "confirmed", "start_sec": 0, "end_sec": 17,
                           "comment": "Проверено человеком"} for case in review_cases(details)]
    value = build_report(_job(details), details, state, tmp_path)
    document = Document(value)
    cases = [attrs for tag, attrs in document.tags if tag == "section" and attrs.get("class") == "case"]
    assert len(cases) == 1
    assert "00:00–00:17" in value and "4 наблюдения" in value
    assert "непрерывное присутствие" in value
    assert "Подтверждено проверяющим" in value and "Требует проверки" in value
    assert "Обоснование" in value and "В кассовой зоне виден предмет" in value
    assert "Опознание предмета, не относящегося к наличным" in value
    assert "non cash object visually supported" not in value
    assert "Unsupported theft" not in value and details == original
    assert "Редакция проверки: 2" in value and "Полный просмотр видео" in value


def test_missing_or_invalid_citations_never_turn_model_prose_into_facts(tmp_path):
    case = _case(checks=[_check("non_cash_object_visually_supported", [])], probability=True)
    details = _details([case])
    value = build_report(_job(details), details, default_review_state(), tmp_path)
    assert "По доступным кадрам действие не подтверждено." in value
    assert "Не оценено" in value and "<td>100%" not in value
    assert "Unsupported theft" not in value


def test_group_cannot_borrow_a_location_from_another_sample():
    first = _case(0, checks=[_check("non_cash_object_visually_supported", [0])])
    second = _case(8, checks=[_check("object_cash_area_relationship_visible", [8])])
    assert justification([first, second]).startswith("Наблюдения в этом диапазоне различаются")
    assert "В кассовой зоне" not in justification([first, second])


def test_human_interval_never_replaces_original_model_time(tmp_path):
    details = _details([_case(8)])
    state = default_review_state()
    state["decisions"] = [{"case_id": review_cases(details)[0]["case_id"], "decision": "confirmed",
                           "start_sec": 3, "end_sec": 17, "comment": "Уточнены границы"}]
    value = build_report(_job(details), details, state, tmp_path)
    assert "Время модели: 00:08" in value
    assert "Интервал проверяющего: 00:03–00:17" in value


def test_contradicted_and_insufficient_presence_are_not_merged(tmp_path):
    details = _details([_case(0, status="rejected", requires_review=False, action_decision="rejected", checks=[]),
                        _case(8, status="rejected", requires_review=False, action_decision="uncertain", checks=[])])
    value = build_report(_job(details), details, default_review_state(), tmp_path)
    assert "Не подтвердилось" in value and "Недостаточно данных" in value
    assert len([attrs for tag, attrs in Document(value).tags if tag == "section" and attrs.get("class") == "case"]) == 2


def test_huge_legacy_timestamps_cannot_crash_export(tmp_path):
    details = _details([_case(10**1000, checks=[])])
    value = build_report(_job(details), details, default_review_state(), tmp_path)
    assert "Отчёт проверки видео" in value


def test_one_frame_and_known_source_gaps_cannot_narrate_departure():
    case = _case(type_id="workstation_departure_screen_left_on", checks=[_check("workstation_departure_observed", [0]),
                                                                                 _check("same_screen_visibly_on_after_departure", [8])])
    assert justification([case]) == "Экран компьютера виден включённым."
    case["checks"][0] = _check("workstation_departure_observed", [0, 8])
    assert justification([case]) == "Человек покидает рабочее место. После ухода человека экран виден включённым."
    case["reason_codes"] = ["source_timestamp_gap"]
    assert justification([case]) == "Экран компьютера виден включённым."


def test_report_is_offline_embeds_declared_jpeg_once_and_escapes_text(tmp_path):
    image = _jpeg()
    (tmp_path / "source.jpg").write_bytes(image)
    case = _case(8)
    details = _details([case])
    details["evidence"] = [{"filename": "source.jpg", "timestamp_sec": 8, "source_index": 240}]
    state = default_review_state()
    state["reviewer"] = '<script>alert("reviewer")</script>'
    state["decisions"] = [{"case_id": review_cases(details)[0]["case_id"], "decision": "dismissed",
                           "comment": '<img src="https://evil.example/tracker" onerror="alert(1)">', "start_sec": 8, "end_sec": 8}]
    value = build_report(_job(details, original_name='<script src="https://evil.example/a"></script>.mov'), details, state, tmp_path)
    document = Document(value)
    assert not any(tag == "script" for tag, _ in document.tags)
    images = [attrs for tag, attrs in document.tags if tag == "img"]
    assert len(images) == 1 and images[0]["src"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(images[0]["src"].split(",", 1)[1]) == image
    assert '&lt;script' in value and '&lt;img' in value
    assert 'default-src \'none\'' in value
    assert not any(attrs.get("src", "").startswith(("http", "/api")) for _, attrs in document.tags)
    assert str(tmp_path) not in value


def test_undeclared_and_traversal_filenames_cannot_read_files(tmp_path):
    directory = tmp_path / "evidence"
    directory.mkdir()
    (tmp_path / "secret.jpg").write_bytes(_jpeg())
    (directory / "undeclared.jpg").write_bytes(_jpeg())
    details = _details([_case(8)])
    details["evidence"] = [{"filename": "../secret.jpg", "timestamp_sec": 8},
                           {"filename": "subdir/secret.jpg", "timestamp_sec": 8}]
    value = build_report(_job(details), details, default_review_state(), directory)
    assert not any(tag == "img" for tag, _ in Document(value).tags)
    assert "JPEG-кадры недоступны" in value


def test_export_uses_authorized_public_jpeg_and_safe_rule_links(tmp_path):
    case = _case(8, status="confirmed", requires_review=False, probability=.91, probability_scope="joint_checks")
    details = _details([case])
    details["findings"] = [{"type_id": case["type_id"], "timecode_sec": 8, "episode_id": case["episode_id"],
                            "rule_name": "Проверенное правило", "paragraph": "12", "rule_url": "javascript:alert(1)"}]
    result = {"verdict": "violation", "violations": [{"type": case["type_id"], "timecode_sec": 8,
               "evidence_frames": [{"timestamp_sec": 8, "image_base64": base64.b64encode(_jpeg()).decode()}]}]}
    value = build_report(_job(details, result_json=json.dumps(result)), details, default_review_state(), tmp_path)
    assert "91%" in value and "Все условия правила" in value
    assert any(tag == "img" for tag, _ in Document(value).tags)
    assert "javascript:" not in value
    details["findings"][0]["rule_url"] = "https://adilet.zan.kz/rus/docs/V1900019680#z81"
    assert 'href="https://adilet.zan.kz/rus/docs/V1900019680#z81"' in build_report(_job(details), details, default_review_state(), tmp_path)


def test_report_limits_evidence_explicitly_without_dropping_decisions(tmp_path, monkeypatch):
    image = _jpeg()
    (tmp_path / "source.jpg").write_bytes(image)
    monkeypatch.setattr("app.report._MAX_EVIDENCE", len(image) - 1)
    details = _details([_case(8)])
    details["evidence"] = [{"filename": "source.jpg", "timestamp_sec": 8}]
    state = default_review_state()
    state["missed_events"] = [{"type_id": "personal_food_at_workstation", "start_sec": 18, "end_sec": 20,
                               "comment": "Добавлено после просмотра"}]
    value = build_report(_job(details), details, state, tmp_path)
    assert "Достигнут предел объёма" in value
    assert "События, добавленные проверяющим" in value and "00:18–00:20" in value
    assert "Добавлено после просмотра" in value
    assert not any(tag == "img" for tag, _ in Document(value).tags)


def test_incomplete_analysis_and_missing_saved_description_are_explicit(tmp_path):
    details = _details([_case(status="confirmed", checks=[])])
    details["coverage"]["verification_complete"] = False
    value = build_report(_job(details), details, default_review_state(), tmp_path)
    assert "Анализ неполный" in value
    assert "Для описания действия недостаточно сохранённых данных" in value
    assert "Нарушение подтверждено моделью" in value


def test_real_authenticated_report_route_returns_one_complete_file(tmp_path):
    from uuid import uuid4
    from fastapi.testclient import TestClient
    from app.db import JobStore
    from app.main import create_app
    from app.settings import Settings

    settings = Settings(root=tmp_path, api_key="synthetic-report-key")
    store = JobStore(settings.db_path)
    store.heartbeat_worker("synthetic-report-worker", True)
    job_id = str(uuid4())
    source = settings.data_dir / "uploads" / f"{job_id}.mov"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"synthetic-source-only")
    store.create_job(job_id, source, "synthetic.mov")
    details = _details([_case(8)])
    details["evidence"] = [{"filename": "source.jpg", "timestamp_sec": 8, "source_index": 240}]
    directory = settings.data_dir / "evidence" / job_id
    directory.mkdir(parents=True)
    (directory / "source.jpg").write_bytes(_jpeg())
    store.complete_job(job_id, {"verdict": "no_violation", "violations": []}, details)
    with TestClient(create_app(settings, store)) as client:
        assert client.get(f"/api/ui/jobs/{job_id}/report.html").status_code == 401
        response = client.get(f"/api/ui/jobs/{job_id}/report.html", headers={"X-API-Key": "synthetic-report-key"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["content-disposition"] == f'attachment; filename="review-{job_id}.html"'
    assert response.headers["cache-control"] == "no-store"
    assert 'data:image/jpeg;base64,' in response.text
    assert 'synthetic-report-key' not in response.text and str(source) not in response.text
