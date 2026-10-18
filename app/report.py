"""One offline HTML report, with escaped text and embedded declared JPEG evidence."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from app.review_presentation import (CHECK_LABELS, HUMAN_DECISIONS, SCOPES, format_time, justification,
                                     model_label, model_status, number, rows, title)
from app.reviews import review_cases


_PRESENCE = {"personal_belongings_in_cash_unit", "personal_food_at_workstation"}
_JPEG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:jpg|jpeg)\Z", re.I)
_MAX_IMAGE = 8 * 1024 * 1024
_MAX_EVIDENCE = 24 * 1024 * 1024
_CSS = """
*{box-sizing:border-box}body{margin:0;background:#f4f5f9;color:#202235;font:14px/1.65 system-ui,sans-serif}
main{max-width:1180px;margin:28px auto;padding:34px;background:#fff;border:1px solid #dfe2eb;border-radius:16px}
h1{font-size:27px;margin:0 0 8px}h2{font-size:20px;margin:30px 0 12px}h3{font-size:16px;margin:0 0 8px}
p{margin:8px 0}.muted,small{color:#60667a}.meta,.metrics{display:flex;flex-wrap:wrap;gap:12px 24px}.metrics>div{padding:12px 18px;background:#f4f5f9;border-radius:8px}
table{border-collapse:collapse;width:100%;margin:15px 0}th,td{padding:12px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #dfe2eb}thead{background:#f0f2f8}th{font-weight:650}
.case{padding:22px 0;border-top:1px solid #dfe2eb;break-inside:avoid}.badge{display:inline-block;padding:3px 8px;border-radius:5px;background:#f1f2f6;font-weight:600;font-size:12px}.review_required{background:#fff0c7;color:#714d00}.confirmed{background:#fbe3eb;color:#8e1847}
.human{padding:10px 14px;background:#f3f6fc;border-left:3px solid #737bc8;margin-top:12px}.comment{white-space:pre-wrap;overflow-wrap:anywhere}.occurrence{margin-top:14px;padding:12px 16px;border:1px solid #dfe2eb;border-radius:8px}
.evidence-links{display:flex;gap:10px;flex-wrap:wrap}a{color:#8e1847;text-underline-offset:3px}.evidence{break-inside:avoid;border:1px solid #dfe2eb;padding:16px;margin:20px 0}.evidence img{display:block;max-width:100%;max-height:740px;margin:auto}.notice{padding:12px 16px;background:#fff6df;border:1px solid #ead59a;border-radius:8px}
.check-list{padding-left:20px}.yes{color:#216444}.no{color:#923247}.unknown{color:#795800}.footnote{border-top:1px solid #dfe2eb;margin-top:28px;padding-top:16px;font-size:12px;color:#60667a}
@media print{body{background:white}main{margin:0;padding:0;border:0;max-width:none}.evidence img{max-height:220mm}a{color:inherit}thead{display:table-header-group}}
"""


def _text(value) -> str:
    return escape(value if isinstance(value, str) else str(value) if value is not None else "", quote=True)


def _decoded(value, default):
    if isinstance(value, type(default)):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, type(default)) else default
    except (TypeError, ValueError):
        return default


def _date(value) -> str:
    try:
        return datetime.fromtimestamp(value, timezone.utc).strftime("%d.%m.%Y %H:%M UTC") if number(value) else "—"
    except (ValueError, OverflowError, OSError):
        return "—"


def _interval(case, decision=None):
    decision = decision or {}
    if all(number(decision.get(key)) and decision[key] >= 0 for key in ("start_sec", "end_sec")):
        return decision["start_sec"], decision["end_sec"]
    if all(number(case.get(key)) and case[key] >= 0 for key in ("start_sec", "end_sec")):
        return case["start_sec"], case["end_sec"]
    anchor = case.get("timecode_sec")
    return (anchor, anchor) if number(anchor) and anchor >= 0 else (None, None)


def _span(start, end):
    return format_time(start) if start == end else f"{format_time(start)}–{format_time(end)}"


def _safe_source(value):
    if not isinstance(value, str):
        return ""
    try:
        url = urlsplit(value)
        if (url.scheme == "https" and url.hostname in {"adilet.zan.kz", "old.adilet.zan.kz", "nationalbank.kz", "www.nationalbank.kz"}
                and not url.username and not url.password and url.port in {None, 443}):
            return value
    except ValueError:
        pass
    return ""


class _Evidence:
    def __init__(self, directory: Path, details: dict):
        self.directory = directory.resolve()
        self.bytes = 0
        self.frames = []
        self.by_digest = {}
        self.limited = False
        self.declared = {}
        evidence = rows(details.get("evidence"))
        for finding in rows(details.get("findings")):
            evidence += rows(finding.get("evidence_frames", finding.get("evidence")))
        for item in evidence:
            name = item.get("filename")
            if isinstance(name, str) and _JPEG.fullmatch(name):
                self.declared[name] = item

    def add(self, frame: dict, backlink: str):
        raw, timestamp = None, frame.get("timestamp_sec")
        name = frame.get("filename")
        if isinstance(name, str) and name in self.declared:
            try:
                path = (self.directory / name).resolve()
                if path.parent == self.directory and path.is_file() and path.stat().st_size <= _MAX_IMAGE:
                    raw = path.read_bytes()
            except OSError:
                pass
            if not number(timestamp):
                timestamp = self.declared[name].get("timestamp_sec")
        if raw is None and isinstance(frame.get("image_base64"), str):
            encoded = re.sub(r"\Adata:image/jpe?g;base64,", "", frame["image_base64"], flags=re.I)
            if len(encoded) <= (_MAX_IMAGE * 4 // 3 + 4):
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    pass
        if not raw or len(raw) > _MAX_IMAGE or not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
            return None
        digest = hashlib.sha256(raw).hexdigest()
        # Same pixels at distinct source times are separate evidence references.
        key = (digest, timestamp if number(timestamp) else None)
        if key in self.by_digest:
            item = self.by_digest[key]
            if backlink not in item["backlinks"]:
                item["backlinks"].append(backlink)
            return item
        if self.bytes + len(raw) > _MAX_EVIDENCE:
            self.limited = True
            return None
        self.bytes += len(raw)
        item = {"id": f"frame-{len(self.frames) + 1}", "timestamp_sec": timestamp,
                "base64": base64.b64encode(raw).decode("ascii"), "backlinks": [backlink]}
        self.frames.append(item)
        self.by_digest[key] = item
        return item


def _matches(case, finding):
    if finding.get("type_id", finding.get("type")) != case.get("type_id"):
        return False
    if case.get("episode_id") and finding.get("episode_id"):
        return case["episode_id"] == finding["episode_id"]
    anchor = finding.get("candidate_timecode_sec", finding.get("timecode_sec"))
    return number(anchor) and number(case.get("timecode_sec")) and abs(anchor - case["timecode_sec"]) < .001


def _case_evidence(case, details, result, library, backlink):
    frames = []
    citations = [ref for check in rows(case.get("checks")) for ref in rows(check.get("evidence_refs"))]
    for item in library.declared.values():
        if any(number(ref.get("timestamp_sec")) and number(item.get("timestamp_sec"))
               and abs(ref["timestamp_sec"] - item["timestamp_sec"]) < .000001
               and ("source_index" not in item or item["source_index"] == ref.get("source_index")) for ref in citations):
            frames.append(item)
    for finding in rows(details.get("findings")) + rows(result.get("violations")) + rows(details.get("review_queue")):
        if _matches(case, finding):
            frames += rows(finding.get("evidence_frames", finding.get("evidence")))
    chosen = []
    for frame in frames:
        item = library.add(frame, backlink)
        if item and item not in chosen:
            chosen.append(item)
        if len(chosen) == 3:
            break
    return chosen


def _checks(case):
    items = []
    for index, check in enumerate(rows(case.get("checks")), 1):
        status = check.get("status") if check.get("status") in {"yes", "no"} else "unknown"
        # Free-form model labels remain escaped diagnostic data, never descriptions.
        identifier = check.get("name")
        name = CHECK_LABELS.get(identifier, f"Дополнительное условие {index}") if isinstance(identifier, str) else f"Дополнительное условие {index}"
        stage = {"action": "Видимое действие", "context": "Контекст правила", "exception_exclusion": "Исключение"}.get(check.get("stage"), "Проверка")
        times = sorted({ref["timestamp_sec"] for ref in rows(check.get("evidence_refs")) if number(ref.get("timestamp_sec")) and ref["timestamp_sec"] >= 0})
        label = {"yes": "Да", "no": "Нет", "unknown": "Не установлено"}[status]
        items.append(f'<li>{stage}: {_text(name)} — <span class="{status}">{label}</span>'
                     + (f' <small>({", ".join(format_time(time) for time in times[:6])})</small>' if times else "") + "</li>")
    return '<ul class="check-list">' + "".join(items) + "</ul>" if items else ""


def build_report(job: dict, details: dict, review_state: dict, evidence_dir: Path) -> str:
    """Generate a static document; no API keys, scripts, local paths or live requests."""
    details = details if isinstance(details, dict) else {}
    review_state = review_state if isinstance(review_state, dict) else {}
    result = _decoded(job.get("result_json", job.get("result")), {})
    cases = review_cases(details, result)
    decisions = {item.get("case_id"): item for item in rows(review_state.get("decisions"))}
    groups, keyed = [], {}
    for case in cases:
        start, end = _interval(case)
        key = (case.get("type_id"), model_status(case), model_label(case), case.get("probability_scope"), decisions.get(case["case_id"], {}).get("decision"))
        group = keyed.get(key) if case.get("type_id") in _PRESENCE and start is not None else None
        if group is None:
            group = {"cases": [], "start": start, "end": end}
            groups.append(group)
            if case.get("type_id") in _PRESENCE and start is not None:
                keyed[key] = group
        group["cases"].append(case)
        if start is not None:
            group["start"], group["end"] = min(group["start"], start), max(group["end"], end)
    groups.sort(key=lambda group: ({"review_required": 0, "confirmed": 1, "rejected": 2}[model_status(group["cases"][0])],
                                   group["start"] if group["start"] is not None else float("inf")))
    library = _Evidence(Path(evidence_dir), details)
    summary, sections = [], []
    for index, group in enumerate(groups, 1):
        members, anchor = group["cases"], f"case-{index}"
        first = members[0]
        human = decisions.get(first["case_id"], {})
        human_label = HUMAN_DECISIONS.get(human.get("decision"), HUMAN_DECISIONS["unreviewed"])
        estimates = [case["probability"] for case in members if number(case.get("probability")) and 0 <= case["probability"] <= 1]
        estimate = (f"{min(estimates):.0%}" if min(estimates, default=0) == max(estimates, default=0)
                    else f"{min(estimates):.0%}–{max(estimates):.0%}") if estimates else "Не оценено"
        scope = SCOPES.get(first.get("probability_scope"), "") if estimates else ""
        span, label = _span(group["start"], group["end"]), model_label(first)
        summary.append(f'<tr><th><a href="#{anchor}">{_text(title(first.get("type_id")))}</a></th><td>{_text(label)}</td>'
                       f'<td>{_text(human_label)}</td><td>{span}</td><td>{estimate}<br><small>{scope}</small></td></tr>')
        content = [f'<section class="case" id="{anchor}"><h3>{index}. {_text(title(first.get("type_id")))}</h3>',
                   f'<p><span class="badge {model_status(first)}">{_text(label)}</span> · Время модели: {span}</p>',
                   f'<p><strong>Обоснование — </strong>{_text(justification(members))}</p>']
        if len(members) > 1:
            content.append(f'<p class="muted">{len(members)} наблюдения. Диапазон объединяет отдельные кадры; непрерывное присутствие между ними не установлено.</p>')
        for member in members:
            decision = decisions.get(member["case_id"], {})
            corrected = (_span(decision["start_sec"], decision["end_sec"])
                         if all(number(decision.get(key)) and decision[key] >= 0 for key in ("start_sec", "end_sec")) else "Не задан")
            if len(members) > 1:
                content.append(f'<div class="occurrence"><strong>Наблюдение {format_time(member.get("timecode_sec"))}</strong>'
                               f'<p><strong>Обоснование — </strong>{_text(justification([member]))}</p>')
            content.append('<div class="human"><strong>Решение человека: '
                           + _text(HUMAN_DECISIONS.get(decision.get("decision"), HUMAN_DECISIONS["unreviewed"])) + '</strong>')
            if decision:
                content.append(f'<p>Проверяющий: {_text(review_state.get("reviewer"))} · Интервал проверяющего: {corrected}</p>')
                if decision.get("comment"):
                    content.append(f'<p class="comment">Комментарий проверяющего: {_text(decision["comment"])}</p>')
            content.append('</div>')
            selected = _case_evidence(member, details, result, library, anchor)
            content.append('<p class="evidence-links">' + ' '.join(f'<a href="#{frame["id"]}">Кадр {format_time(frame["timestamp_sec"])}</a>' for frame in selected) + '</p>'
                           if selected else '<p class="muted">Сохранённые кадры для этого наблюдения недоступны.</p>')
            content.append(_checks(member))
            matches = [finding for finding in rows(details.get("findings")) if _matches(member, finding)]
            if matches:
                finding = matches[0]
                policy = finding.get("basis") == "workplace_policy"
                content.append('<p><strong>Основание правила:</strong> ' + ("Политика организации" if policy else _text(finding.get("rule_name", finding.get("document", "Нормативное правило")))) + '</p>')
                if finding.get("paragraph"):
                    content.append(f'<p>{"Раздел" if policy else "Пункт"}: {_text(finding["paragraph"])}</p>')
                url = _safe_source(finding.get("rule_url", finding.get("source_url")))
                if url:
                    content.append(f'<p><a href="{_text(url)}" rel="noreferrer">Официальный источник</a></p>')
            if len(members) > 1:
                content.append('</div>')
        content.append('</section>')
        sections.append("".join(content))
    missed = []
    for event in rows(review_state.get("missed_events")):
        missed.append(f'<article class="case"><h3>{_text(title(event.get("type_id")))}</h3>'
                      f'<p>Интервал: {_span(event.get("start_sec"), event.get("end_sec"))}</p>'
                      f'<p class="comment">Комментарий проверяющего: {_text(event.get("comment", ""))}</p></article>')
    frames = []
    for frame in library.frames:
        backlinks = " · ".join(f'<a href="#{anchor}">К проверке {anchor.split("-")[-1]}</a>' for anchor in frame["backlinks"])
        frames.append(f'<figure class="evidence" id="{frame["id"]}"><img src="data:image/jpeg;base64,{frame["base64"]}" '
                      f'alt="Сохранённый кадр {format_time(frame["timestamp_sec"])}"><figcaption>Кадр {format_time(frame["timestamp_sec"])} · {backlinks}</figcaption></figure>')
    coverage = details.get("coverage") if isinstance(details.get("coverage"), dict) else {}
    incomplete = details.get("processing_status") == "review_incomplete" or coverage.get("verification_complete") is False
    model_verdict = "Есть подтверждённые моделью нарушения" if result.get("violations") else "Подтверждённых моделью нарушений не найдено"
    header = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Отчёт проверки видео — {_text(job.get('original_name', 'Видео'))}</title><style>{_CSS}</style></head><body><main>
<h1>Отчёт проверки видео</h1><p>{_text(job.get('original_name', 'Видео'))}</p>
<div class="meta"><span>Анализ: {_text(job.get('id', job.get('job_id', '')))}</span><span>Создан: {_date(job.get('created_at'))}</span><span>Редакция проверки: {_text(review_state.get('revision', 0))}</span></div>
<p><strong>Вывод модели:</strong> {model_verdict}.</p>
<p><strong>Проверяющий:</strong> {_text(review_state.get('reviewer')) or 'Не указан'} · <strong>Полный просмотр видео:</strong> {'Подтверждён проверяющим' if review_state.get('coverage_complete') is True else 'Не подтверждён'}.</p>
<p class="muted">Оценки модели не откалиброваны. Число может относиться только к видимому действию. Решения человека сохранены отдельно от исходного результата модели.</p>'''
    if incomplete:
        header += '<p class="notice">Анализ неполный: часть проверок не завершена. Отсутствие подтверждённого нарушения не означает, что весь ролик проверен.</p>'
    header += f'<div class="metrics"><div>Проверок в отчёте: <strong>{len(groups)}</strong></div><div>Решений человека: <strong>{len(decisions)}</strong></div><div>Добавлено событий: <strong>{len(missed)}</strong></div></div>'
    body = '<h2>Результаты</h2><table><thead><tr><th>Проверка</th><th>Вывод модели</th><th>Решение проверяющего</th><th>Время модели</th><th>Оценка модели</th></tr></thead><tbody>' + ''.join(summary) + '</tbody></table>' if cases else '<h2>Результаты</h2><p>Модель не сохранила отдельных проверок для этого видео.</p>'
    body += ''.join(sections)
    if missed:
        body += '<h2>События, добавленные проверяющим</h2><p class="muted">Это разметка человека, а не новые автоматические выводы модели.</p>' + ''.join(missed)
    body += '<h2>Кадры доказательств</h2>' + (''.join(frames) if frames else '<p>Сохранённые JPEG-кадры недоступны.</p>')
    if library.limited:
        body += '<p class="notice">Достигнут предел объёма встроенных кадров. Часть изображений не включена; проверьте исходный анализ в приложении.</p>'
    return header + body + '<p class="footnote">Автономный HTML-отчёт. Изображения встроены в этот файл; для его просмотра приложение и подключение к интернету не требуются. Исходное видео в отчёт не включено.</p></main></body></html>'
