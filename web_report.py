"""Local evidence images and portable rule-review exports (no model imports)."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from pathlib import Path
import threading
import time
import uuid

REPORT_LOCK = threading.RLock()


def _write(path: Path, data: bytes):
    try:
        if path.read_bytes() == data:
            return
    except OSError:
        pass
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(data)
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.05 * 2**attempt, .5))
    finally:
        temporary.unlink(missing_ok=True)


def _evidence(folder, job, events):
    """Extract the exact sampled source frame, never substitute a nearby thumbnail."""
    sources = [p for p in folder.glob('input.*') if p.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}]
    if not sources or not events:
        return
    import cv2
    import math
    cap = None
    try:
        for event in events:
            frame_index = event.get('source_frame')
            if not isinstance(frame_index, int) or frame_index < 0:
                continue
            fingerprint = hashlib.sha256(json.dumps({
                'frame': frame_index, 'box': event.get('evidence_box'),
                'source_bytes': sources[0].stat().st_size,
            }, sort_keys=True).encode()).hexdigest()[:20]
            picture = folder / 'report/evidence' / (fingerprint + '.jpg')
            if not picture.exists():
                if cap is None:
                    cap = cv2.VideoCapture(str(sources[0]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
                if not cap.isOpened():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if not ok or actual != frame_index:
                    event.setdefault('limitations', []).append('Не удалось извлечь точный исходный кадр; проверьте момент в видео.')
                    continue
                box = event.get('evidence_box')
                if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
                    x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 190, 255), 2)
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    continue
                picture.parent.mkdir(parents=True, exist_ok=True)
                _write(picture, encoded.tobytes())
            relative = picture.relative_to(folder).as_posix()
            event['thumbnail_url'] = '/api/jobs/' + job['id'] + '/files/' + relative
            event['evidence_file'] = picture.relative_to(folder / 'report').as_posix()
    finally:
        if cap is not None:
            cap.release()


def save_rule_report(folder: Path, job: dict, report: dict):
    """Enrich events in-place and atomically write downloadable JSON/CSV/HTML."""
    folder = Path(folder)
    with REPORT_LOCK:
        target = folder / 'report'
        target.mkdir(exist_ok=True)
        _evidence(folder, job, report.get('events', []))
        data = {'video': job.get('name'), 'job_id': job['id'], **report}
        _write(target / 'violations.json', json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['rule_id', 'start_sec', 'end_sec', 'title', 'status', 'description', 'evidence', 'limitations', 'steps_json'])
        for event in report.get('events', []):
            writer.writerow([event.get('rule_id'), event.get('start_sec'), event.get('end_sec'), event.get('title'),
                             'Требует проверки', event.get('description'), event.get('evidence_text'),
                             '; '.join(event.get('limitations', [])), json.dumps(event.get('steps', []), ensure_ascii=False)])
        _write(target / 'violations.csv', stream.getvalue().encode('utf-8-sig'))
        esc = lambda value: html.escape(str(value or ''), quote=True)
        def ordered_steps(steps):
            lines = []
            for step in steps:
                if isinstance(step, str):
                    lines.append('<li>' + esc(step) + '</li>')
                    continue
                stamp = step.get('timestamp_sec')
                stamp = f'{stamp:.2f} с · ' if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) else ''
                lines.append('<li><b>' + esc(stamp + step.get('title', 'Шаг')) + '</b><p>' + esc(step.get('detail')) + '</p></li>')
            return '<ol>' + ''.join(lines) + '</ol>' if lines else ''
        cards = []
        for event in report.get('events', []):
            picture = '<img src="' + esc(event['evidence_file']) + '" alt="Исходный кадр с областью наблюдения">' if event.get('evidence_file') else ''
            limits = ''.join('<li>' + esc(line) + '</li>' for line in event.get('limitations', []))
            steps = '<h3>Что произошло по шагам</h3>' + ordered_steps(event['steps']) if event.get('steps') else ''
            cards.append(f'<article><h2>{esc(event.get("title"))}</h2><b>{esc(event.get("start_sec"))}–{esc(event.get("end_sec"))} с · Требует проверки</b><p>{esc(event.get("description"))}</p>{steps}<p>{esc(event.get("evidence_text"))}</p>{picture}<ul>{limits}</ul></article>')
        statuses = {'partial': 'Частично', 'not_checked': 'Не проверено', 'checked': 'Проверено'}
        def check_details(row):
            lines = []
            for check in row.get('checks', []):
                status = ('Наблюдений: ' + str(check.get('observation_count', 0)) if check.get('observation_count') else 'Проверено, признаков не найдено') if check.get('available') else 'Недостаточно данных'
                lines.append('<li><b>' + esc(check.get('title')) + '</b> — ' + esc(status) + '. ' + esc(check.get('detail')) + '</li>')
            checks = '<ul>' + ''.join(lines) + '</ul>' if lines else ''
            if row.get('procedure_steps'):
                title = 'Порядок проверки — ещё не реализовано' if row.get('procedure_status') == 'planned' else 'Как проверяем, шаг за шагом'
                checks += '<details><summary>' + title + '</summary><p>' + esc(row.get('procedure_note')) + '</p>' + ordered_steps(row['procedure_steps']) + '</details>'
            return checks
        coverage = ''.join('<tr><td>' + esc(row.get('title')) + '</td><td>' + esc(row.get('status_label') or statuses.get(row.get('status'), row.get('status'))) + '</td><td>' + esc(row.get('detail')) + check_details(row) + '</td></tr>' for row in report.get('rule_coverage', []))
        page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Проверка записи</title><style>body{{font:16px/1.6 'Segoe UI',sans-serif;margin:36px auto;padding:0 24px;max-width:1050px;color:#202d43}}h1{{font-size:30px}}article{{border-top:2px solid #9b214b;padding:20px 0}}img{{display:block;max-width:100%;max-height:560px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border:1px solid #dbe0e8;text-align:left;vertical-align:top}}small{{color:#596578}}</style><h1>Проверка записи: {esc(job.get('name'))}</h1><p>{esc(report.get('summary'))}</p><p>Это возможные признаки нарушений по правилам пользователя и кейса. Требуется просмотр человеком; нормативные статьи автоматически не устанавливаются.</p>{''.join(cards)}<h2>Какие правила удалось проверить</h2><table><thead><tr><th>Правило</th><th>Охват</th><th>Пояснение</th></tr></thead><tbody>{coverage}</tbody></table><p><small>Отсутствие эпизодов не подтверждает отсутствие нарушений. Кадры доступны при открытии этого отчёта из распакованного архива.</small></p></html>'''
        _write(target / 'review.html', page.encode('utf-8'))
    base = '/api/jobs/' + job['id'] + '/files/report/'
    return [{'label': 'Отчёт по правилам — HTML', 'url': base + 'review.html'},
            {'label': 'Возможные нарушения — CSV', 'url': base + 'violations.csv'},
            {'label': 'Эпизоды и охват правил — JSON', 'url': base + 'violations.json'}]
