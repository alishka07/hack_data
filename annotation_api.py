"""Local, source-bound annotation workbench. Does not feed answers to detectors."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
import threading
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

VIDEO_ID = re.compile(r'^X[1-9][0-9]{0,3}$')
SHA256 = re.compile(r'^[a-f0-9]{64}$')
STATUSES = {'confirmed', 'rejected', 'uncertain'}
OBJECTS = {'cash', 'card', 'identity_card', 'document', 'food', 'drink', 'bag', 'safe',
           'banknote_counter', 'laptop', 'smartphone', 'key', 'stamp', 'person', 'person_reflection', 'other'}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def metadata(video):
    return video.get('metadata', video.get('video', {}))


def validate_annotation(value, video):
    if not isinstance(value, dict):
        raise ValueError('Отметка должна содержать поля события или предмета.')
    kind = value.get('kind')
    if kind not in {'event', 'object'} or value.get('status') not in STATUSES:
        raise ValueError('Выберите тип отметки и результат проверки.')
    label = value.get('label', '')
    if not isinstance(label, str) or not label.strip() or len(label) > 100:
        raise ValueError('Укажите название события или предмета.')
    note = value.get('note', '')
    if not isinstance(note, str) or len(note) > 4000:
        raise ValueError('Комментарий слишком длинный.')
    meta = metadata(video)
    proof = video.get('decode_proof', {})
    duration = proof.get('source_timeline_end_sec', proof.get('duration_sec', meta.get('duration', video.get('duration_sec', 0))))
    start, end = value.get('start_sec'), value.get('end_sec')
    first_pts = proof.get('first_timestamp_sec', proof.get('first_pts', 0))
    lower = min(0., first_pts) if kind == 'object' and finite(first_pts) else 0.
    if not all(finite(x) for x in (start, end, duration)) or not lower <= start <= end <= duration + .001:
        raise ValueError('Время должно находиться в пределах записи; конец не раньше начала.')
    result = dict(kind=kind, label=label.strip(), status=value['status'], start_sec=float(start),
                  end_sec=float(end), note=note, reviewer='user',
                  provenance='explicit_user_workbench_annotation',
                  independent_test_eligible=False, training_eligible=kind == 'object' and value['status'] == 'confirmed')
    source_note_id = value.get('source_note_id')
    if source_note_id is not None:
        ids = {n.get('id', n.get('note_id')) for n in video.get('notes', [])}
        if source_note_id not in ids:
            raise ValueError('Исходная отметка не принадлежит этому видео.')
        result['source_note_id'] = source_note_id
        source_note = next(n for n in video.get('notes', []) if n.get('id', n.get('note_id')) == source_note_id)
        result['source_note_provenance'] = source_note.get('provenance', {})
    if kind == 'object':
        if label not in OBJECTS:
            raise ValueError('Неизвестный класс предмета.')
        box = value.get('box')
        width, height = meta.get('width', 0), meta.get('height', 0)
        if (not isinstance(box, list) or len(box) != 4 or not all(finite(x) for x in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError('Обведите предмет внутри исходного кадра.')
        frame = value.get('source_frame')
        frames = meta.get('frames', 0)
        if type(frame) is not int or not 0 <= frame < frames or abs(end-start) > .001:
            raise ValueError('Предмет отмечается на одном исходном кадре.')
        # The frame/time relation is checked against actual decoded source PTS by the API.
        result.update(box=list(map(float, box)), source_frame=frame)
    result['source_sha256'] = video['source_sha256']
    if video.get('source_document_revision_id') is not None:
        result['source_document_revision_id'] = video['source_document_revision_id']
    result['id'] = uuid.uuid4().hex
    result['created_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return result


def register_annotation_routes(app, case: Path, review_root=None):
    root = Path(review_root or case/'annotations/video_review').resolve()
    lock = threading.Lock()
    hash_cache = {}

    def file_hash(path):
        path = Path(path)
        try:
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = hash_cache.get(str(path))
            if cached and cached[0] == stamp:
                return cached[1]
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = path.stat()
            if stamp != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise HTTPException(409, 'Исходный файл изменился во время проверки.')
            hash_cache[str(path)] = (stamp, digest)
            return digest
        except OSError:
            raise HTTPException(409, 'Исходный файл недоступен; привязку отметок проверить нельзя.')

    def catalog():
        data = read_json(root/'catalog.json')
        if not isinstance(data, dict) or not isinstance(data.get('videos'), list):
            raise HTTPException(503, 'Каталог видео ещё готовится. Обновите страницу позже.')
        seen = set()
        for row in data['videos']:
            if (not isinstance(row, dict) or not VIDEO_ID.fullmatch(str(row.get('video_id', '')))
                    or not SHA256.fullmatch(str(row.get('source_sha256', ''))) or row['video_id'] in seen):
                raise HTTPException(409, 'Некорректная привязка видео в каталоге.')
            seen.add(row['video_id'])
        source = data.get('source', {})
        if not isinstance(source, dict) or (source.get('revision_id') is not None
                and (not isinstance(source['revision_id'], str) or not source['revision_id'])):
            raise HTTPException(409, 'Некорректная версия исходного документа.')
        if source.get('original_notes'):
            original = (root/source['original_notes']).resolve()
            if not original.is_relative_to(root) or file_hash(original) != source.get('original_notes_sha256'):
                raise HTTPException(409, 'Снимок исходного документа изменился.')
        return data

    def entry(video_id):
        if not VIDEO_ID.fullmatch(video_id):
            raise HTTPException(404, 'Видео не найдено.')
        data = catalog()
        row = next((v for v in data['videos'] if v.get('video_id') == video_id), None)
        if row is None:
            raise HTTPException(404, 'Видео не найдено.')
        manifest = row.get('manifest_path')
        if manifest:
            path = (root/manifest).resolve()
            if not path.is_relative_to(root):
                raise HTTPException(409, 'Некорректный путь каталога.')
            full = read_json(path)
            if (not isinstance(full, dict) or full.get('video_id') != video_id
                    or full.get('source_sha256') != row['source_sha256']):
                raise HTTPException(409, 'Манифест относится к другой версии видео.')
            row = full
        row = dict(row)
        source = data.get('source', {})
        row['source_document_revision_id'] = source.get('revision_id')
        for note in row.get('notes', []):
            provenance = note.get('provenance', {})
            if note.get('source_sha256', row['source_sha256']) != row['source_sha256']:
                raise HTTPException(409, 'Исходная отметка относится к другой версии видео.')
            for key in ('document_id', 'revision_id', 'document_url'):
                if key in source and key in provenance and provenance[key] != source[key]:
                    raise HTTPException(409, 'Версия исходной заметки не совпадает с каталогом.')
        if row.get('source_path') and file_hash(Path(row['source_path'])) != row['source_sha256']:
            raise HTTPException(409, 'Исходное видео изменилось; требуется обновить каталог.')
        group = next((g for g in data.get('duplicate_groups', []) if video_id in g.get('video_ids', [])), None)
        if group:
            if group.get('source_sha256') != row['source_sha256']:
                raise HTTPException(409, 'Группа дубликатов не соответствует видео.')
            row['duplicate_group'] = group
        return row

    def review(video):
        path = root/'reviews'/f"{video['video_id']}.json"
        value = read_json(path)
        if path.exists():
            if (not isinstance(value, dict) or value.get('source_sha256') != video['source_sha256']
                    or value.get('video_id') != video['video_id'] or type(value.get('revision')) is not int
                    or value['revision'] < 0 or not isinstance(value.get('annotations'), list)
                    or not isinstance(value.get('history'), list)):
                raise HTTPException(409, 'Сохранённые отметки повреждены или относятся к другой версии видео.')
        return value or {'video_id': video['video_id'], 'source_sha256': video['source_sha256'],
                         'revision': 0, 'annotations': [], 'history': []}

    def source_revision_matches(value, video):
        expected = video.get('source_document_revision_id')
        return expected is None or value.get('source_document_revision_id') == expected

    def asset_path(relative):
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, 'Файл не найден.')
        return path

    def source_timeline(video):
        proof = video.get('decode_proof', {})
        path = asset_path(proof.get('frame_ledger', video.get('frame_timeline', f"videos/{video['video_id']}/frames.jsonl")))
        digest = file_hash(path)
        if proof.get('frame_ledger_sha256') and digest != proof['frame_ledger_sha256']:
            raise HTTPException(409, 'Журнал исходных кадров изменился.')
        try:
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        except (OSError, ValueError):
            raise HTTPException(409, 'Некорректный журнал исходных кадров.')
        count = proof.get('decoded_frames', metadata(video).get('frames'))
        if type(count) is not int or count <= 0 or len(rows) != count:
            raise HTTPException(409, 'Число исходных кадров не совпадает с каталогом.')
        previous = None
        for i, row in enumerate(rows):
            if (not isinstance(row, dict) or type(row.get('frame_index')) is not int or row['frame_index'] != i
                    or not finite(row.get('timestamp_sec'))
                    or previous is not None and row['timestamp_sec'] <= previous):
                raise HTTPException(409, 'Журнал кадров не задаёт однозначную исходную шкалу времени.')
            previous = row['timestamp_sec']
        first_pts = proof.get('first_timestamp_sec', proof.get('first_pts'))
        if first_pts is not None and (not finite(first_pts) or abs(rows[0]['timestamp_sec']-first_pts) > 1e-6):
            raise HTTPException(409, 'Начало исходной шкалы времени изменилось.')
        return rows, digest

    def playback_timeline(video, rows, ledger_hash):
        playback = read_json(root/'previews'/f"{video['video_id']}.timeline.json")
        if playback is None:
            return None, 'Карта кадров видеокопии ещё не готова.'
        if (not isinstance(playback, dict) or playback.get('verified') is not True
                or playback.get('mapping') != 'same_frame_index_after_full_count_verification'
                or playback.get('source_sha256') != video['source_sha256']
                or playback.get('source_frame_ledger_sha256') != ledger_hash):
            return None, 'Карта видеокопии не привязана к проверенной исходной записи.'
        times = playback.get('timestamps')
        if (not isinstance(times, list) or len(times) != len(rows)
                or any(not finite(t) or t < 0 for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))):
            return None, 'Карта видеокопии содержит неоднозначные времена кадров.'
        expected = playback.get('preview_sha256')
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            return None, 'Контрольная сумма видеокопии отсутствует.'
        preview = root/'previews'/f"{video['video_id']}.mp4"
        if not preview.is_file() or file_hash(preview) != expected:
            return None, 'Видеокопия изменилась после индексации.'
        return times, None

    @app.get('/annotations')
    async def annotation_page():
        return FileResponse(case/'web/annotations.html', media_type='text/html')

    @app.get('/api/annotations')
    async def annotation_catalog():
        data = catalog()
        result = []
        for v in data['videos']:
            r = review(v)
            result.append({k: val for k, val in v.items() if k not in {'samples', 'source_path', 'frames'}}
                          | {'review_count': len(r['annotations']),
                             'preview_ready': (root/'previews'/f"{v['video_id']}.mp4").is_file()})
        return {k: val for k, val in data.items() if k != 'videos'} | {
            'videos': result, 'source_document_revision_id': data.get('source', {}).get('revision_id')}

    @app.get('/api/annotations/export')
    async def annotation_export():
        data = catalog()
        records = []
        for v in data['videos']:
            full = entry(v['video_id'])
            records.append({'video_id': v['video_id'], 'source_sha256': v['source_sha256'],
                            'notes': full.get('notes', []), 'review': review(full),
                            'assistant_review': read_json(root/'assistant_reviews'/f"{v['video_id']}.json", {}),
                            'decode_proof': full.get('decode_proof')})
        return JSONResponse({'schema_version': '1.0', 'purpose': 'development_annotation_review',
                             'independent_test_eligible': False, 'source': data.get('source', {}),
                             'duplicate_groups': data.get('duplicate_groups', []), 'videos': records},
                            headers={'Content-Disposition': 'attachment; filename="video-annotations.json"'})

    @app.get('/api/annotations/assets/{relative:path}')
    async def annotation_asset(relative: str):
        path = asset_path(relative)
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.mp4'}:
            raise HTTPException(404, 'Файл не найден.')
        return FileResponse(path)

    @app.get('/api/annotations/{video_id}')
    async def annotation_video(video_id: str):
        video = entry(video_id)
        return {k: val for k, val in video.items() if k != 'source_path'} | {
            'review': review(video),
            'assistant_review': read_json(root/'assistant_reviews'/f'{video_id}.json', {}),
            'preview_url': f'/api/annotations/assets/previews/{video_id}.mp4'
                if (root/'previews'/f'{video_id}.mp4').is_file() else None}

    @app.get('/api/annotations/{video_id}/timeline')
    async def annotation_timeline(video_id: str):
        video = entry(video_id)
        rows, digest = source_timeline(video)
        playback, reason = playback_timeline(video, rows, digest)
        if playback is not None:
            for row, timestamp in zip(rows, playback):
                row['playback_timestamp_sec'] = timestamp
        return {'source_sha256': video['source_sha256'], 'source_document_revision_id': video.get('source_document_revision_id'),
                'source_frame_ledger_sha256': digest, 'playback_mapping_valid': playback is not None,
                'playback_mapping_reason': reason, 'frames': rows}

    @app.post('/api/annotations/{video_id}/review/{annotation_id}/remove')
    async def annotation_remove(video_id: str, annotation_id: str, request: Request):
        if not re.fullmatch(r'[a-f0-9]{32}', annotation_id):
            raise HTTPException(404, 'Отметка не найдена.')
        try:
            value = await request.json()
        except ValueError:
            raise HTTPException(422, 'Некорректный запрос.')
        video = entry(video_id)
        with lock:
            current = review(video)
            if (not isinstance(value, dict) or value.get('source_sha256') != video['source_sha256']
                    or not source_revision_matches(value, video) or type(value.get('revision')) is not int
                    or value['revision'] != current['revision']):
                raise HTTPException(409, 'Отметки изменились. Обновите страницу.')
            item = next((a for a in current['annotations'] if a['id'] == annotation_id), None)
            if item is None:
                raise HTTPException(404, 'Отметка не найдена.')
            current['annotations'] = [a for a in current['annotations'] if a['id'] != annotation_id]
            current['revision'] += 1
            current['history'].append({'revision':current['revision'], 'action':'remove', 'annotation':item,
                                       'at':datetime.now(timezone.utc).isoformat(timespec='seconds')})
            atomic_json(root/'reviews'/f'{video_id}.json', current)
        return current

    @app.post('/api/annotations/{video_id}/review')
    async def annotation_save(video_id: str, request: Request):
        try:
            value = await request.json()
        except ValueError:
            raise HTTPException(422, 'Некорректная отметка.')
        video = entry(video_id)
        if (not isinstance(value, dict) or value.get('source_sha256') != video['source_sha256']
                or not source_revision_matches(value, video)):
            raise HTTPException(409, 'Версия исходного видео изменилась. Обновите страницу.')
        try:
            item = validate_annotation(value.get('annotation'), video)
            if item['kind'] == 'object':
                timeline, _ = source_timeline(video)
                frame_row = timeline[item['source_frame']] if item['source_frame'] < len(timeline) else None
                if frame_row is None or abs(frame_row['timestamp_sec']-item['start_sec']) > .001:
                    raise ValueError('Кадр и время не совпадают с исходной записью.')
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error))
        with lock:
            current = review(video)
            if type(value.get('revision')) is not int or value['revision'] != current['revision']:
                raise HTTPException(409, 'Отметки изменились в другой вкладке. Обновите страницу.')
            current['annotations'].append(item)
            current['revision'] += 1
            current['history'].append({'revision': current['revision'], 'action': 'add',
                                        'annotation_id': item['id'], 'at': item['created_at']})
            atomic_json(root/'reviews'/f'{video_id}.json', current)
        return current
