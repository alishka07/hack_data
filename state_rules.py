"""Translate verified visible states into conservative, timed review episodes."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from rule_settings import DOOR_OPEN_THRESHOLD_LABEL, DOOR_OPEN_THRESHOLD_SEC, DOOR_OPEN_TITLE

SOURCE = 'Правило пользователя / кейса'
LIMITS = ['Прототип проверен на данном помещении и близких ракурсах; неизвестное состояние не считается закрытым.',
          'Длительность измерена между наблюдаемыми кадрами; краткое изменение между ними могло быть пропущено.',
          'Открытая или закрытая дверца не устанавливает состояние замка.']


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _coverage(key, title, status, detail, events):
    count = sum(event['rule_id'] == key for event in events)
    return dict(rule_id=key, title=title, status=status, detail=detail, reason=detail,
                event_count=count, events_count=count, rule_source=SOURCE, limitations=LIMITS,
                **({'user_threshold_sec': DOOR_OPEN_THRESHOLD_SEC} if key == 'door_open' else {}))


def _event(kind, rows, departure=None):
    first, last, middle = rows[0], rows[-1], rows[len(rows)//2]
    start, end = first['timestamp_sec'], last['timestamp_sec']
    key = 'door_open' if kind == 'door' else 'safe_closed'
    evidence = [dict(timestamp_sec=r['timestamp_sec'], source_frame=r['frame_index'],
                     mode='visual_states', playback_sec=r['timestamp_sec'], box=r[kind]['box'], labels=[kind])
                for r in (first, middle, last)]
    title = DOOR_OPEN_TITLE if kind == 'door' else 'Открытый сейф после возможного ухода кассира'
    description = (f'Дверь отмечена открытой с {start:.2f} до {end:.2f} с. Наблюдаемый интервал {end-start:.2f} с достигает указанного пользователем порога {DOOR_OPEN_THRESHOLD_LABEL}.'
                   if kind == 'door' else 'Дверца сейфа отмечена открытой после наблюдаемого ухода кандидата кассира. Проверьте, кто оставался ответственным за сейф.')
    return dict(id=hashlib.sha256(f'{key}:{start}:{end}'.encode()).hexdigest()[:16], rule_id=key,
                title=title, start_sec=start, end_sec=end, description=description,
                evidence_text=f'{start:.2f}–{end:.2f} с: {len(rows)} наблюдений открытого состояния.',
                source_frame=middle['frame_index'], mode='visual_states', playback_sec=middle['timestamp_sec'],
                evidence_box=middle[kind]['box'], evidence_times=evidence, labels=[kind],
                limitations=LIMITS + ([] if kind == 'door' else ['Роль кассира позиционная; окончание доступа и фактический присмотр не установлены.']),
                context={'locked': None, 'cashier_departure': departure, 'duration_basis': 'observed_sample_span',
                         **({'user_threshold_sec': DOOR_OPEN_THRESHOLD_SEC} if kind == 'door' else {})},
                severity='review', confidence_label='Требует проверки', rule_source=SOURCE)


def load_verified_states(folder, job):
    """Validate source, sampling and regions before any rule consumes them."""
    if not any(s.get('key') == 'visual_states' and s.get('status') == 'completed' for s in job.get('stages', [])):
        raise ValueError('этап состояний не завершён')
    folder = Path(folder)
    data = json.loads((folder/'visual_states/observations.json').read_text(encoding='utf-8'))
    meta = job['video']
    if (any(type(meta[key]) is not int or meta[key] <= 0 for key in ('frames', 'width', 'height'))
            or not _finite(meta['fps']) or meta['fps'] <= 0):
        raise ValueError('некорректные параметры исходного видео')
    if (data['status'] != 'completed' or data['frame_count_matches'] is not True or
            data['decoded_frames'] != meta['frames'] or data['expected_frames'] != meta['frames']):
        raise ValueError('неполная обработка')
    video = data['video']
    if (any(video[key] != meta[key] for key in ('width', 'height'))
            or not _finite(video['fps']) or abs(video['fps']-meta['fps']) > .01):
        raise ValueError('различаются параметры видео')
    sources = [p for p in folder.glob('input.*') if p.suffix.lower() in {'.mov','.mp4','.avi','.mkv','.m4v','.webm'}]
    if len(sources) != 1:
        raise ValueError('нет однозначного исходного видео')
    with sources[0].open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != video['sha256']:
            raise ValueError('состояния относятся к другому видео')
    rows, sample = data['observations'], data['sample_seconds']
    if not isinstance(rows, list) or not rows or not _finite(sample) or not .05 <= sample <= .5:
        raise ValueError('нет допустимой выборки')
    previous_t, previous_f = -1., -1
    for row in rows:
        t, f = row['timestamp_sec'], row['frame_index']
        if (not _finite(t) or type(f) is not int or not 0 <= f < meta['frames'] or t <= previous_t or f <= previous_f
                or abs(t - f/meta['fps']) > max(.15, 2/meta['fps'])):
            raise ValueError('повреждена последовательность кадров')
        for kind in ('door', 'safe'):
            item = row[kind]
            if item['state'] not in {'open','closed','unknown'}:
                raise ValueError('неверное состояние')
            if item['state'] != 'unknown':
                if not isinstance(row.get('scene'), dict) or row['scene'].get('matched') is not True:
                    raise ValueError('известное состояние без подтверждённого ракурса')
                box = item.get('box')
                if (not isinstance(box, list) or len(box) != 4 or not all(_finite(v) for v in box)
                        or not 0 <= box[0] < box[2] <= meta['width']
                        or not 0 <= box[1] < box[3] <= meta['height']):
                    raise ValueError('неверная область состояния')
        previous_t, previous_f = t, f
    if rows[0]['timestamp_sec'] > .15 or ((meta['frames']-1)/meta['fps'] - rows[-1]['timestamp_sec']) > sample+2/meta['fps']:
        raise ValueError('выборка не покрывает запись')
    for row in rows:
        regions = row.get('scene', {}).get('regions', {})
        for region in regions.get('reflections', []):
            box = region.get('box')
            if (not isinstance(box, list) or len(box) != 4 or not all(_finite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= meta['width']
                    or not 0 <= box[1] < box[3] <= meta['height']
                    or row.get('scene', {}).get('matched') is not True):
                raise ValueError('неверная область отражения')
    return data


def analyze_states(folder, job, service, verified_data=None):
    result = {'status': 'not_checked', 'events': [], 'coverage': [], 'state_counts': {}, 'reason': ''}
    def unavailable(reason):
        result['reason'] = reason
        result['coverage'] = [_coverage(key, title, 'not_checked', reason, []) for key, title in
                              [('door_open', DOOR_OPEN_TITLE),
                               ('safe_closed', 'Сейф закрыт после доступа и получения денег')]]
        return result
    if not any(s.get('key') == 'visual_states' and s.get('status') == 'completed' for s in job.get('stages', [])):
        return unavailable('Проверка состояний двери и сейфа не запускалась или не завершилась. Для старой записи нужен повторный анализ.')
    folder = Path(folder)
    try:
        data = verified_data if verified_data is not None else load_verified_states(folder, job)
        meta = job['video']
        rows, sample = data['observations'], data['sample_seconds']
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        return unavailable('Состояния не используются: ' + str(error) + '.')
    # The producer samples at the first frame reaching the requested interval.
    # At low FPS a 1.5x tolerance would bridge a missing observation (e.g. 4 FPS,
    # .25-second sampling, a missing frame creates .5 seconds). Use the actual
    # maximum CFR sampling stride; timestamp tolerance alone cannot catch it.
    max_frame_gap = max(1, math.ceil(sample * meta['fps'] - 1e-6))
    max_gap = max_frame_gap / meta['fps'] + 2e-6
    role = service.get('role_summary', {})
    qualified_departures = []
    if role.get('status') == 'candidate' and role.get('cashier_track_id') is not None:
        for departure in service.get('cashier_departures', []):
            if (isinstance(departure, dict) and departure.get('role_status') == 'candidate'
                    and departure.get('cashier_track_id') == role['cashier_track_id']
                    and departure.get('people_decode_complete') is True
                    and departure.get('no_visible_replacement') is True
                    and all(_finite(departure.get(key)) for key in ('exit_sec', 'start_sec', 'end_sec'))
                    and 0 <= departure['exit_sec'] <= departure['start_sec'] < departure['end_sec'] <= (meta['frames']-1)/meta['fps'] + 1e-6):
                qualified_departures.append(departure)
    for kind in ('door', 'safe'):
        result['state_counts'][kind] = dict(Counter(row[kind]['state'] for row in rows))
        active, runs = [], []
        for row in rows:
            if active and (row['timestamp_sec'] - active[-1]['timestamp_sec'] > max_gap
                           or row['frame_index'] - active[-1]['frame_index'] > max_frame_gap):
                runs.append(active); active = []
            if row[kind]['state'] == 'open':
                active.append(row)
            elif active:
                runs.append(active); active = []
        if active:
            runs.append(active)
        for run in runs:
            if kind == 'door':
                if run[-1]['timestamp_sec'] - run[0]['timestamp_sec'] >= DOOR_OPEN_THRESHOLD_SEC - 1e-6:
                    result['events'].append(_event(kind, run))
            else:
                for departure in qualified_departures:
                    after = [r for r in run if departure['start_sec'] <= r['timestamp_sec'] <= departure['end_sec']]
                    if len(after) >= 3 and after[-1]['timestamp_sec'] - after[0]['timestamp_sec'] >= 2 - 1e-6:
                        result['events'].append(_event(kind, after, departure))
                        break
    counts = result['state_counts']
    result['capabilities'] = data.get('capabilities', {})
    def usable(kind):
        return counts[kind].get('open',0)+counts[kind].get('closed',0) >= 3
    result['status'] = 'partial' if any(usable(k) for k in counts) else 'not_checked'
    result['reason'] = 'Наблюдаются состояния в согласованном ракурсе; неопределённые кадры прерывают интервалы.'
    result['coverage'] = [
        _coverage('door_open', DOOR_OPEN_TITLE, 'partial' if usable('door') else 'not_checked',
                  f'Проверяются непрерывные наблюдения открытой двери с порогом пользователя {DOOR_OPEN_THRESHOLD_LABEL}; неизвестные кадры не заполняются.' if usable('door') else 'Ракурс или состояние двери не удалось уверенно сопоставить с проверенными примерами.', result['events']),
        _coverage('safe_closed', 'Сейф закрыт после доступа и получения денег', 'partial' if usable('safe') else 'not_checked',
                  'Эксперимент: надёжность распознавания открытого сейфа на отдельных проверочных видео не подтверждена. Кандидат оставления открытой дверцы создаётся только после наблюдаемого ухода кандидата кассира; получение денег и запирание не устанавливаются.' if usable('safe') else 'Дверца сейфа не распознана в проверенном ракурсе. Экспериментальный детектор пропускает проверочные открытые состояния.', result['events'])]
    return result
