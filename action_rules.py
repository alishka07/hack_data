"""Observable temporal checks built from validated local video journals.

These checks describe projected motion and visible presence. They never certify
cash counting, ID authenticity, ownership, operation completion or room safety.
Availability is reported independently of whether a review candidate is found.
No model inference or additional video decode is performed here.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import hashlib
import math

VERSION = '2.0'
KINDS = {'cash': 'cash', 'banknote': 'cash', 'banknotes': 'cash',
         'document': 'document', 'paper_document': 'document',
         'identity_card': 'card', 'card': 'card', 'bank_card': 'card'}
TABLE_LABELS = {'work_table', 'table', 'desk'}
MIN_SCORE = .4
STABLE_SECONDS = 1.
ABSENCE_SECONDS = 2.
COMMON = [
    'Это проверяемый визуальный признак, а не автоматически установленное нарушение.',
    'Роли кассира и посетителя определяются по расположению; принадлежность предметов неизвестна.',
    'Класс небольшого предмета может быть ошибочным; банкноты, документы и карточки требуют проверки по кадрам.',
    'Проекции рамок не доказывают контакт с рукой или столом, передачу и работу счётчика.',
    'Исчезновение у края кадра не доказывает выход из помещения; конец посещения не равен концу операции.',
]
TITLES = {
    'cash_after_client': 'Деньги убраны после ухода клиента',
    'previous_client_documents': 'На столе нет документов прошлого клиента',
    'unattended_room': 'Кабинет не оставлен без присмотра',
    'cash_placed_on_counter': 'Клиент выкладывает деньги на стол или в лоток',
    'cash_counted_visible': 'Деньги пересчитаны через счётчик на виду у клиента',
    'change_through_counter': 'Остаток и возврат проходят через счётчик',
    'original_id_verification': 'Проверка оригинала удостоверения клиента',
}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    return (list(map(float, value)) if isinstance(value, (list, tuple)) and len(value) == 4
            and all(_number(v) for v in value) and value[2] > value[0] and value[3] > value[1] else None)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    overlap = _intersection(a, b)
    return overlap / max(1e-9, _area(a) + _area(b) - overlap)


def _inside(box, container, fraction=.6):
    x, y = _center(box)
    return (container[0] <= x <= container[2] and container[1] <= y <= container[3]
            and _intersection(box, container) / _area(box) >= fraction)


def _distance(a, b):
    ac, bc = _center(a), _center(b)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


def _token(*parts):
    return hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16]


def _evidence(row):
    return {'timestamp_sec': row['time'], 'source_frame': row['frame'],
            'mode': row.get('mode', 'people'), 'playback_sec': row.get('playback_sec', row['time']),
            'box': row['box'], 'labels': [row.get('label', 'person')]}


def _event(rule_id, title, rows, description, context):
    rows = sorted(rows, key=lambda row: (row['time'], row['frame']))
    # Limit export size, retain start/middle/end of each visible action.
    evidence = rows if len(rows) <= 9 else [rows[round(i * (len(rows) - 1) / 8)] for i in range(9)]
    reference = rows[len(rows) // 2]
    return {'id': _token('action', rule_id, rows[0]['frame'], rows[-1]['frame'], context),
            'rule_id': rule_id, 'title': title,
            'start_sec': rows[0]['time'], 'end_sec': rows[-1]['time'],
            'source_frame': reference['frame'], 'mode': reference.get('mode', 'people'),
            'playback_sec': reference.get('playback_sec', reference['time']),
            'evidence_box': reference['box'], 'evidence_times': [_evidence(row) for row in evidence],
            'evidence_text': f"{rows[0]['time']:.2f}–{rows[-1]['time']:.2f} с: последовательность видимых наблюдений.",
            'description': description, 'context': {'analysis_source': 'observable_actions_v2', **context},
            'labels': sorted({row.get('label', 'person') for row in rows}),
            'limitations': list(COMMON), 'status': 'review', 'severity': 'review',
            'confidence_label': 'Требует проверки', 'rule_source': 'Правило пользователя / кейса'}


def _edge_exit(track, width, height):
    if len(track) < 3:
        return False
    tail = [row for row in track if track[-1]['time'] - row['time'] <= 1.]
    if len(tail) < 3 or tail[-1]['time'] - tail[0]['time'] < .3:
        return False
    box = tail[-1]['box']
    distances = [box[0], box[1], width - box[2], height - box[3]]
    edge = min(range(4), key=lambda i: distances[i])
    if distances[edge] > max(8., .03 * min(width, height)):
        return False
    first, last = _center(tail[0]['box']), _center(tail[-1]['box'])
    movement = [first[0] - last[0], first[1] - last[1], last[0] - first[0], last[1] - first[1]][edge]
    return movement >= max(8., .02 * min(width, height))


def _long(rows, seconds=STABLE_SECONDS):
    return len(rows) >= 3 and rows[-1]['time'] - rows[0]['time'] >= seconds - 1e-6


def _stationary(rows):
    return bool(rows) and all(_iou(rows[0]['box'], row['box']) >= .65 for row in rows)


def _reflection_ids(people, role):
    excluded = set(role.get('excluded_reflection_track_ids', []))
    excluded.update(people.get('reflections', {}).get('excluded_track_ids', []))
    ambiguous = {identity for pair in role.get('reflection_candidate_pairs', [])
                 if not any(identity in excluded for identity in pair) for identity in pair}
    return excluded, ambiguous


def _records(modes, fps, frames):
    """Validate temporal records again and build a small shared frame index."""
    records, context, interval, issues = {}, defaultdict(list), {}, []
    for mode, data in modes.items():
        rows, previous = [], None
        step = data.get('interval')
        interval[mode] = step if _number(step) and step > 0 else None
        for raw in data.get('records', []):
            time, frame = raw.get('time'), raw.get('frame')
            if (not _number(time) or time < 0 or type(frame) is not int or not 0 <= frame < frames
                    or abs(time - frame / fps) > max(.15, 2 / fps)
                    or (previous and (time <= previous['time'] or frame <= previous['frame']))):
                issues.append({'mode': mode, 'reason_code': 'invalid_temporal_journal'})
                rows = []
                break
            objects = []
            for obj in raw.get('objects', []):
                box, score = _box(obj.get('box')), obj.get('score')
                if box and _number(score) and MIN_SCORE <= score <= 1:
                    objects.append({**obj, 'box': box})
            rows.append({**raw, 'objects': objects, 'mode': mode})
            previous = raw
        records[mode] = rows
        for row in rows:
            context[row['frame']].extend(row['objects'])
    return records, context, interval, issues


def _object_sequences(records, context, intervals, people, role, fps):
    """Link only unique same-class boxes on consecutive sampled frames.

    Low-overlap small displacement is allowed for moving objects, but every
    competing association breaks the chain. Missing objects and samples break
    continuity. No interpolation can invent passage through the counter.
    """
    all_sequences, counts = [], defaultdict(int)
    excluded, ambiguous = _reflection_ids(people, role)
    cashier = role.get('cashier_track_id')
    visitors = set(role.get('visitor_track_ids', [])) - excluded - ambiguous
    anchor = _box(role.get('anchors', {}).get('table_box'))
    for mode, rows in records.items():
        previous, previous_objects = None, []
        step = intervals[mode]
        for row in rows:
            current = []
            contexts = context[row['frame']]
            tables = [obj['box'] for obj in contexts if obj.get('label') in TABLE_LABELS
                      and (not anchor or _iou(obj['box'], anchor) >= .6)]
            counters = [obj['box'] for obj in contexts if obj.get('label') == 'banknote_counter']
            for obj in row['objects']:
                kind = KINDS.get(obj.get('label'))
                if not kind:
                    continue
                counts[kind] += 1
                conflicting = any(KINDS.get(other.get('label')) not in {kind, None}
                                  and _iou(obj['box'], other['box']) >= .45 for other in contexts)
                hardware_conflict = any(other.get('label') in {'safe', 'banknote_counter', 'laptop', 'bag', 'computer_monitor'}
                                        and _iou(obj['box'], other['box']) >= .7 for other in contexts)
                food_conflict = any(other.get('label') in {'food', 'drink'}
                                    and _iou(obj['box'], other['box']) >= .45 for other in contexts)
                if conflicting or hardware_conflict or food_conflict:
                    counts['class_conflicts'] += 1
                    continue
                near = [p['id'] for p in people.get('by_frame', {}).get(row['frame'], [])
                        if p.get('id') not in excluded and _box(p.get('box')) and _inside(obj['box'], p['box'], .25)]
                one = near[0] if len(set(near)) == 1 and near[0] not in ambiguous else None
                candidate = {**obj, 'kind': kind, 'time': row['time'], 'frame': row['frame'], 'mode': mode,
                             'playback_sec': row.get('playback_sec', row['time']),
                             'over_table': any(_inside(obj['box'], table) for table in tables),
                             'table_visible': bool(tables), 'counter_visible': bool(counters),
                             'at_counter': any(_intersection(obj['box'], box) / _area(obj['box']) >= .25 for box in counters),
                             'near_track_id': one, 'near_role': ('cashier' if one == cashier and one is not None
                                                               else 'visitor' if one in visitors else 'unknown')}
                current.append(candidate)
            can_link = bool(previous and step and row['time'] - previous['time'] <= step + 1 / fps + 1e-9)
            forward, backward = defaultdict(list), defaultdict(list)
            if can_link:
                for old_index, old in enumerate(previous_objects):
                    for new_index, new in enumerate(current):
                        if old['kind'] != new['kind']:
                            continue
                        scale_ratio = _area(old['box']) / _area(new['box'])
                        diagonal = min(math.hypot(box[2] - box[0], box[3] - box[1]) for box in (old['box'], new['box']))
                        if .5 <= scale_ratio <= 2 and (_iou(old['box'], new['box']) >= .15 or _distance(old['box'], new['box']) <= .65 * diagonal):
                            forward[old_index].append(new_index)
                            backward[new_index].append(old_index)
            for new_index, item in enumerate(current):
                parents = backward[new_index]
                old = previous_objects[parents[0]] if len(parents) == 1 and len(forward[parents[0]]) == 1 else None
                if old:
                    index = old['_sequence']
                else:
                    index = len(all_sequences)
                    all_sequences.append({'id': _token('object', mode, item['frame'], item['kind'], item['box']),
                                          'kind': item['kind'], 'rows': []})
                item['_sequence'] = index
                all_sequences[index]['rows'].append(item)
            previous, previous_objects = row, current
    for sequence in all_sequences:
        for row in sequence['rows']:
            row.pop('_sequence', None)
    return all_sequences, dict(counts)


def _visitor_departures(service, people, job, role):
    video = job['video']
    duration = (video['frames'] - 1) / video['fps']
    excluded, ambiguous = _reflection_ids(people, role)
    sessions = [s for s in service.get('sessions', []) if s.get('sustained') and not s.get('reflection_ambiguous')
                and s.get('track_id') not in excluded | ambiguous]
    departures, rejected = [], []
    table = _box(role.get('anchors', {}).get('table_box'))
    cashier = role.get('cashier_track_id')
    for session in sessions:
        identity = session['track_id']
        track = people.get('by_id', {}).get(identity, [])
        if not track or not _edge_exit(track, video['width'], video['height']):
            rejected.append({'track_id': identity, 'reason_code': 'no_visible_edge_exit'})
            continue
        exit_sec = track[-1]['time']
        if (not 0 <= exit_sec - session['end_sec'] <= 6.
                or duration - exit_sec < ABSENCE_SECONDS
                or any(other['track_id'] == identity and other['start_sec'] > session['end_sec'] for other in sessions)):
            rejected.append({'track_id': identity, 'reason_code': 'departure_window_incomplete'})
            continue
        # Simultaneous companions are not separate completed operations.
        if any(other['track_id'] != identity and other['start_sec'] <= exit_sec
               and other['end_sec'] >= session['end_sec'] - .5 for other in sessions):
            rejected.append({'track_id': identity, 'reason_code': 'visitor_group_or_ongoing_visit'})
            continue
        start_frame = track[-1]['frame'] + 1
        end_frame = min(video['frames'] - 1, round((exit_sec + ABSENCE_SECONDS) * video['fps']))
        unknown = False
        for frame in range(start_frame, end_frame + 1):
            for person in people.get('by_frame', {}).get(frame, []):
                if person.get('id') in excluded or person.get('id') == cashier:
                    continue
                box = _box(person.get('box'))
                # Nearby new/renumbered people make client departure ambiguous.
                if box and table and _intersection(box, table) / min(_area(box), _area(table)) >= .05:
                    unknown = True
                    break
            if unknown:
                break
        if unknown:
            rejected.append({'track_id': identity, 'reason_code': 'person_remains_or_track_replacement'})
            continue
        departures.append({'track_id': identity, 'exit_sec': exit_sec,
                           'session_start_sec': session['start_sec'], 'session_end_sec': session['end_sec'],
                           'exit_row': track[-1], 'absence_verified_until_sec': end_frame / video['fps'],
                           'operation_completion_known': False})
    # A track can have several positional runs; use its last qualifying run once.
    distinct = {row['track_id']: row for row in departures}
    return list(distinct.values()), rejected


def _sequence(kind, title, rows, **context):
    return {'id': _token(kind, rows[0]['frame'], rows[-1]['frame'], rows[0].get('mode')),
            'kind': kind, 'title': title, 'start_sec': rows[0]['time'], 'end_sec': rows[-1]['time'],
            'status': 'observation', 'evidence_times': [_evidence(row) for row in rows],
            'context': context, 'limitations': list(COMMON)}


def analyze_actions(modes: dict, people: dict, service: dict, job: dict) -> dict:
    """Return compatible events, per-rule availability, and concrete subchecks.

    Parent must replace the legacy global-count cash-after-client implementation
    with this module's rule rather than combining both. ``checks`` are explicit
    observable subchecks; ``sequences`` are explanatory observations and do not
    themselves count as violations. The original ID/counting rules remain only
    partially covered even when a corresponding projected sequence is visible.
    """
    events, sequences, checks = [], [], []
    result = {'version': VERSION, 'status': 'not_checked', 'events': events, 'coverage': [],
              'checks': checks, 'sequences': sequences, 'visitor_departures': [],
              'limitations': list(COMMON), 'settings': {'minimum_stable_observations': 3,
              'minimum_stable_span_sec': STABLE_SECONDS, 'post_exit_observation_sec': ABSENCE_SECONDS,
              'note': 'Параметры устойчивости наблюдений; не нормативные ограничения.'}}
    role = service.get('role_summary', {})
    video = job.get('video', {})
    fps, frames = video.get('fps'), video.get('frames')
    valid_video = (_number(fps) and fps > 0 and type(frames) is int and frames > 0
                   and all(_number(video.get(k)) and video[k] > 0 for k in ('width', 'height')))
    people_ok = bool(valid_video and people.get('complete') is True and people.get('frames') == frames
                     and _number(people.get('fps')) and abs(people['fps'] - fps) < .01)
    table = _box(role.get('anchors', {}).get('table_box'))
    excluded, ambiguous = _reflection_ids(people, role)
    roles_ok = bool(people_ok and role.get('status') == 'candidate' and type(role.get('cashier_track_id')) is int
                    and role['cashier_track_id'] not in excluded | ambiguous and table)

    def check(check_id, rule_id, title, available, detail, observation_count=0, reason_code=None, unsupported=None):
        row = {'check_id': check_id, 'rule_id': rule_id, 'title': title,
               'implementation_status': 'implemented',
               'status': 'observed' if available and observation_count else 'analyzed' if available else 'insufficient_evidence',
               'reason_code': reason_code or ('observations_found' if observation_count else 'no_candidates' if available else 'insufficient_evidence'),
               'detail': detail, 'observation_count': observation_count, 'available': bool(available),
               'unsupported': unsupported or []}
        checks.append(row)
        return row

    if not valid_video:
        records, context, intervals, issues, object_sequences, counts = {}, {}, {}, [], [], {}
    else:
        records, context, intervals, issues = _records(modes, fps, frames)
        object_sequences, counts = _object_sequences(records, context, intervals, people if people_ok else {}, role, fps)
    desk_usable = bool(records.get('desk') and intervals.get('desk'))
    groups = modes.get('desk', {}).get('summary', {}).get('groups')
    cash_detector = desk_usable and (not isinstance(groups, list) or 'cash' in groups)
    document_detector = desk_usable and (not isinstance(groups, list) or 'desk' in groups)
    stable_table_seen = bool(table and sum(any(obj.get('label') in TABLE_LABELS and _iou(obj['box'], table) >= .6
                                             for obj in objects) for objects in context.values()) >= 3)
    role_reason = ('people_journal_unavailable' if not people_ok else 'role_or_table_ambiguous' if not roles_ok
                   else 'table_not_visible' if not stable_table_seen else None)
    departure_available = bool(roles_ok and stable_table_seen)
    departures, rejected = _visitor_departures(service, people, job, role) if departure_available else ([], [])
    result['visitor_departures'] = [{k: v for k, v in row.items() if k != 'exit_row'} for row in departures]
    result['rejected_departures'] = rejected
    checks.append({'check_id': 'visitor_edge_exit', 'rule_id': 'cash_after_client',
                   'title': 'Посетитель ушёл от стола к краю кадра; другой посетитель не остался',
                   'implementation_status': 'implemented', 'available': departure_available,
                   'status': 'observed' if departures else 'analyzed' if departure_available else 'insufficient_evidence',
                   'reason_code': 'observations_found' if departures else role_reason or 'no_qualified_visitor_exit',
                   'observation_count': len(departures),
                   'detail': 'Используется позиционная роль посетителя и движение к краю, а не уменьшение общего числа людей.'})

    for departure in departures:
        for chain in object_sequences:
            if chain['kind'] not in {'cash', 'document'}:
                continue
            rows = chain['rows']
            prior = [row for row in rows if max(departure['session_start_sec'], departure['exit_sec'] - 2.5) <= row['time'] <= departure['exit_sec']]
            after = [row for row in rows if departure['exit_sec'] < row['time'] <= departure['absence_verified_until_sec']]
            if not (_long(prior) and _long(after) and all(row['over_table'] for row in prior + after)
                    and _stationary(prior + after)):
                continue
            rule_id = 'cash_after_client' if chain['kind'] == 'cash' else 'previous_client_documents'
            cash = chain['kind'] == 'cash'
            observed = prior[-3:] + after
            description = ('После движения посетителя от стола к краю кадра похожие банкноты продолжают наблюдаться на том же месте в проекции стола.' if cash else
                           'После движения посетителя от стола к краю кадра похожий документ продолжает наблюдаться на том же месте в проекции стола. Принадлежность документа этому посетителю неизвестна.')
            event = _event(rule_id, 'Похожие банкноты остались после ухода посетителя' if cash else 'Документ остался после ухода посетителя', observed,
                           description + ' Проверьте кадры до и после ухода; завершение операции не установлено.',
                           {'visitor_track_id': departure['track_id'], 'cashier_track_id': role['cashier_track_id'],
                            'visitor_exit_sec': departure['exit_sec'], 'people_decode_complete': True,
                            'global_people_count_used': False, 'same_projected_location': True,
                            'ownership_known': False, 'operation_completion_known': False,
                            'object_chain_id': chain['id']})
            event['evidence_times'].append(_evidence(departure['exit_row']))
            if not cash:
                following = [session for session in service.get('sessions', [])
                             if session.get('track_id') != departure['track_id'] and session.get('sustained')
                             and session.get('entry_motion_observed') is True and not session.get('reflection_ambiguous')
                             and session.get('track_id') not in excluded | ambiguous
                             and session.get('start_sec', -1) > departure['absence_verified_until_sec']]
                for visitor in sorted(following, key=lambda session: session['start_sec']):
                    carry = [row for row in rows if after[-1]['time'] < row['time'] <= visitor['start_sec'] + 1.5]
                    during_next = [row for row in carry if row['time'] >= visitor['start_sec']]
                    if (_long(during_next) and all(row['over_table'] for row in carry)
                            and _stationary(prior + after + carry)):
                        event['context'].update(next_visitor_track_id=visitor['track_id'], next_visit_start_sec=visitor['start_sec'])
                        event['description'] += ' Тот же кандидат документа остаётся на месте, когда от края кадра приходит следующий посетитель.'
                        event['evidence_times'].append(_evidence(during_next[-1]))
                        sequences.append(_sequence('document_persists_into_next_visit', 'Документ остался между двумя наблюдаемыми посещениями',
                                                   [prior[-1], after[-1], during_next[-1]],
                                                   previous_visitor_track_id=departure['track_id'],
                                                   next_visitor_track_id=visitor['track_id'], ownership_known=False,
                                                   operation_completion_known=False))
                        break
            events.append(event)
    cash_count = sum(event['rule_id'] == 'cash_after_client' for event in events)
    docs_count = sum(event['rule_id'] == 'previous_client_documents' for event in events)
    check('cash_stays_after_visitor', 'cash_after_client', 'Банкноты на одном месте до и после ухода посетителя',
          departure_available and cash_detector,
          'Сравниваются непрерывные наблюдения одних и тех же предполагаемых банкнот до и после наблюдаемого ухода посетителя. Пропуски, спорный класс и оставшаяся группа людей разрывают проверку.', cash_count,
          role_reason or (None if cash_detector else 'cash_detector_unavailable'), ['Завершение операции', 'Принадлежность денег'])
    check('documents_stay_after_visitor', 'previous_client_documents', 'Документ на одном месте до и после ухода посетителя',
          departure_available and document_detector,
          'Сравниваются непрерывные наблюдения документа на том же месте после ухода посетителя; владелец и содержание документа не определяются.', docs_count,
          role_reason or (None if document_detector else 'document_detector_unavailable'), ['Принадлежность документа', 'Содержание документа'])
    check('document_between_visits', 'previous_client_documents', 'Документ остался до прихода следующего посетителя',
          departure_available and document_detector,
          'Для двух раздельных наблюдаемых посещений проверяется, продолжает ли тот же кандидат документа лежать на прежнем месте. Одновременная группа не считается последовательностью клиентов.',
          sum(row['kind'] == 'document_persists_into_next_visit' for row in sequences),
          role_reason or (None if document_detector else 'document_detector_unavailable'), ['Принадлежность документа', 'Завершение банковской операции'])

    cashier = role.get('cashier_track_id')
    for departure in service.get('cashier_departures', []) if roles_ok else []:
        start, end = departure.get('start_sec'), departure.get('end_sec')
        if (departure.get('role_status') != 'candidate' or departure.get('people_decode_complete') is not True
                or departure.get('no_visible_replacement') is not True or departure.get('cashier_track_id') != cashier
                or not _number(start) or not _number(end) or end - start < ABSENCE_SECONDS):
            continue
        track = people.get('by_id', {}).get(cashier, [])
        if not track or not _edge_exit(track, video['width'], video['height']):
            continue
        if start <= track[-1]['time'] or end > (frames - 1) / fps + 1e-6:
            continue
        if any(row['time'] >= start and row['time'] <= end for row in track):
            continue
        # No added asset requirement: the controlled visible work zone itself
        # can lack its assigned cashier while visitors remain.
        first, last = round(start * fps), min(frames - 1, round(end * fps))
        rows = [{**track[-1], 'label': 'person'},
                {'time': first / fps, 'frame': first, 'mode': 'people', 'box': table, 'label': 'work_table'},
                {'time': last / fps, 'frame': last, 'mode': 'people', 'box': table, 'label': 'work_table'}]
        events.append(_event('unattended_room', 'Кассир ушёл из наблюдаемой рабочей зоны', rows,
                             'Кандидат кассира двигался к краю и перестал наблюдаться; в проверяемой зоне не видна замена. Наличие посетителей не считается присутствием кассира. Проверьте фактический присмотр.',
                             {'cashier_track_id': cashier, 'people_decode_complete': True,
                              'no_visible_replacement': True, 'scope': 'visible_cashier_work_zone',
                              'whole_room_visibility_confirmed': False, 'asset_presence_required': False}))
    check('cashier_work_zone_empty', 'unattended_room', 'Уход кассира без видимой замены в рабочей зоне',
          departure_available, 'Проверяется наблюдаемая рабочая зона, независимо от наличия денег или пакета. Люди за пределами камеры и реальный присмотр неизвестны.',
          sum(event['rule_id'] == 'unattended_room' for event in events), role_reason, ['Весь кабинет вне поля зрения', 'Фактический присмотр'])

    for chain in object_sequences:
        rows = chain['rows']
        if len(rows) < 4 or not roles_ok:
            continue
        # Moving object -> stationary position in the table projection.
        if chain['kind'] == 'cash':
            for split in range(1, len(rows) - 2):
                before, settled = rows[:split], rows[split:]
                visitor_rows = [row for row in before if row['near_role'] == 'visitor']
                if (_long(settled) and visitor_rows and all(row['over_table'] and not row['at_counter'] for row in settled)
                        and _stationary(settled) and _distance(visitor_rows[0]['box'], settled[0]['box']) >=
                        .5 * math.sqrt(_area(settled[0]['box']))):
                    sequences.append(_sequence('cash_settled_over_table', 'Банкноты переместились от посетителя к устойчивому месту над столом',
                                               [visitor_rows[0], settled[0], settled[-1]],
                                               visitor_track_id=visitor_rows[0]['near_track_id'], surface_contact_confirmed=False,
                                               hand_contact_confirmed=False, placement_confirmed=False))
                    break
            # Explicit outside -> apparatus projection -> outside chain.
            contact = [i for i, row in enumerate(rows) if row['at_counter']]
            if (contact and contact[0] > 0 and contact[-1] < len(rows) - 1 and len(contact) >= 2
                    and all(rows[i]['at_counter'] for i in range(contact[0], contact[-1] + 1))
                    and all(row['counter_visible'] for row in rows[contact[0] - 1:contact[-1] + 2])):
                path = rows[contact[0] - 1:contact[-1] + 2]
                if _distance(path[0]['box'], path[1]['box']) + _distance(path[-2]['box'], path[-1]['box']) > .3 * math.sqrt(_area(path[1]['box'])):
                    sequences.append(_sequence('cash_counter_approach_leave', 'Банкноты приблизились к счётчику и удалились от него',
                                               [path[0], path[1], path[-2], path[-1]], machine_running_confirmed=False,
                                               notes_passed_through_confirmed=False, counting_confirmed=False,
                                               client_line_of_sight_confirmed=False))
        # Directional projected handoff. Require sustained endpoint association
        # and actual object displacement; moving people around stationary cash
        # must never count as a handoff.
        start_role = 'cashier' if chain['kind'] == 'cash' else 'visitor' if chain['kind'] == 'card' else None
        end_role = 'visitor' if chain['kind'] == 'cash' else 'cashier'
        if start_role is None:
            continue
        starts = [row for row in rows if row['near_role'] == start_role]
        ends = [row for row in rows if row['near_role'] == end_role]
        if len(starts) < 2 or len(ends) < 2 or starts[1]['time'] >= ends[-2]['time']:
            continue
        initial = [row for row in starts if row['time'] < ends[-2]['time']]
        final = [row for row in ends if row['time'] > initial[-1]['time']] if initial else []
        if len(initial) < 2 or len(final) < 2 or initial[-1]['near_track_id'] == final[0]['near_track_id']:
            continue
        if _distance(initial[0]['box'], final[-1]['box']) < max(10., math.sqrt(_area(initial[0]['box']))):
            continue
        path = [row for row in rows if initial[0]['time'] <= row['time'] <= final[-1]['time']]
        role_changes = []
        for row in path:
            if row['near_role'] != 'unknown' and (not role_changes or role_changes[-1] != row['near_role']):
                role_changes.append(row['near_role'])
        if role_changes != [start_role, end_role]:
            continue
        if any(row['near_track_id'] not in {None, initial[0]['near_track_id'], final[-1]['near_track_id']} for row in path):
            continue
        kind = 'cash_cashier_to_visitor' if chain['kind'] == 'cash' else 'card_visitor_to_cashier'
        sequences.append(_sequence(kind, 'Банкноты переместились от кассира к посетителю' if chain['kind'] == 'cash' else 'Карточка переместилась от посетителя к кассиру',
                                   [initial[0], initial[-1], final[0], final[-1]],
                                   from_track_id=initial[0]['near_track_id'], to_track_id=final[-1]['near_track_id'],
                                   handoff_confirmed=False, return_semantics_known=False,
                                   original_id_confirmed=False, ownership_known=False))
        if chain['kind'] == 'cash' and all(row['counter_visible'] and not row['at_counter'] for row in path):
            events.append(_event('change_through_counter', 'Проверить движение банкнот от кассира к посетителю', path,
                                 'Похожая рамка банкнот непрерывно перемещается от проекции кассира к проекции посетителя. Счётчик виден, но в этой цепочке приближения к нему не наблюдалось. Это повод проверить фрагмент: возврат, остаток и пересчёт до начала цепочки неизвестны.',
                                 {'from_track_id': initial[0]['near_track_id'], 'to_track_id': final[-1]['near_track_id'],
                                  'counter_observed_throughout_chain': True, 'counter_approach_observed_in_chain': False,
                                  'counting_confirmed': False, 'return_semantics_known': False, 'handoff_confirmed': False}))

    sequence_counts = defaultdict(int)
    for row in sequences:
        sequence_counts[row['kind']] += 1
    cash_actions = bool(roles_ok and stable_table_seen and cash_detector)
    counter_seen = sum(any(obj.get('label') == 'banknote_counter' for obj in objects) for objects in context.values()) >= 3
    check('cash_move_and_settle', 'cash_placed_on_counter', 'Банкноты переместились от посетителя и остановились над столом',
          cash_actions, 'Ищется движение рамки банкнот с последующим устойчивым положением в проекции стола. Физическое касание поверхности не устанавливается.',
          sequence_counts['cash_settled_over_table'], role_reason or (None if cash_detector else 'cash_detector_unavailable'), ['Контакт со столом', 'Специальный лоток'])
    check('cash_approach_counter_and_leave', 'cash_counted_visible', 'Банкноты подошли к счётчику и вышли из его проекции',
          cash_actions and counter_seen, 'Проверяется непрерывная цепочка вне счётчика → у счётчика → вне счётчика. Работа аппарата, прохождение пачки и видимость клиенту остаются отдельными неподдерживаемыми частями правила.',
          sequence_counts['cash_counter_approach_leave'], role_reason or ('counter_not_visible' if not counter_seen else None),
          ['Работа аппарата', 'Прохождение пачки через аппарат', 'Линия зрения клиента'])
    check('cash_cashier_to_visitor', 'change_through_counter', 'Движение банкнот от кассира к посетителю и подход к счётчику',
          cash_actions and counter_seen, 'Сопоставляются направление движения предмета и видимое приближение к аппарату. Выдача, остаток, возврат и ранее выполненный пересчёт не различаются.',
          sequence_counts['cash_cashier_to_visitor'], role_reason or ('counter_not_visible' if not counter_seen else None),
          ['Смысл операции возврата', 'Пересчёт до наблюдаемой цепочки'])
    check('card_visitor_to_cashier', 'original_id_verification', 'Карточка переместилась от посетителя к кассиру',
          roles_ok and document_detector, 'Ищется последовательное перемещение предполагаемой карточки между проекциями посетителя и кассира. Это наблюдение предъявления предмета; тип карточки и проверка оригинала неизвестны.',
          sequence_counts['card_visitor_to_cashier'], role_reason or (None if document_detector else 'card_detector_unavailable'),
          ['Тип карточки', 'Подлинность оригинала', 'Принадлежность', 'Результат верификации'])

    for rule_id, title in TITLES.items():
        relevant = [row for row in checks if row['rule_id'] == rule_id]
        available = any(row['available'] for row in relevant if row['check_id'] != 'visitor_edge_exit')
        count = sum(row['rule_id'] == rule_id for row in events)
        detail = ' '.join(row['detail'] for row in relevant if row['check_id'] != 'visitor_edge_exit')
        if not available:
            reason_codes = sorted({row['reason_code'] for row in relevant if not row['available']})
            detail = 'На этой записи не хватает наблюдений для проверки. ' + detail
        else:
            reason_codes = ['partial_observable_scope']
            detail += f' Найдено кандидатов для просмотра: {count}.'
        result['coverage'].append({'rule_id': rule_id, 'title': title, 'status': 'partial' if available else 'not_checked',
                                   'implementation_status': 'implemented_partial',
                                   'availability': 'available' if available else 'insufficient_evidence',
                                   'reason_code': reason_codes[0] if len(reason_codes) == 1 else 'multiple_missing_inputs',
                                   'reason_codes': reason_codes, 'detail': detail, 'reason': detail,
                                   'event_count': count, 'events_count': count,
                                   'observed_checks': [row['check_id'] for row in relevant if row['available']],
                                   'unsupported': sorted({item for row in relevant for item in row.get('unsupported', [])}),
                                   'source': 'observable_actions_v2', 'rule_source': 'Правило пользователя / кейса',
                                   'limitations': list(COMMON)})
    result['status'] = 'partial' if any(row['available'] for row in checks) else 'not_checked'
    result['data_quality'] = {'people_usable': people_ok, 'roles_usable': roles_ok, 'issues': issues,
                              'object_observation_counts': counts, 'excluded_reflection_track_ids': sorted(excluded),
                              'ambiguous_reflection_track_ids': sorted(ambiguous),
                              'object_association': 'unique_same_class_adjacent_sample_proximity',
                              'counting_action_model': False, 'hand_contact_model': False}
    events.sort(key=lambda row: (row['start_sec'], row['rule_id']))
    sequences.sort(key=lambda row: (row['start_sec'], row['kind']))
    return result
