"""Position-based service observations, not identities or completed operations.

Only consumes the validated journals loaded by rule_engine. No face recognition,
fixed camera coordinates, legal rules, network requests or model inference.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import hashlib
import math
from statistics import median

RULE_SOURCE = 'Правило пользователя / кейса'
USER_RULE = ('По уточнению пользователя кассир находится со стороны сейфа; '
             'либо один человек устойчиво находится за рабочим столом у ноутбука. '
             'Направление экрана и принтера требует отдельного подтверждения.')
STEP_SEC = .25
ROLE_SECONDS = 3.
WORKSTATION_ALONE_SECONDS = 2.
VISIT_SECONDS = 2.
MIN_SCORE = .4
MAX_ANCHOR_GAP_SEC = 1.5
COMMON = [
    'Позиционная роль — предположение по рабочему месту, а не подтверждённая должность или личность.',
    'Исключаются только устойчивые кандидаты отражения в сопоставленной проверенной области; остальные неоднозначные люди сохраняются.',
    'Перекрытия, смена номера трекера и люди вне кадра могут менять видимую картину.',
    'Начало и конец видимой траектории не устанавливают начало и завершение банковской операции.',
]


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value):
        return None
    return list(map(float, value)) if value[2] > value[0] and value[3] > value[1] else None


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    overlap = _intersection(a, b)
    return overlap / max(1e-9, _area(a) + _area(b) - overlap)


def _near_table(box, table, person=False):
    if _intersection(box, table) / min(_area(box), _area(table)) >= .1:
        return True
    if person:
        # A standing customer's torso can be far above the tabletop in the
        # image while the lower edge of their box is next to it. Use the gap
        # between extents, with a narrow border around the table; a distant
        # reflected person high in the glass must not gain a service role.
        dx = max(table[0] - box[2], box[0] - table[2], 0.)
        dy = max(table[1] - box[3], box[1] - table[3], 0.)
        margin = max(4., .1 * min(table[2] - table[0], table[3] - table[1]))
        return math.hypot(dx, dy) <= margin
    x, y = _center(box)
    distance = math.hypot(max(table[0] - x, 0, x - table[2]), max(table[1] - y, 0, y - table[3]))
    return distance < .2 * math.hypot(table[2] - table[0], table[3] - table[1])


def _asset_unconflicted(obj, timestamp, modes):
    if obj['label'] != 'cash':
        return True
    for data in modes.values():
        records = data.get('records', [])
        times = data.get('times', [row['time'] for row in records])
        index = bisect_left(times, timestamp)
        for i in (index - 1, index):
            if not 0 <= i < len(records) or abs(records[i]['time'] - timestamp) > .2:
                continue
            for other in records[i].get('objects', []):
                if other.get('score', 0) < .3 or not _box(other.get('box')):
                    continue
                if (other.get('label') in {'safe', 'banknote_counter', 'laptop', 'computer_monitor', 'desktop_computer'}
                        and _intersection(obj['box'], other['box']) / _area(obj['box']) > .7):
                    return False
                if other.get('label') == 'bag' and _intersection(obj['box'], other['box']) / _area(obj['box']) >= .5:
                    return False
    return True


def _stable_anchor(modes, label):
    observations = []
    for data in modes.values():
        for record in data.get('records', []):
            objects = [obj for obj in record.get('objects', [])
                       if obj.get('label') == label and obj.get('score', 0) >= MIN_SCORE and _box(obj.get('box'))]
            # Multiple separate safes/tables need explicit scene calibration.
            if len(objects) > 1 and any(_iou(objects[0]['box'], obj['box']) < .65 for obj in objects[1:]):
                return None, f'Несколько различных объектов «{label}»; сторона обслуживания неоднозначна.'
            if objects:
                observations.append(dict(time=record['time'], box=objects[0]['box']))
    observations.sort(key=lambda row: row['time'])
    if len({row['time'] for row in observations}) < 3 or observations[-1]['time'] - observations[0]['time'] < 2:
        return None, f'Нет устойчивых наблюдений «{label}» минимум на протяжении 2 секунд.'
    box = [median(row['box'][i] for row in observations) for i in range(4)]
    stable = [row for row in observations if _iou(row['box'], box) >= .6]
    if len(stable) / len(observations) < .85:
        return None, f'Положение «{label}» меняется или детекции противоречат друг другу.'
    return {'box': box, 'times': sorted({row['time'] for row in stable}),
            'observations': stable, 'rejected_times': [row['time'] for row in observations if _iou(row['box'], box) < .6],
            'tolerance': .3}, None


def _direct_visible(anchor, timestamp):
    if any(abs(time - timestamp) <= .3 for time in anchor.get('rejected_times', [])):
        return False
    times = anchor['times']
    index = bisect_left(times, timestamp)
    return any(abs(times[i] - timestamp) <= anchor['tolerance'] + 1e-6
               for i in (index - 1, index) if 0 <= i < len(times))


def _visible(anchor, timestamp, peer=None):
    """Bounded interpolation of a stationary, temporarily hidden anchor.

    Both endpoints must agree and a second physical anchor must remain directly
    observed. Do not extrapolate a missing table through the entire recording or
    use two simultaneously interpolated anchors as a scene-stability check.
    """
    if _direct_visible(anchor, timestamp):
        return True
    if peer is None or not _direct_visible(peer, timestamp):
        return False
    times = anchor['times']
    index = bisect_left(times, timestamp)
    if not 0 < index < len(times):
        return False
    start, end = times[index - 1:index + 1]
    if end - start > MAX_ANCHOR_GAP_SEC + 1e-6:
        return False
    if any(start <= time <= end for time in anchor.get('rejected_times', [])):
        return False
    endpoints = [row for row in anchor['observations'] if row['time'] in {start, end}]
    if len(endpoints) != 2 or _iou(endpoints[0]['box'], endpoints[-1]['box']) < .75:
        return False
    scene = [row for row in peer['observations'] if start - .3 <= row['time'] <= end + .3]
    if (len(scene) < 2 or not _direct_visible(peer, start) or not _direct_visible(peer, end)
            or any(start <= time <= end for time in peer.get('rejected_times', []))):
        return False
    return all(_iou(row['box'], scene[0]['box']) >= .75 for row in scene)


def _geometry(safe, table):
    sx, sy = _center(safe['box'])
    tx, ty = _center(table['box'])
    w, h = table['box'][2] - table['box'][0], table['box'][3] - table['box'][1]
    dx, dy = (sx - tx) / w, (sy - ty) / h
    axis = 0 if abs(dx) >= abs(dy) else 1
    primary, secondary = (dx, dy) if axis == 0 else (dy, dx)
    if abs(primary) < .2 or abs(primary) < abs(secondary) * 1.5:
        return None
    return {'axis': axis, 'sign': 1 if primary > 0 else -1,
            'origin': (tx, ty)[axis], 'margin': (w, h)[axis] * .12}


def _side(box, geometry, table):
    if not _near_table(box, table, person=True):
        return 'away'
    # A torso point is less affected by clipped feet than the bottom of a box.
    point = ((box[0] + box[2]) / 2, box[1] + .55 * (box[3] - box[1]))
    displacement = (point[geometry['axis']] - geometry['origin']) * geometry['sign']
    if abs(displacement) <= geometry['margin']:
        return 'uncertain'
    return 'cashier_side' if displacement > 0 else 'visitor_side'


def _workstation_laptop(modes, table):
    if not table:
        return None, 'Нет устойчивого рабочего стола.'
    # The small POS display has previously been labelled laptop. Restrict the
    # fallback anchor to candidates actually projected into the tabletop.
    filtered = {}
    for mode, data in modes.items():
        records = []
        for record in data.get('records', []):
            objects = []
            for obj in record.get('objects', []):
                if obj.get('label') != 'laptop' or not _box(obj.get('box')):
                    continue
                x, y = _center(obj['box'])
                if (table['box'][0] <= x <= table['box'][2] and table['box'][1] <= y <= table['box'][3]
                        and _intersection(obj['box'], table['box']) / _area(obj['box']) >= .4):
                    objects.append(obj)
            records.append({**record, 'objects': objects})
        filtered[mode] = {**data, 'records': records}
    return _stable_anchor(filtered, 'laptop')


def _workstation_candidate(table, laptop, people, video, safe=None):
    """A lone stable workstation occupant can anchor an ambiguous safe side.

    Bboxes do not prove sitting or screen orientation. This fallback exposes
    those missing checks explicitly and never promotes someone merely because
    they are the only person anywhere in the room.
    """
    if not table or not laptop or not _near_table(laptop['box'], table['box']):
        return None
    fps, frames = video['fps'], video['frames']
    peer = safe or laptop
    stride = max(1, round(fps * STEP_SEC))
    rows = defaultdict(list)
    for frame in range(0, int(frames), stride):
        timestamp = frame / fps
        if not (_visible(table, timestamp, peer) and _visible(peer, timestamp, table)):
            continue
        near = [r for r in people.get('by_frame', {}).get(frame, [])
                if _box(r.get('box')) and _near_table(r['box'], table['box'], person=True)]
        # Companions near the same desk prevent a lone-occupant inference.
        if len(near) != 1:
            continue
        person = near[0]
        box = person['box']
        point = ((box[0] + box[2]) / 2, box[1] + .55 * (box[3] - box[1]))
        lx, ly = _center(laptop['box'])
        tw, th = table['box'][2] - table['box'][0], table['box'][3] - table['box'][1]
        if box[3] - box[1] > video['height'] * .55 or math.hypot(point[0] - lx, point[1] - ly) > .8 * math.hypot(tw, th):
            continue
        rows[person['id']].append({**person, 'time': timestamp})
    qualified = []
    for identity, observations in rows.items():
        # A later customer left alone must not silently become the cashier.
        # This fallback establishes the initial workstation occupant only.
        full_track = people.get('by_id', {}).get(identity, [])
        if not full_track or full_track[0]['time'] > .5:
            continue
        for run in _runs(observations, lambda r: True, stride / fps * 1.6):
            if run[0]['time'] > 1.:
                continue
            if not _long(run, WORKSTATION_ALONE_SECONDS):
                continue
            if not any(_direct_visible(laptop, row['time']) for row in run):
                continue
            centers = [_center(r['box']) for r in run]
            movement = math.hypot(max(p[0] for p in centers) - min(p[0] for p in centers),
                                  max(p[1] for p in centers) - min(p[1] for p in centers))
            if movement > .25 * math.hypot(tw, th):
                continue
            torso = [median((r['box'][0] + r['box'][2]) / 2 for r in run),
                     median(r['box'][1] + .55 * (r['box'][3] - r['box'][1]) for r in run)]
            tx, ty = _center(table['box'])
            displacement = [(torso[0] - tx) / tw, (torso[1] - ty) / th]
            axis = max(range(2), key=lambda i: abs(displacement[i]))
            if abs(displacement[axis]) < .08 or abs(displacement[axis]) < abs(displacement[1 - axis]) * 1.5:
                continue
            geometry = {'axis': axis, 'sign': 1 if displacement[axis] > 0 else -1,
                        'origin': (tx, ty)[axis], 'margin': (tw, th)[axis] * .05}
            qualified.append((run[0]['time'], identity, geometry, run))
            break
    qualified.sort(key=lambda item: (item[0], item[1]))
    if not qualified or any(row[0] <= qualified[0][0] + WORKSTATION_ALONE_SECONDS and row[1] != qualified[0][1]
                            for row in qualified[1:]):
        return None
    return {'track_id': qualified[0][1], 'geometry': qualified[0][2], 'evidence': qualified[0][3]}


def _runs(rows, predicate, max_gap=STEP_SEC * 1.6):
    result, current = [], []
    for row in rows:
        if predicate(row):
            if current and row['time'] - current[-1]['time'] > max_gap + 1e-6:
                result.append(current)
                current = []
            current.append(row)
        elif current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def _long(run, seconds):
    return len(run) >= 3 and run[-1]['time'] - run[0]['time'] >= seconds - 1e-6


def _edge_transition(track, width, height, entering=False):
    if len(track) < 3:
        return False
    origin = track[0] if entering else track[-1]
    tail = [row for row in track if abs(row['time'] - origin['time']) <= 1.]
    if len(tail) < 3 or tail[-1]['time'] - tail[0]['time'] < .3:
        return False
    box = origin['box']
    distances = [box[0], box[1], width - box[2], height - box[3]]
    edge = min(range(4), key=lambda i: distances[i])
    if distances[edge] > max(8., .03 * min(width, height)):
        return False
    a, b = _center(tail[0]['box']), _center(tail[-1]['box'])
    outward = [a[0] - b[0], a[1] - b[1], b[0] - a[0], b[1] - a[1]][edge]
    return (-outward if entering else outward) >= max(8., .02 * min(width, height))


def _reflection_pairs(tracks, width, height):
    """Flag strong motion symmetry only; stationary people are not reflections.

    This does not remove any observation or establish that glass is present.
    A flagged pair blocks role-dependent events until manual scene review.
    """
    result = []
    ids = list(tracks)
    for index, identity in enumerate(ids):
        a = {row['frame']: row for row in tracks[identity]}
        for other in ids[index + 1:]:
            shared = [(a[row['frame']], row) for row in tracks[other] if row['frame'] in a]
            if len(shared) < 8 or shared[-1][0]['time'] - shared[0][0]['time'] < 2.:
                continue
            if sum(_iou(x['box'], y['box']) > .65 for x, y in shared) / len(shared) > .7:
                result.append([identity, other])
                continue
            ac = [_center(x['box']) for x, _ in shared]
            bc = [_center(y['box']) for _, y in shared]
            movement = max(p[0] for p in ac) - min(p[0] for p in ac)
            if movement < max(20., width * .04):
                continue
            sums = [x[0] + y[0] for x, y in zip(ac, bc)]
            ydiff = [x[1] - y[1] for x, y in zip(ac, bc)]
            if (max(sums) - min(sums) < max(8., movement * .15)
                    and max(ydiff) - min(ydiff) < max(8., height * .02)):
                result.append([identity, other])
    return result


def _evidence(row, label='person'):
    return {'timestamp_sec': row['time'], 'source_frame': row['frame'],
            'mode': row.get('mode', 'people'), 'playback_sec': row.get('playback_sec', row['time']),
            'box': row['box'], 'labels': [row.get('label', label)]}


def _event(rule_id, title, observations, description, context, limitations=()):
    rows = sorted(observations, key=lambda row: row['time'])
    first, last = rows[0], rows[-1]
    token = f"service:{rule_id}:{first['frame']}:{last['frame']}:{context}"
    labels = sorted({row.get('label', 'person') for row in rows})
    return {'id': hashlib.sha256(token.encode()).hexdigest()[:16], 'rule_id': rule_id,
            'title': title, 'start_sec': first['time'], 'end_sec': last['time'],
            'description': description, 'evidence_text': f"{first['time']:.2f}–{last['time']:.2f} с: позиционные наблюдения для проверки.",
            'limitations': COMMON + list(limitations), 'mode': first.get('mode', 'people'),
            'playback_sec': first.get('playback_sec', first['time']), 'source_frame': first['frame'],
            'labels': labels, 'evidence_box': first['box'], 'evidence_times': [_evidence(row) for row in rows],
            'context': {'user_rule': USER_RULE, 'role_source': 'position_candidate', **context},
            'severity': 'review', 'status': 'review', 'confidence_label': 'Требует проверки', 'rule_source': RULE_SOURCE}


def _coverage(rule_id, title, status, detail, events):
    count = sum(event['rule_id'] == rule_id for event in events)
    return {'rule_id': rule_id, 'title': title, 'status': status, 'detail': detail, 'reason': detail,
            'event_count': count, 'events_count': count, 'rule_source': RULE_SOURCE,
            'limitations': list(COMMON), 'source': 'service_position_analysis'}


def analyze_service(modes: dict, people: dict, job: dict) -> dict:
    """Return role candidates, visible visits, compatible review events/coverage.

    `modes` and `people` have the shapes returned by rule_engine._load_modes and
    _load_people. The caller must pass the original job.video metadata. All time
    values refer to source-video seconds; operation boundaries remain unknown.
    """
    events, sessions, cashier_departures = [], [], []
    role = {'status': 'unknown', 'cashier_track_id': None, 'visitor_track_ids': [],
            'user_rule': USER_RULE, 'printer_orientation_checked': False,
            'laptop_screen_orientation_checked': False, 'seated_posture_checked': False,
            'coordinate_calibration_confirmed': False, 'anchors': {}, 'reflection_candidate_pairs': [],
            'excluded_reflection_track_ids': list(people.get('excluded_reflection_track_ids', [])),
            'role_method': None}
    result = {'version': '1.2', 'status': 'not_checked', 'role_summary': role,
              'sessions': sessions, 'cashier_departures': cashier_departures,
              'events': events, 'coverage': [], 'limitations': list(COMMON),
              'settings': {'sample_step_sec': STEP_SEC, 'minimum_role_span_sec': ROLE_SECONDS,
                           'minimum_visit_span_sec': VISIT_SECONDS,
                           'minimum_lone_workstation_span_sec': WORKSTATION_ALONE_SECONDS,
                           'workstation_fallback_initial_occupant_only': True,
                           'maximum_anchor_interpolation_sec': MAX_ANCHOR_GAP_SEC,
                           'note': 'Пороги устойчивости прототипа, не нормы и не оценка качества.'}}

    def finish(reason, available=False):
        result['status'] = 'partial' if available else 'not_checked'
        role['detail'] = reason
        status = result['status']
        result['coverage'] = [
            _coverage('one_client_at_a_time', 'Предыдущий клиент обслужен до следующего', status,
                      reason + ' Оцениваются только пересечения видимого присутствия; конец операции неизвестен.', events),
            _coverage('restricted_access_card_transfer', 'Посторонний у кассира и передача чужой карты', status,
                      reason + ' Дополнительный человек со стороны сейфа требует проверки допуска; передача и принадлежность карты не устанавливаются.', events),
            _coverage('unattended_assets', 'Деньги, пакеты и компьютер под присмотром', status,
                      reason + ' Уход кандидата кассира проверяется отдельно от присутствия посетителей; фактический присмотр неизвестен.', events),
            _coverage('previous_client_documents', 'На столе нет документов прошлого клиента', 'not_checked',
                      'Видимые посещения не устанавливают принадлежность документов и конец предыдущей операции.', events),
        ]
        events.sort(key=lambda event: (event['start_sec'], event['rule_id']))
        return result

    if not people.get('complete'):
        return finish('Нет полного проверенного журнала людей: ' + people.get('reason', 'причина неизвестна'))
    video = job.get('video', {})
    width, height, fps, frames = (video.get(key, 0) for key in ('width', 'height', 'fps', 'frames'))
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (width, height, fps, frames)):
        return finish('Нет корректной геометрии и длительности исходного видео.')
    if people.get('frames') != frames or abs(people.get('fps', 0) - fps) > .01:
        return finish('Границы журнала людей не совпадают с исходным видео.')
    safe, safe_error = _stable_anchor(modes, 'safe')
    table, table_error = _stable_anchor(modes, 'work_table')
    laptop, laptop_error = _workstation_laptop(modes, table)
    if not table:
        return finish(' '.join(reason for reason in (safe_error, table_error) if reason))
    geometry = _geometry(safe, table) if safe else None
    workstation = _workstation_candidate(table, laptop, people, video, safe)
    use_workstation = workstation and (not geometry or
        sum(_side(r['box'], geometry, table['box']) == 'cashier_side' for r in workstation['evidence'])
        / len(workstation['evidence']) < .7)
    anchor = (safe or laptop) if use_workstation else safe
    if use_workstation:
        geometry = workstation['geometry']
        role['role_method'] = 'lone_stable_workstation_occupant'
    else:
        role['role_method'] = 'safe_side_position' if geometry else None
    role['anchors'] = {'table_box': table['box'], 'source': 'stable_model_candidates'}
    if safe:
        role['anchors']['safe_box'] = safe['box']
    if laptop:
        role['anchors']['laptop_box'] = laptop['box']
    if not geometry:
        return finish('Сторона сейфа неоднозначна; устойчивый единственный человек у стола и ноутбука тоже не выделен. '
                      + (safe_error or '') + ' ' + (laptop_error or ''))
    role['anchors']['projected_side'] = ('right' if geometry['sign'] > 0 else 'left') if geometry['axis'] == 0 else ('bottom' if geometry['sign'] > 0 else 'top')
    stride = max(1, round(fps * STEP_SEC))
    sampled, by_id = {}, defaultdict(list)
    for frame in range(0, int(frames), stride):
        timestamp = frame / fps
        valid = _visible(anchor, timestamp, table) and _visible(table, timestamp, anchor)
        rows = []
        for raw in people.get('by_frame', {}).get(frame, []):
            if not _box(raw.get('box')):
                continue
            row = {**raw, 'time': timestamp, 'side': _side(raw['box'], geometry, table['box']) if valid else 'unknown'}
            rows.append(row)
            by_id[row['id']].append(row)
        sampled[frame] = {'time': timestamp, 'rows': rows, 'anchors_visible': valid}
    pairs = _reflection_pairs(by_id, width, height)
    role['reflection_candidate_pairs'] = pairs
    suspicious = {identity for pair in pairs for identity in pair}
    candidates, ranking = [], []
    for identity, rows in by_id.items():
        meaningful = [row for row in rows if row['side'] in {'cashier_side', 'visitor_side'}]
        side_runs = _runs(rows, lambda row: row['side'] == 'cashier_side', stride / fps * 1.6)
        eligible = [run for run in side_runs if _long(run, ROLE_SECONDS)]
        initial_side = meaningful[0]['side'] if meaningful else 'unknown'
        ranking.append({'track_id': identity, 'sample_counts': {side: sum(row['side'] == side for row in rows)
                         for side in ('cashier_side', 'visitor_side', 'uncertain', 'away', 'unknown')},
                        'longest_cashier_span_sec': max((run[-1]['time'] - run[0]['time'] for run in side_runs), default=0.),
                        'initial_observed_side': initial_side, 'reflection_ambiguous': identity in suspicious})
        if initial_side != 'cashier_side' or identity in suspicious:
            continue
        if eligible:
            candidates.append((eligible[0][0]['time'], identity, eligible[0]))
    role['track_ranking'] = sorted(ranking, key=lambda row: (-row['longest_cashier_span_sec'], -row['sample_counts']['cashier_side'], row['track_id']))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if not candidates:
        return finish('Нет единственной устойчивой траектории со стороны сейфа; возможны перекрытия или отражения.')
    earliest, cashier, enrollment = candidates[0]
    if any(start <= earliest + ROLE_SECONDS for start, _, _ in candidates[1:]):
        return finish('Несколько людей устойчиво находятся со стороны сейфа; ответственный кассир не определён.')
    # Reflected/duplicated observations touching the selected cashier block roles.
    if cashier in suspicious:
        return finish('Траектория кандидата кассира имеет признаки дублирования или зеркального движения.')
    role.update(status='candidate', cashier_track_id=cashier,
                evidence_times=[_evidence(enrollment[0]), _evidence(enrollment[-1])])
    side_name = 'рабочего места кассира' if use_workstation else 'сейфа'
    if use_workstation:
        role['workstation_evidence'] = [_evidence(workstation['evidence'][0]), _evidence(workstation['evidence'][-1])]
    visit_runs = {}
    for identity, rows in by_id.items():
        if identity == cashier:
            continue
        runs = _runs(rows, lambda row: row['side'] == 'visitor_side', stride / fps * 1.6)
        visit_runs[identity] = runs
        for run in runs:
            if not _long(run, .5):
                continue
            full = people.get('by_id', {}).get(identity, rows)
            entry = (full[0]['time'] > .25 and _edge_transition(full, width, height, entering=True)
                     and 0 <= run[0]['time'] - full[0]['time'] <= 5.)
            sessions.append({'track_id': identity, 'role': 'visitor_candidate',
                             'start_sec': run[0]['time'], 'end_sec': run[-1]['time'],
                             'duration_sec': run[-1]['time'] - run[0]['time'],
                             'sustained': _long(run, VISIT_SECONDS), 'entry_motion_observed': entry,
                             'reflection_ambiguous': identity in suspicious,
                             'operation_start_sec': None, 'operation_end_sec': None,
                             'end_reason': 'Конец наблюдаемого присутствия; завершение операции неизвестно.',
                             'evidence_times': [_evidence(run[0]), _evidence(run[-1])]})
    role['visitor_track_ids'] = sorted({session['track_id'] for session in sessions})
    # A simultaneous group already in view is not a second-client arrival.
    for incoming in sessions:
        if not incoming['sustained'] or not incoming['entry_motion_observed'] or incoming['reflection_ambiguous']:
            continue
        for prior in sessions:
            if (prior['track_id'] == incoming['track_id'] or not prior['sustained'] or prior['reflection_ambiguous']
                    or prior['start_sec'] + ROLE_SECONDS > incoming['start_sec']):
                continue
            start, end = incoming['start_sec'], min(incoming['end_sec'], prior['end_sec'])
            if end - start < VISIT_SECONDS:
                continue
            cashier_runs = [run for run in _runs(by_id[cashier], lambda row: start <= row['time'] <= end and row['side'] == 'cashier_side', stride / fps * 1.6)
                            if _long(run, VISIT_SECONDS)]
            if not cashier_runs:
                continue
            observations = cashier_runs[0].copy()
            observations += [row for identity in (prior['track_id'], incoming['track_id'])
                             for row in by_id[identity] if start <= row['time'] <= end and row['side'] == 'visitor_side']
            events.append(_event('one_client_at_a_time', 'Проверить приход ещё одного посетителя', observations,
                                 'Видно движение новой траектории от края кадра к столу, пока предыдущая остаётся у стола и виден кандидат кассира. Проверьте порядок обслуживания: это может быть сопровождающий или совместная операция.',
                                 {'cashier_track_id': cashier, 'visitor_track_ids': [prior['track_id'], incoming['track_id']],
                                  'operation_completion_known': False}))
            break
    for identity, rows in by_id.items():
        if identity == cashier or identity in suspicious:
            continue
        full = people.get('by_id', {}).get(identity, rows)
        for run in _runs(rows, lambda row: row['side'] == 'cashier_side', stride / fps * 1.6):
            if not _long(run, VISIT_SECONDS) or run[0]['time'] < earliest + ROLE_SECONDS:
                continue
            prior_visit = any(_long(visit, VISIT_SECONDS) and visit[-1]['time'] < run[0]['time'] for visit in visit_runs.get(identity, []))
            entered = full[0]['time'] > earliest + ROLE_SECONDS and _edge_transition(full, width, height, entering=True)
            cash_runs = [observed for observed in _runs(by_id[cashier], lambda row: run[0]['time'] <= row['time'] <= run[-1]['time'] and row['side'] == 'cashier_side', stride / fps * 1.6)
                         if _long(observed, VISIT_SECONDS)]
            if not (prior_visit or entered) or not cash_runs:
                continue
            cash_rows = cash_runs[0]
            events.append(_event('restricted_access_card_transfer', f'Дополнительный человек со стороны {side_name}', run + cash_rows,
                                 f'Дополнительная траектория переместилась со стороны посетителей или от края кадра на сторону {side_name}, пока там остаётся кандидат кассира. Проверьте назначение зоны и наличие допуска.',
                                 {'cashier_track_id': cashier, 'additional_track_id': identity, 'card_transfer_checked': False},
                                 ['Допуск человека, родственные связи и принадлежность карточки неизвестны.']))
    # The responsible-side track can leave while a customer remains. Require an
    # observed outward edge movement, not disappearance behind furniture.
    full_cashier = people.get('by_id', {}).get(cashier, by_id[cashier])
    departure = full_cashier[-1]['time']
    if _edge_transition(full_cashier, width, height) and (frames - 1) / fps - departure >= VISIT_SECONDS:
        absence = []
        for sample in sampled.values():
            if sample['time'] <= departure:
                continue
            if (not sample['anchors_visible'] or any(row['side'] in {'cashier_side', 'uncertain', 'unknown'}
                                                      or row['id'] in suspicious for row in sample['rows'])):
                break
            absence.append(sample)
        if len(absence) >= 3 and absence[-1]['time'] - absence[0]['time'] >= VISIT_SECONDS:
            start, end = absence[0]['time'], absence[-1]['time']
            cashier_departures.append({
                'track_id': cashier, 'cashier_track_id': cashier, 'role_status': 'candidate',
                'exit_sec': departure, 'start_sec': start, 'end_sec': end,
                'exit_source_frame': full_cashier[-1]['frame'],
                'start_frame': round(start * fps), 'end_frame': round(end * fps),
                'people_decode_complete': True, 'no_visible_replacement': True,
                'visitors_remain': any(sample['rows'] for sample in absence),
                'operation_completion_known': False,
                'evidence_times': [_evidence(enrollment[0]), _evidence(enrollment[-1]), _evidence(full_cashier[-1])],
                'limitations': list(COMMON) + ['Движение к краю и исчезновение из кадра не доказывают выход из помещения.'],
                'user_rule': USER_RULE,
            })
            assets = defaultdict(list)
            for mode, data in modes.items():
                for record in data.get('records', []):
                    if not start <= record['time'] <= end:
                        continue
                    for obj in record.get('objects', []):
                        if (obj.get('label') in {'cash', 'bag', 'laptop', 'computer_monitor'}
                                and obj.get('score', 0) >= MIN_SCORE and _box(obj.get('box'))
                                and _near_table(obj['box'], table['box'])
                                and _asset_unconflicted(obj, record['time'], modes)):
                            assets[(mode, obj['label'])].append({**obj, 'time': record['time'], 'frame': record['frame'],
                                                              'mode': mode, 'playback_sec': record.get('playback_sec', record['time'])})
            stable = []
            for (mode, _), rows in assets.items():
                # Empty frames and changing object positions break evidence.
                interval = modes[mode].get('interval', .5)
                for run in _runs(rows, lambda row: True, interval * 1.6 + .05):
                    if _long(run, VISIT_SECONDS) and all(_iou(row['box'], run[0]['box']) >= .65 for row in run):
                        stable.extend(run)
            if stable:
                events.append(_event('unattended_assets', 'Предметы после возможного ухода кассира', stable,
                                     f'Кандидат кассира двигался к краю кадра и перестал наблюдаться. На стороне {side_name} замена не видна, а предметы у стола остаются; присутствие посетителя само по себе не считается присмотром кассира.',
                                     {'cashier_track_id': cashier, 'people_decode_complete': True,
                                      'departure_sec': departure, 'departure_evidence': _evidence(full_cashier[-1]),
                                      'visitors_remain': any(sample['rows'] for sample in absence)},
                                     ['Выход из кадра не доказывает выход из помещения; фактический присмотр за пределами камеры неизвестен.',
                                      'Пакет не подтверждает наличие денег. Положение у стола не доказывает контакт с его поверхностью.']))
                events[-1]['evidence_times'].append(_evidence(full_cashier[-1]))
    detail = ('Один человек устойчиво находится у рабочего стола и ноутбука; выделена позиционная траектория кандидата кассира. '
              'Поза сидя и направление экрана автоматически не подтверждены.' if use_workstation else
              'Сейф и стол устойчиво видны; выделена позиционная траектория кандидата кассира. Направление принтера не распознаётся.')
    if role['excluded_reflection_track_ids']:
        detail += ' Устойчивые кандидаты отражения исключены из подсчёта посетителей по проверенной области.'
    return finish(detail, True)
