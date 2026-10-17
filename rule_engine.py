"""Conservative temporal review candidates, never automatic violation verdicts.

Thresholds below suppress weak/isolated evidence. They are prototype analysis
settings, not legal or bank-policy limits; only the door's 3.5 seconds comes from
the user's rule. Primary JSONL/CSV outputs supply empty observations as well as
positive ones. Position-based roles are candidates, never personal identities.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from rule_settings import DOOR_OPEN_THRESHOLD_SEC, DOOR_OPEN_TITLE

REPORT_VERSION = '2.3'
MIN_SCORE = .4
FOOD_DRINK_MIN_SCORE = .4
FOOD_DRINK_MIN_SPAN = 1.0
MIN_OBSERVATIONS = 3
MIN_EVIDENCE_SPAN = 2.0
CLIP_END_MIN_ABSENCE = 1.0
CLIP_END_MIN_CASH_SPAN = 1.0
RULE_SOURCE = 'Правило пользователя / кейса'
NAMES = {'smartphone': 'смартфон', 'usb_drive': 'флешка', 'notebook': 'блокнот',
         'cash': 'банкноты', 'bag': 'мешок / сумка', 'laptop': 'ноутбук',
         'computer_monitor': 'монитор', 'food': 'еда / шоколадка', 'drink': 'чашка / напиток'}
EQUIPMENT = {'safe', 'laptop', 'banknote_counter', 'computer_monitor', 'desktop_computer'}
FORBIDDEN = {'smartphone', 'usb_drive', 'notebook'}
UNATTENDED = {'cash', 'bag', 'laptop', 'computer_monitor'}
FOOD_DRINK = {'food', 'drink'}


def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_number(x) for x in value):
        return None
    return list(map(float, value)) if value[2] > value[0] and value[3] > value[1] else None


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    intersection = _intersection(a, b)
    return intersection / max(1e-9, _area(a) + _area(b) - intersection)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _inside(box, container, fraction=.6):
    x, y = _center(box)
    return container[0] <= x <= container[2] and container[1] <= y <= container[3] and _intersection(box, container) / _area(box) >= fraction


def _label(label):
    return {'plastic_bag': 'bag', 'monitor': 'computer_monitor', 'desk': 'work_table', 'table': 'work_table',
            'chocolate': 'food', 'chocolate_bar': 'food', 'wrapped_chocolate': 'food', 'candy_bar': 'food',
            'snack': 'food', 'cup': 'drink', 'coffee_cup': 'drink', 'paper_cup': 'drink', 'mug': 'drink'}.get(label, label)


def _cash_food_conflict(cash, other):
    """Return a cash/food class ambiguity for similar object projections.

    Both model hypotheses must be withheld as temporal evidence. A source-bound
    human reference can resolve the object class, but a higher model score cannot.
    """
    return (other['label'] in FOOD_DRINK and other['score'] >= .28
            and (_intersection(cash['box'], other['box']) / _area(cash['box']) >= .5
                 or _iou(cash['box'], other['box']) >= .3))


def _load_modes(folder, job):
    loaded, reasons = {}, {}
    completed = {stage.get('key') for stage in job.get('stages', []) if stage.get('status') == 'completed'}
    for mode in ('trained', 'desk'):
        if mode not in completed:
            reasons[mode] = 'Этап распознавания не завершён или не запускался.'
            continue
        paths = list((folder / mode).glob('*/summary.json'))
        if len(paths) != 1:
            reasons[mode] = 'Нет однозначной сводки этапа.'
            continue
        summary = _json(paths[0], {})
        if not isinstance(summary, dict) or summary.get('status') not in {'completed', 'completed_with_warnings'}:
            reasons[mode] = 'Этап не подтверждён завершённой сводкой.'
            continue
        records = []
        cash_food_conflicts = []
        try:
            interval = float(summary['sample_interval_sec'])
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError('invalid interval')
            with paths[0].with_name('frames.jsonl').open(encoding='utf-8') as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    timestamp, frame = row['timestamp_sec'], row['source_frame']
                    if not _number(timestamp) or timestamp < 0 or type(frame) is not int or frame < 0:
                        raise ValueError('invalid time/frame')
                    if records and (timestamp <= records[-1]['time'] or frame <= records[-1]['frame']):
                        raise ValueError('nonmonotonic time/frame')
                    objects = []
                    for item in row.get('candidates', []):
                        if not isinstance(item, dict):
                            raise ValueError('invalid object row')
                        box, score = _box(item.get('box')), item.get('score')
                        if box and _number(score):
                            objects.append({'label': _label(item.get('label', '')), 'box': box, 'score': float(score),
                                            'source': item.get('source', 'model'), 'score_kind': item.get('score_kind', 'model_score'),
                                            'reference_frame': item.get('reference_frame_zero_based', item.get('reference_frame'))})
                    alternatives = list(objects)
                    for item in row.get('review', []):
                        if not isinstance(item, dict):
                            continue
                        box, score = _box(item.get('box')), item.get('score')
                        if box and _number(score):
                            alternatives.append({'label': _label(item.get('label', '')), 'box': box, 'score': float(score)})
                    records.append({'time': float(timestamp), 'frame': frame, 'index': index, 'mode': mode,
                                    'playback_sec': index / max(.1, 1 / interval), 'objects': objects,
                                    '_class_alternatives': alternatives})
            if not records or len(records) != summary.get('analyzed_samples'):
                raise ValueError('incomplete JSONL')
            loaded[mode] = {'records': records, 'times': [r['time'] for r in records], 'interval': interval, 'summary': summary,
                            'cash_food_conflicts': cash_food_conflicts, 'food_cash_conflicts': []}
        except (OSError, ValueError, TypeError, KeyError):
            reasons[mode] = 'Журнал кадров отсутствует, неполон или имеет некорректные временные отметки.'
    # Inspect an immutable snapshot across modes before removing either class.
    # Otherwise removing cash first would let a conflicting food class win.
    decisions = []
    for mode, data in loaded.items():
        for record in data['records']:
            alternatives = [item for nearby in _nearby_records(loaded, record['time'])
                            for item in nearby['_class_alternatives']]
            retained, cash_conflicts, food_conflicts = [], [], []
            for item in record['objects']:
                conflict = {'timestamp_sec': record['time'], 'source_frame': record['frame'],
                            'box': item['box'], 'reason': 'food_drink_class_conflict'}
                if item['label'] == 'cash' and any(_cash_food_conflict(item, other) for other in alternatives):
                    cash_conflicts.append(conflict)
                elif (item['label'] in FOOD_DRINK and item.get('source') != 'human_confirmed_reference_match'
                      and any(other['label'] == 'cash' and _cash_food_conflict(other, item) for other in alternatives)):
                    food_conflicts.append(conflict)
                else:
                    retained.append(item)
            decisions.append((mode, record, retained, cash_conflicts, food_conflicts))
    for mode, record, retained, cash_conflicts, food_conflicts in decisions:
        record['objects'] = retained
        record.pop('_class_alternatives')
        loaded[mode]['cash_food_conflicts'].extend(cash_conflicts)
        loaded[mode]['food_cash_conflicts'].extend(food_conflicts)
    return loaded, reasons


def _load_people(folder, job):
    metadata = job.get('video', {})
    expected, fps = metadata.get('frames'), metadata.get('fps')
    invalid = lambda reason: {'complete': False, 'reason': reason}
    if not any(s.get('key') == 'people' and s.get('status') == 'completed' for s in job.get('stages', [])):
        return invalid('Трекинг людей не завершён или не запускался.')
    report = _json(folder / 'people/review.json', {})
    if not isinstance(report, dict):
        return invalid('Сводка трекинга повреждена; отсутствие людей не оценивается.')
    if type(expected) is not int or expected <= 0 or not _number(fps) or fps <= 0:
        return invalid('Нет проверенной длительности исходного видео.')
    if (report.get('frames') != expected or report.get('metadata_frames', expected) != expected
            or report.get('full_video_requested', True) is not True or report.get('frame_count_matches', True) is not True
            or report.get('decoding_warning')):
        return invalid('Трекинг прочитал не всё видео или сообщил об ошибке чтения. Отсутствие людей по такому результату не оценивается.')
    tolerance = max(.15, 2 / fps)
    first, last = report.get('first_timestamp_sec'), report.get('last_timestamp_sec')
    if not _number(first) or not _number(last) or abs(first) > tolerance or abs(last - (expected - 1) / fps) > tolerance:
        return invalid('Временные границы трекинга не совпадают с полным видео.')
    expected_hash, expected_rows = report.get('csv_sha256'), report.get('csv_rows')
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 or type(expected_rows) is not int or expected_rows < 0:
        return invalid('Нет проверки полноты журнала траекторий; повторите обработку видео.')
    try:
        with (folder / 'people/review.csv').open('rb') as stream:
            actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual_hash != expected_hash:
            return invalid('Контрольная сумма журнала траекторий не совпадает; отсутствие людей не оценивается. Повторите обработку.')
    except OSError:
        return invalid('Журнал траекторий отсутствует; отсутствие людей не оценивается.')
    by_frame = defaultdict(list)
    by_id = defaultdict(list)
    ids = set()
    previous_frame = -1
    actual_rows = 0
    try:
        with (folder / 'people/review.csv').open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'source_frame', 'timestamp_sec', 'track_id', 'class', 'score', 'x1', 'y1', 'x2', 'y2'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('missing columns')
            for row in reader:
                actual_rows += 1
                frame, timestamp, identity = int(row['source_frame']), float(row['timestamp_sec']), int(row['track_id'])
                box = _box([float(row[key]) for key in ('x1', 'y1', 'x2', 'y2')])
                score = float(row['score'])
                if (row['class'] != 'person' or not box or not _number(score) or score <= 0 or not _number(timestamp)
                        or not 0 <= frame < expected or frame < previous_frame or abs(timestamp - frame / fps) > tolerance):
                    raise ValueError('invalid track row')
                previous_frame = frame
                observation = {'frame': frame, 'time': timestamp, 'box': box, 'id': identity}
                by_frame[frame].append(observation)
                by_id[identity].append(observation)
                ids.add(identity)
        if actual_rows != expected_rows or ids != set(report.get('track_ids', [])):
            raise ValueError('track summary/CSV mismatch')
    except (OSError, ValueError, TypeError, KeyError):
        return invalid('Журнал траекторий неполон или не согласуется со сводкой. Отсутствие людей не оценивается.')
    counts = [0] * expected
    # Count overlapping duplicate tracks once; IDs remain only temporary tracks.
    for frame, rows in by_frame.items():
        unique = []
        for row in rows:
            if not any(row['id'] == other['id'] or _iou(row['box'], other['box']) >= .7 for other in unique):
                unique.append(row)
        counts[frame] = len(unique)
    return {'complete': True, 'reason': 'Все исходные кадры прочитаны; CSV согласуется со сводкой.',
            'counts': counts, 'fps': float(fps), 'by_frame': by_frame, 'by_id': by_id, 'frames': expected}


def _nearby_records(modes, timestamp):
    result = []
    for data in modes.values():
        index = bisect_left(data['times'], timestamp)
        choices = [i for i in (index - 1, index) if 0 <= i < len(data['records'])]
        if choices:
            nearest = min(choices, key=lambda i: abs(data['times'][i] - timestamp))
            # Camera samples from separate stages should line up; do not use a
            # remote frame as if it proved same-time spatial context.
            if abs(data['times'][nearest] - timestamp) <= .2:
                result.append(data['records'][nearest])
    return result


def _context_boxes(modes, timestamp, labels):
    return [item['box'] for row in _nearby_records(modes, timestamp) for item in row['objects']
            if item['label'] in labels and item['score'] >= .3]


def _workspace_box(job):
    context = job.get('context', {})
    if context.get('workspace_confirmed') is not True:
        return None
    box = _box(context.get('workspace_box'))
    width, height = job.get('video', {}).get('width', 0), job.get('video', {}).get('height', 0)
    return box if box and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height else None


def _on_workspace(observation, modes, workspace):
    boxes = [workspace] if workspace else _context_boxes(modes, observation['time'], {'work_table'})
    return any(_inside(observation['box'], box) for box in boxes)


def _unconflicted(observation, modes):
    label, box, timestamp = observation['label'], observation['box'], observation['time']
    if label in FORBIDDEN | FOOD_DRINK | {'cash'}:
        if any(_intersection(box, equipment) / _area(box) > .7 for equipment in _context_boxes(modes, timestamp, EQUIPMENT)):
            return False
    if label == 'cash':
        if any(_intersection(box, bag) / _area(box) >= .5 or _iou(box, bag) > .15
               for bag in _context_boxes(modes, timestamp, {'bag'})):
            return False
        if any(_cash_food_conflict(observation, item) for row in _nearby_records(modes, timestamp)
               for item in row['objects']):
            return False
    return True


def _object_tracks(modes):
    tracks = []
    wanted = FORBIDDEN | UNATTENDED | FOOD_DRINK
    for mode, data in modes.items():
        active = []
        for record in data['records']:
            observations = [dict(item, time=record['time'], frame=record['frame'], mode=mode,
                                 playback_sec=record['playback_sec'], index=record['index'])
                            for item in record['objects'] if item['label'] in wanted and item['score'] >= (FOOD_DRINK_MIN_SCORE if item['label'] in FOOD_DRINK else MIN_SCORE)]
            remaining = set(range(len(active)))
            next_active = []
            for observation in sorted(observations, key=lambda o: o['score'], reverse=True):
                candidates = [i for i in remaining if active[i][-1]['label'] == observation['label']
                              and observation['time'] - active[i][-1]['time'] <= data['interval'] * 1.6 + .05
                              and _iou(active[i][-1]['box'], observation['box']) >= .4]
                if candidates:
                    selected = max(candidates, key=lambda i: _iou(active[i][-1]['box'], observation['box']))
                    remaining.remove(selected)
                    active[selected].append(observation)
                    next_active.append(active[selected])
                else:
                    next_active.append([observation])
            tracks.extend(active[i] for i in remaining)
            active = next_active
        tracks.extend(active)
    return tracks


def _segments(track, predicate):
    segments, current = [], []
    for observation in track:
        if predicate(observation):
            current.append(observation)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _sustained(track, stationary=False, minimum_span=MIN_EVIDENCE_SPAN):
    if len(track) < MIN_OBSERVATIONS or track[-1]['time'] - track[0]['time'] < minimum_span - 1e-6:
        return False
    if stationary:
        centers = [_center(row['box']) for row in track]
        center = (median(p[0] for p in centers), median(p[1] for p in centers))
        diagonal = median(math.hypot(row['box'][2] - row['box'][0], row['box'][3] - row['box'][1]) for row in track)
        if max(math.dist(p, center) for p in centers) > max(6, .2 * diagonal):
            return False
    return True


def _food_detector_available(modes):
    """Legacy journals without an explicit food prompt cannot assess this rule."""
    data = modes.get('desk', {})
    summary = data.get('summary', {})
    signature = summary.get('signature', {})
    groups = summary.get('groups', signature.get('groups', []) if isinstance(signature, dict) else [])
    return bool(data.get('records') and isinstance(groups, list) and 'food_drink' in groups)


def _near_cashier_workplace(observation, modes, workspace, people, service):
    """Spatial association of a cup with a role; never proof of holding/drinking."""
    role = service.get('role_summary', {})
    cashier = role.get('cashier_track_id')
    if not people.get('complete') or role.get('status') != 'candidate' or type(cashier) is not int:
        return False
    rows = people.get('by_frame', {}).get(observation['frame'], [])
    cashier_rows = [row for row in rows if row['id'] == cashier]
    if len(cashier_rows) != 1:
        return False
    box, person = observation['box'], cashier_rows[0]['box']
    association = _intersection(box, person) / _area(box)
    if association < .5 or any(row['id'] != cashier and _intersection(box, row['box']) / _area(box) >= .5 for row in rows):
        return False
    # A cup near the floor / feet is not associated with a desk interaction.
    if _center(box)[1] > person[1] + .85 * (person[3] - person[1]):
        return False
    tables = [workspace] if workspace else _context_boxes(modes, observation['time'], {'work_table'})
    for table in tables:
        width, height = table[2] - table[0], table[3] - table[1]
        near_table = [table[0] - .05 * width, table[1] - .6 * height, table[2] + .05 * width, table[3]]
        if _inside(box, near_table, fraction=.5):
            return True
    return False


def _food_drink_events(tracks, modes, workspace, has_workspace, people, service):
    role = service.get('role_summary', {})
    anchors = role.get('anchors', {})
    stable_table = (_box(anchors.get('table_box'))
                    if role.get('status') == 'candidate' and anchors.get('source') == 'stable_model_candidates' else None)
    food_workspace = workspace or stable_table
    workspace_source = 'user_confirmed' if workspace else 'stable_model_anchor' if stable_table else 'model_candidate'
    if not (has_workspace or food_workspace) or not _food_detector_available(modes):
        return []
    events = []
    for track in tracks:
        if track[0]['mode'] != 'desk' or track[0]['label'] not in FOOD_DRINK:
            continue
        # Choose one supported spatial explanation for each observation.
        def relation(row):
            if not _unconflicted(row, modes):
                return None
            if _near_cashier_workplace(row, modes, food_workspace, people, service):
                return 'cashier_near_table'
            if _on_workspace(row, modes, food_workspace):
                return 'table_projection'
            return None
        for segment in _segments(track, lambda row: relation(row) is not None):
            if not _sustained(segment, minimum_span=FOOD_DRINK_MIN_SPAN):
                continue
            relations = {relation(row) for row in segment}
            # A desk-only item must be stationary; movement near the role may
            # represent a held cup, but is not promoted to an action label.
            if relations == {'table_projection'} and not _sustained(segment, stationary=True, minimum_span=FOOD_DRINK_MIN_SPAN):
                continue
            name = NAMES[segment[0]['label']]
            location = ('в проекции рабочего стола' if relations == {'table_projection'} else
                        'возле рабочего стола и в проекции кандидата кассира')
            reference_count = sum(row.get('source') == 'human_confirmed_reference_match' for row in segment)
            if reference_count == len(segment):
                evidence_source = 'human_confirmed_reference_match'
                origin = 'В каждом сохранённом наблюдении найдено совпадение с образцом предмета из этой же записи, класс которого уточнён пользователем.'
            elif reference_count:
                evidence_source = 'mixed_model_and_human_reference'
                origin = 'В части кадров есть совпадение с образцом предмета из этой же записи, класс которого уточнён пользователем; в остальных кадрах класс предложен моделью и требует просмотра.'
            else:
                evidence_source = 'model'
                origin = 'Класс предмета предложен моделью и требует просмотра.'
            events.append(_event('food_drinks_at_workplace', 'Еда и напитки у рабочего места', segment,
                                 f'Повторно отмечен предмет «{name}» {location}. {origin} По правилу пользователя еда и напитки у рабочего места требуют проверки независимо от наличия денег и ухода клиента.',
                                 ['Распознавание еды или чашки может быть ошибочным; содержимое чашки (кофе или другой напиток) не установлено.',
                                  'Присутствие чашки или шоколадки не доказывает, что человек пьёт или ест. Личная принадлежность предмета не определяется.',
                                  'Положение рамки в проекции стола не доказывает контакт с поверхностью; пространственная связь с кассиром не доказывает удержание рукой.'],
                                 {'workspace_source': workspace_source,
                                  'spatial_relations': sorted(relations), 'requires_cash': False,
                                  'requires_client_departure': False, 'consumption_action_checked': False,
                                  'object_evidence_source': evidence_source,
                                  'reference_match_observations': reference_count,
                                  'model_observations': len(segment) - reference_count,
                                  'cashier_track_id': service.get('role_summary', {}).get('cashier_track_id') if 'cashier_near_table' in relations else None}))
    return events


def _runs(values):
    if not values:
        return []
    runs, start = [], 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            runs.append((start, index - 1, values[start]))
            start = index
    return runs


def _absence_windows(people):
    if not people['complete']:
        return []
    fps, counts = people['fps'], people['counts']
    context_frames = math.ceil(MIN_EVIDENCE_SPAN * fps)
    result = []
    for start, end, count in _runs([int(value > 0) for value in counts]):
        if count or (end - start + 1) / fps < MIN_EVIDENCE_SPAN or start < context_frames:
            continue
        before = counts[start - context_frames:start]
        if sum(value > 0 for value in before) / len(before) >= .8:
            result.append({'start_sec': start / fps, 'end_sec': end / fps,
                           'start_frame': start, 'end_frame': end,
                           'prior_presence_start_sec': (start - context_frames) / fps})
    return result


def _edge_exit(track, transition_frame, width, height, fps):
    if len(track) < 3 or abs(track[-1]['frame'] - transition_frame) > max(2, fps * .2):
        return False
    tail = [row for row in track if track[-1]['time'] - row['time'] <= 1]
    if len(tail) < 3 or tail[-1]['time'] - tail[0]['time'] < .3:
        return False
    last = tail[-1]['box']
    distances = [last[0], last[1], width - last[2], height - last[3]]
    edge = min(range(4), key=lambda i: distances[i])
    if distances[edge] > max(8, .03 * min(width, height)):
        return False
    a, b = _center(tail[0]['box']), _center(last)
    movement = [a[0] - b[0], a[1] - b[1], b[0] - a[0], b[1] - a[1]][edge]
    return movement >= max(8, .02 * min(width, height))


def _departure_windows(people, job):
    if not people['complete']:
        return []
    fps = people['fps']
    width, height = job.get('video', {}).get('width', 0), job.get('video', {}).get('height', 0)
    if width <= 0 or height <= 0:
        return []
    runs = _runs(people['counts'])
    result = []
    for before, after in zip(runs, runs[1:]):
        a, b, count_before = before
        c, d, count_after = after
        if not (count_before >= 2 and 1 <= count_after < count_before
                and (b - a + 1) / fps >= MIN_EVIDENCE_SPAN and (d - c + 1) / fps >= MIN_EVIDENCE_SPAN):
            continue
        for identity, track in people['by_id'].items():
            if track[-1]['frame'] < c and _edge_exit(track, b, width, height, fps):
                result.append({'start_sec': c / fps, 'end_sec': d / fps, 'before_count': count_before,
                               'after_count': count_after, 'track_id': identity,
                               'exit_time': track[-1]['time'], 'exit_box': track[-1]['box'],
                               'exit_source_frame': track[-1]['frame'],
                               'person_tail': [row for row in track if c / fps - 2 <= row['time'] < c / fps]})
    return result


def _clip_end_cash_events(tracks, modes, workspace, people, service, job):
    """Review a short, fully decoded empty tail without claiming an exit.

    This is limited to cash already observed at the same table position before
    the last visible cashier. It never bridges a missing object sample and does
    not relax the ordinary two-second absence or visitor-departure checks.
    """
    if people.get('complete') is not True:
        return []
    role = service.get('role_summary', {})
    cashier = role.get('cashier_track_id')
    if role.get('status') != 'candidate' or type(cashier) is not int:
        return []
    counts, fps, frames = people.get('counts'), people.get('fps'), people.get('frames')
    if (not isinstance(counts, list) or type(frames) is not int or len(counts) != frames
            or frames < 2 or not _number(fps) or fps <= 0
            or any(type(value) is not int or value < 0 for value in counts)
            or counts[-1] != 0):
        return []
    occupied = [frame for frame, count in enumerate(counts) if count]
    if not occupied:
        return []
    last_visible = occupied[-1]
    start_frame = last_visible + 1
    duration = (frames - start_frame) / fps
    if not CLIP_END_MIN_ABSENCE <= duration < MIN_EVIDENCE_SPAN:
        return []
    context_frames = math.ceil(MIN_EVIDENCE_SPAN * fps)
    if start_frame < context_frames:
        return []
    before_frames = range(start_frame - context_frames, start_frame)
    by_frame = people.get('by_frame', {})
    if (sum(counts[frame] > 0 for frame in before_frames) / context_frames < .8
            or sum(any(row.get('id') == cashier for row in by_frame.get(frame, []))
                   for frame in before_frames) / context_frames < .8):
        return []
    last_people = by_frame.get(last_visible, [])
    cashier_track = people.get('by_id', {}).get(cashier, [])
    if (not last_people or any(row.get('id') != cashier for row in last_people)
            or not cashier_track or cashier_track[-1].get('frame') != last_visible
            or not _box(cashier_track[-1].get('box'))
            or any(by_frame.get(frame) for frame in range(start_frame, frames))):
        return []
    anchors = role.get('anchors', {})
    if not workspace and not (anchors.get('source') == 'stable_model_candidates'
                              and _box(anchors.get('table_box'))):
        return []
    start, end = start_frame / fps, (frames - 1) / fps
    events = []
    for track in tracks:
        if not track or track[0]['label'] != 'cash':
            continue
        segments = _segments(track, lambda row: _on_workspace(row, modes, workspace)
                             and _unconflicted(row, modes))
        for segment in segments:
            before = [row for row in segment if start - MIN_EVIDENCE_SPAN <= row['time'] < start]
            after = [row for row in segment if start <= row['time'] <= end + 1e-6]
            if (len({row['time'] for row in before}) < MIN_OBSERVATIONS
                    or len({row['time'] for row in after}) < MIN_OBSERVATIONS
                    or not _sustained(before, stationary=True, minimum_span=CLIP_END_MIN_CASH_SPAN)
                    or not _sustained(after, stationary=True, minimum_span=CLIP_END_MIN_CASH_SPAN)
                    or any(_iou(row['box'], before[0]['box']) < .65 for row in before + after)):
                continue
            last_cashier = cashier_track[-1]
            cashier_evidence = {'timestamp_sec': last_cashier['time'], 'source_frame': last_visible,
                                'mode': 'people', 'playback_sec': last_cashier['time'],
                                'box': last_cashier['box'], 'labels': ['person'], 'source': 'tracker'}
            event = _event('unattended_assets', 'Деньги на столе, кассир не виден в конце записи', after,
                           'В конце полностью прочитанной записи трекер не отмечает людей. '
                           'Предмет, отмеченный как банкноты, повторно виден на прежнем месте в проекции стола до и после последней видимой траектории кассира. '
                           'Короткий конец записи выделен для просмотра; наблюдаемый выход из помещения не установлен.',
                           ['Запись заканчивается во время отсутствия траекторий; дальнейшее присутствие людей неизвестно.',
                            'Пропуск или перекрытие человека детектором возможны. Фактический присмотр вне кадра не установлен.',
                            'Положение в проекции стола не доказывает физический контакт с поверхностью; класс небольшого предмета требует проверки.',
                            'Одна секунда — фильтр короткого конца записи, а не разрешённое время оставления денег.'],
                           {'recording_ends_during_absence': True, 'cashier_track_id': cashier,
                            'people_decode_complete': True, 'scope': 'visible_frame_only',
                            'same_projected_location': True, 'visitor_exit_observed': False,
                            'cashier_exit_observed': False, 'absence_duration_sec': duration,
                            'absence': {'start_sec': start, 'end_sec': end,
                                        'start_frame': start_frame, 'end_frame': frames - 1},
                            'last_visible_cashier_sec': last_cashier['time'],
                            'cash_before_absence_count': len(before), 'cash_during_absence_count': len(after),
                            'minimum_cash_span_sec': CLIP_END_MIN_CASH_SPAN})
            # The displayed event bounds remain the cash observations after the
            # absence starts. Context frames never turn its label into person.
            event['evidence_times'].extend([_evidence(before[0]), _evidence(before[-1]), cashier_evidence])
            events.append(event)
    return events


def _evidence(observation):
    return {'timestamp_sec': observation['time'], 'source_frame': observation['frame'],
            'mode': observation['mode'], 'playback_sec': observation['playback_sec'],
            'box': observation['box'], 'labels': [observation['label']],
            'source': observation.get('source', 'model'), 'score_kind': observation.get('score_kind', 'model_score')}


def _event(rule_id, title, observations, description, limitations, context=None):
    observations = sorted(observations, key=lambda row: (row['time'], row['mode'], row['label']))
    first, last = observations[0], observations[-1]
    labels = sorted({row['label'] for row in observations})
    token = f"{rule_id}:{first['mode']}:{first['frame']}:{last['frame']}:{first['box']}"
    evidence = [_evidence(row) for row in observations]
    return {'id': hashlib.sha256(token.encode()).hexdigest()[:16], 'rule_id': rule_id, 'title': title,
            'start_sec': first['time'], 'end_sec': last['time'], 'description': description,
            'evidence_text': f"{first['time']:.2f}–{last['time']:.2f} с: {len(observations)} наблюдений; " + ', '.join(NAMES.get(label, label) for label in labels) + '.',
            'limitations': limitations, 'mode': first['mode'], 'playback_sec': first['playback_sec'],
            'source_frame': first['frame'], 'labels': labels, 'evidence_box': first['box'],
            'evidence_times': evidence, 'context': context or {}, 'severity': 'review',
            'confidence_label': 'Требует проверки', 'rule_source': RULE_SOURCE}


def _coverage(rule_id, title, status, detail, events, limitations=None, **extra):
    count = sum(event['rule_id'] == rule_id for event in events)
    return {'rule_id': rule_id, 'title': title, 'status': status, 'detail': detail, 'reason': detail,
            'event_count': count, 'events_count': count, 'rule_source': RULE_SOURCE,
            'limitations': limitations or [], **extra}


def _deduplicate_absence_events(legacy, role_events):
    """Remove only whole asset tracks already covered by a cashier episode.

    The older all-people-absent episode can include other objects elsewhere.
    Matching a shared laptop must not discard those independent observations.
    """
    specific = [event for event in role_events if event['rule_id'] == 'unattended_assets']
    result = []
    for event in legacy:
        if event['rule_id'] != 'unattended_assets' or not specific:
            result.append(event)
            continue
        groups = []
        for row in event['evidence_times']:
            group = next((g for g in groups if g[0]['mode'] == row['mode']
                          and g[0]['labels'] == row['labels'] and _iou(g[0]['box'], row['box']) >= .4), None)
            if group is None:
                groups.append([row])
            else:
                group.append(row)
        retained = []
        for group in groups:
            covered = any(all(any(set(row['labels']) == set(other['labels'])
                                         and abs(row['timestamp_sec'] - other['timestamp_sec']) <= .3
                                         and _iou(row['box'], other['box']) >= .5
                                         for other in candidate['evidence_times'])
                                  for row in group) for candidate in specific)
            if not covered:
                retained.extend(group)
        if len(retained) == len(event['evidence_times']):
            result.append(event)
        elif retained:
            observations = [dict(time=r['timestamp_sec'], frame=r['source_frame'], mode=r['mode'],
                                 playback_sec=r['playback_sec'], box=r['box'], label=r['labels'][0])
                            for r in retained]
            result.append(_event(event['rule_id'], event['title'], observations, event['description'],
                                 event['limitations'], event['context']))
    return result


def build_rule_report(job_dir: Path, job: dict, results: dict) -> dict:
    """Return evidence-backed review episodes and explicit coverage for all rules."""
    folder = Path(job_dir)
    modes, mode_reasons = _load_modes(folder, job)
    from state_rules import load_verified_states, analyze_states
    from reflection_tracks import analyze_reflections, filter_reflections
    raw_people = _load_people(folder, job)
    try:
        verified_states = load_verified_states(folder, job)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        verified_states = None
    reflections = analyze_reflections(raw_people, job, (verified_states or {}).get('observations', []))
    people = filter_reflections(raw_people, reflections)
    workspace = _workspace_box(job)
    has_workspace = bool(workspace or any(item['label'] == 'work_table' and item['score'] >= .3
                                         for data in modes.values() for row in data['records'] for item in row['objects']))
    table_note = ('Область рабочего стола указана пользователем; положение предмета внутри неё не доказывает контакт с поверхностью.'
                  if workspace else 'Граница стола предложена моделью и может быть ошибочной; физический контакт предмета с поверхностью не подтверждён.')
    tracks = _object_tracks(modes)
    events = []
    common = ['Распознавание предмета может ошибаться; оценка модели не является вероятностью нарушения.',
              'Кадры между выбранными наблюдениями модель объектов не анализирует.']
    # A desk context is mandatory. Repeated stationary locks and laptop details
    # must not become banned-item events merely through temporal persistence.
    if has_workspace:
        for track in tracks:
            if track[0]['label'] not in FORBIDDEN:
                continue
            segments = _segments(track, lambda row: _on_workspace(row, modes, workspace) and _unconflicted(row, modes))
            for segment in segments:
                if _sustained(segment, stationary=True):
                    name = NAMES[segment[0]['label']]
                    events.append(_event('clean_desk_banned_items', 'Проверить предмет в рабочей зоне', segment,
                                         f'Модель повторно отмечает {name} в похожем положении внутри предполагаемой рабочей поверхности. Проверьте предмет и применимость ограничения.',
                                         common + [table_note, 'Личная принадлежность телефона, записи в блокноте и содержимое флешки не установлены.'],
                                         {'workspace_source': 'user_confirmed' if workspace else 'model_candidate', 'stationary_observations': True}))
    for window in _absence_windows(people):
        observed = []
        for track in tracks:
            if track[0]['label'] not in UNATTENDED:
                continue
            for segment in _segments(track, lambda row: window['start_sec'] <= row['time'] <= window['end_sec'] and _unconflicted(row, modes)):
                if _sustained(segment, stationary=True):
                    observed.extend(segment)
        if observed:
            events.append(_event('unattended_assets', 'Предметы в кадрах, где не видно людей', observed,
                                 'После периода присутствия людей трекер перестал отмечать людей, а предметы повторяются в выбранных кадрах. Проверьте, оставались ли они под присмотром.',
                                 common + ['Отсутствие траектории не доказывает отсутствие человека: возможны перекрытия и ошибки детектора.',
                                           'Камера не показывает наблюдателей вне кадра; роль и зона ответственности людей неизвестны.',
                                           'Пакет не подтверждает наличие денег внутри.'],
                                 {'people_decode_complete': True, 'absence': window, 'absence_duration_sec': (window['end_frame'] - window['start_frame'] + 1) / people['fps']}))
    events.sort(key=lambda row: (row['start_sec'], row['rule_id']))
    desktop_status = 'partial' if 'desk' in modes and has_workspace else 'not_checked'
    desktop_detail = ('Проверяются устойчивые кандидаты смартфона, флешки и блокнота в предполагаемой рабочей зоне; еда и напитки проверяются отдельно. Предметы внутри сейфа и техники исключаются. Полный перечень личных вещей и принадлежность предметов не определяются.'
                      if desktop_status == 'partial' else 'Невозможно надёжно локализовать стол или нет завершённого распознавания предметов. Неподвижный предмет сам по себе событие не создаёт.')
    assets_status = 'partial' if people['complete'] and modes else 'not_checked'
    assets_detail = ('Сопоставляются повторные наблюдения предметов с устойчивым отсутствием траекторий после периода присутствия. Фактический присмотр не определяется.'
                     if assets_status == 'partial' else people.get('reason') if not people['complete'] else 'Нет завершённого распознавания предметов.')
    coverage = [
        _coverage('clean_desk_banned_items', 'Личные предметы на столе: телефон, флешка, блокнот', desktop_status, desktop_detail, events, [table_note] if has_workspace else []),
        _coverage('unattended_assets', 'Деньги, пакеты и компьютер под присмотром', assets_status, assets_detail, events),
        _coverage('food_drinks_at_workplace', 'Еда и напитки у рабочего места',
                  'partial' if _food_detector_available(modes) and has_workspace else 'not_checked',
                  ('Проверяются устойчивые кандидаты еды и чашек в рабочей зоне, в том числе возле кандидата кассира. Наличие денег и уход клиента не требуются. Приём пищи, питьё и содержимое чашки не устанавливаются.'
                   if _food_detector_available(modes) and has_workspace else
                   'В этой обработке не запускалось распознавание еды и напитков либо нет рабочей зоны. Старый результат нужно обработать новым детектором; отсутствие этого события не подтверждает отсутствие еды.'), events),
    ]
    unsupported = [
        ('door_open', DOOR_OPEN_TITLE, 'Состояния открытой и закрытой двери не распознаются; наличие двери не измеряет длительность открытия.'),
        ('safe_closed', 'Сейф закрыт после доступа и получения денег', 'Детектор находит сейф как предмет, но не распознаёт его дверцу, замок и факт завершения доступа.'),
        ('cash_counted_visible', 'Деньги пересчитаны через счётчик на виду у клиента', 'Наличие счётчика и банкнот не доказывает прохождение денег через аппарат и видимость операции клиенту.'),
        ('change_through_counter', 'Остаток и возврат проходят через счётчик', 'Нет распознавания движения пачек через счётчик, суммы, возврата и завершения операции.'),
        ('original_id_verification', 'Проверка оригинала удостоверения клиента', 'Распознавание карточки не проверяет её подлинность, оригинальность и выполнение верификации.'),
        ('previous_client_documents', 'На столе нет документов прошлого клиента', 'Содержимое и принадлежность документов, границы обслуживания и порядок клиентов не определяются.'),
        ('peripheral_test', 'Тест сканера удостоверений и биометрической камеры', 'По присутствию оборудования нельзя установить выполнение тестового запуска и его результат.'),
        ('permitted_desk_equipment', 'Только разрешённая техника и брендированная канцелярия', 'По заданному пользователем правилу на столе разрешены монитор, клавиатура, мышь, сканер документов, POS-терминал, принтер и брендированная канцелярия. Полного распознавания этого перечня и проверки бренда нет; отдельно проверяются телефон, флешка, блокнот, еда и напитки. Личная принадлежность предметов не определяется.'),
        ('cash_placed_on_counter', 'Клиент выкладывает деньги на стол или в лоток', 'Нет классификатора передачи денег и подтверждённого определения клиента, поверхности и лотка.'),
        ('one_client_at_a_time', 'Предыдущий клиент обслужен до следующего', 'Не определяются роли людей и конец операции. Два человека в кадре не означают двух клиентов.'),
        ('restricted_access_card_transfer', 'Посторонний у кассира и передача чужой карты', 'Нет подтверждённых зон доступа, ролей и принадлежности карты; наличие людей и карточки недостаточно.'),
        ('unattended_room', 'Кабинет не оставлен без присмотра', 'Не определена контролируемая зона и кто отвечает за присмотр; камера не показывает людей за её пределами.'),
        ('lights_after_departure', 'Свет после ухода из кабинета', 'Нет подтверждённого правила освещения и модели состояния света; яркость кадра не доказывает нарушение.'),
        ('eating_while_cash_present', 'При наличии денег кассир не ест', 'Действие приёма пищи пока не распознаётся: нужны подтверждённые банкноты и последовательность приёма пищи в тот же период. Наличие еды или чашки проверяется отдельным правилом независимо от денег. Рука у лица, телефон или чашка сами по себе не доказывают еду.'),
    ]
    for rule_id, title, detail in unsupported:
        coverage.append(_coverage(rule_id, title, 'not_checked', detail, events,
                                  **({'user_threshold_sec': DOOR_OPEN_THRESHOLD_SEC} if rule_id == 'door_open' else {})))
    from service_events import analyze_service
    from interaction_events import analyze_interactions
    service = analyze_service(modes, people, job)
    events.extend(_clip_end_cash_events(tracks, modes, workspace, people, service, job))
    events.extend(_food_drink_events(tracks, modes, workspace, has_workspace, people, service))
    states = analyze_states(folder, job, service, verified_states)
    interactions = analyze_interactions(modes, people, service, job)
    from action_rules import analyze_actions
    actions = analyze_actions(modes, people, service, job)
    events = _deduplicate_absence_events(events, service['events'])
    events.extend(service['events'])
    events.extend(states['events'])
    events.extend(actions['events'])
    revised = {row['rule_id']: row for row in coverage}
    for analysis in (service, states, interactions, actions):
        for row in analysis['coverage']:
            old = revised.get(row['rule_id'])
            if analysis is not actions and old and old['status'] == 'partial' and row['status'] == 'not_checked':
                continue
            revised[row['rule_id']] = row
    coverage = list(revised.values())
    for row in coverage:
        row['event_count'] = row['events_count'] = sum(event['rule_id'] == row['rule_id'] for event in events)
    from check_coverage import describe_coverage
    coverage = describe_coverage(coverage, actions, states, service)
    for row in coverage:
        if row['rule_id'] != 'unattended_assets':
            continue
        note = ('Отдельно проверяется короткий конец полностью прочитанной записи: люди не видны от 1 до 2 секунд, '
                'а те же банкноты повторно наблюдаются на столе до и после последнего видимого кассира. '
                'Такой эпизод не подтверждает выход из помещения или отсутствие присмотра.')
        row['detail'] = row['reason'] = row['detail'] + ' ' + note
        available = bool(people['complete'] and modes and has_workspace
                         and service.get('role_summary', {}).get('status') == 'candidate')
        count = sum(event['rule_id'] == 'unattended_assets'
                    and event.get('context', {}).get('recording_ends_during_absence') is True for event in events)
        row.setdefault('checks', []).append({'check_id': 'cash_at_truncated_recording_end',
                                            'title': 'Деньги в коротком конце записи без видимого кассира',
                                            'available': available,
                                            'status': 'observed' if count else 'analyzed' if available else 'insufficient_evidence',
                                            'observation_count': count, 'detail': note})
    events.sort(key=lambda row: (row['start_sec'], row['rule_id']))
    partial_count = sum(row['status'] == 'partial' for row in coverage)
    if events:
        summary = f'Найдено эпизодов для ручной проверки: {len(events)}. Подтверждённых нарушений модель не устанавливает.'
        verdict = 'review_required'
    elif partial_count:
        summary = 'Устойчивых эпизодов по доступным проверкам не найдено. Это не означает, что нарушений нет: часть правил не проверяется.'
        verdict = 'no_candidates'
    else:
        summary = 'Недостаточно данных для доступных временных проверок. Автоматическое заключение о нарушениях не вынесено.'
        verdict = 'not_evaluated'
    from episode_steps import attach_episode_steps
    return attach_episode_steps({'report_version': REPORT_VERSION, 'events': events, 'rule_coverage': coverage, 'verdict': verdict,
            'service': service, 'visual_states': states, 'interactions': interactions,
            'actions': actions, 'reflections': reflections,
            'summary': summary, 'analysis_settings': {'minimum_object_score': MIN_SCORE, 'minimum_observations': MIN_OBSERVATIONS,
            'minimum_evidence_span_sec': MIN_EVIDENCE_SPAN, 'door_open_threshold_sec': DOOR_OPEN_THRESHOLD_SEC,
            'cash_on_table_policy': 'only_after_observed_visitor_departure',
            'short_clip_cash_review': {'implemented': True, 'eof_tail_only': True,
                                      'minimum_absence_sec': CLIP_END_MIN_ABSENCE,
                                      'exclusive_maximum_absence_sec': MIN_EVIDENCE_SPAN,
                                      'minimum_cash_span_sec': CLIP_END_MIN_CASH_SPAN,
                                      'minimum_observations_before_and_after': MIN_OBSERVATIONS,
                                      'requires_cashier_role': True, 'claims_observed_exit': False,
                                      'threshold_kind': 'prototype_evidence_filter_not_allowed_duration'},
            'food_drinks_policy': {'rule_id': 'food_drinks_at_workplace', 'implemented': True,
                                   'requires_cash': False, 'requires_client_departure': False,
                                   'consumption_action_checked': False,
                                   'minimum_object_score': FOOD_DRINK_MIN_SCORE,
                                   'minimum_observations': MIN_OBSERVATIONS,
                                   'minimum_evidence_span_sec': FOOD_DRINK_MIN_SPAN,
                                   'threshold_kind': 'prototype_evidence_filter_not_allowed_duration'},
            'eating_while_cash_present': {'requested': True, 'implemented': False},
            'note': 'Фильтры устойчивости прототипа, не нормативные пороги. Оценка качества не измерена.'},
            'data_quality': {'object_modes': list(modes), 'unavailable_modes': mode_reasons,
                             'cash_food_class_conflicts': {name: data.get('cash_food_conflicts', []) for name, data in modes.items()
                                                          if data.get('cash_food_conflicts')},
                             'food_cash_class_conflicts': {name: data.get('food_cash_conflicts', []) for name, data in modes.items()
                                                          if data.get('food_cash_conflicts')},
                             'people_complete': people['complete'], 'people_detail': people['reason'],
                             'workspace_context': 'user_confirmed' if workspace else 'model_candidate' if has_workspace else 'missing'}})
