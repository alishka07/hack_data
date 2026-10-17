"""Spatial object observations and short sequence hypotheses, never action proof.

Consumes rule_engine's validated modes/people and analyze_service's result.
Does not infer possession, hand contact, counting, identity or a violation.
No inference, downloads, camera-specific coordinates or training are performed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math

KINDS = {'cash': 'cash', 'banknotes': 'cash', 'banknote': 'cash',
         'identity_card': 'card', 'card': 'card', 'bank_card': 'card',
         'document': 'document', 'paper_document': 'document'}
NAMES = {'cash': 'Предполагаемые банкноты', 'card': 'Предполагаемая карточка', 'document': 'Предполагаемый документ'}
COMMON = [
    'Рамка предмета — гипотеза детектора; класс и положение могут быть ошибочными.',
    'Близость на изображении не доказывает контакт, нахождение в руке, владение или передачу.',
    'Перекрытие со счётчиком не подтверждает прохождение денег через аппарат и пересчёт.',
    'Промежутки между выбранными кадрами не наблюдаются; короткая передача может быть пропущена.',
    'Номер траектории не устанавливает личность человека; отражения и смена номера остаются возможными.',
]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_number(v) for v in value):
        return None
    return list(map(float, value)) if value[2] > value[0] and value[3] > value[1] else None


def _center_inside(box, container):
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return container[0] <= x <= container[2] and container[1] <= y <= container[3]


def _overlap(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def _evidence(observation):
    return {'timestamp_sec': observation['timestamp_sec'], 'source_frame': observation['source_frame'],
            'mode': observation['mode'], 'playback_sec': observation['playback_sec'],
            'box': observation['box'], 'labels': [observation['label']],
            'score': observation['score'], 'observation_id': observation['observation_id']}


def _token(parts):
    return hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:16]


def _coverage(observations):
    kinds = Counter(row['object_type'] for row in observations)
    cash_contexts = Counter(tag for row in observations if row['object_type'] == 'cash' for tag in row['relations'])
    if kinds['cash']:
        context = f"Есть наблюдения предполагаемых банкнот: {kinds['cash']}. "
        if cash_contexts['near_counter']:
            context += 'На отдельных кадрах рамка банкнот пересекается с проекцией счётчика. '
        if cash_contexts['over_table']:
            context += 'На отдельных кадрах центр рамки банкнот находится в проекции стола. '
    else:
        context = 'В доступных кадрах нет пригодных наблюдений банкнот. Это не доказывает их отсутствия. '
    card_context = (f"Есть наблюдения предполагаемой карточки: {kinds['card']}. " if kinds['card'] else
                    'В доступных кадрах нет пригодных наблюдений карточки. Это не доказывает её отсутствия. ')
    rules = [
        ('cash_counted_visible', 'Деньги пересчитаны через счётчик на виду у клиента', context +
         'Пересчёт и видимость операции клиенту не проверены: нет распознавания работы аппарата и прохождения пачки.'),
        ('change_through_counter', 'Остаток и возврат проходят через счётчик', context +
         'Возврат, остаток и их прохождение через счётчик не установлены. Пространственная цепочка не определяет смысл операции.'),
        ('cash_placed_on_counter', 'Клиент выкладывает деньги на стол или в лоток', context +
         'Кто положил деньги и касаются ли они поверхности, не установлено. Проекция над столом не доказывает выкладывание.'),
        ('original_id_verification', 'Проверка оригинала удостоверения клиента', card_context +
         'Тип карточки, оригинальность, принадлежность и выполнение проверки не устанавливаются по рамке предмета.'),
    ]
    return [{'rule_id': rule_id, 'title': title, 'status': 'not_checked', 'detail': detail, 'reason': detail,
             'event_count': 0, 'events_count': 0, 'limitations': list(COMMON), 'source': 'interaction_observations'}
            for rule_id, title, detail in rules]


def analyze_interactions(modes: dict, people: dict, service: dict, job: dict) -> dict:
    """Return timeline, typed sequences, hypotheses and honest rule coverage.

    ``near_person`` = object center inside a person's projected bounding box.
    ``over_table`` = object center inside a table's projected bounding box.
    ``near_counter`` = positive 2-D box intersection with a banknote counter.
    These are image relations, not actions. Object associations are hypotheses
    across adjacent samples of the same mode and class with mutual unique box
    overlap. Missing observations, sampling gaps and ambiguous matches break
    the chain. No duration cutoff is interpreted as a violation.
    """
    observations, sequences, hypotheses, issues = [], [], [], []
    role = (service or {}).get('role_summary', {})
    reflection_ids = {identity for pair in role.get('reflection_candidate_pairs', []) for identity in pair}
    fps = job.get('video', {}).get('fps')
    fps = float(fps) if _number(fps) and fps > 0 else None
    frame_count = job.get('video', {}).get('frames')
    frames_known = type(frame_count) is int and frame_count > 0
    people_usable = bool(people.get('complete') and fps and frames_known and people.get('frames') == frame_count
                         and _number(people.get('fps')) and abs(people['fps'] - fps) < .01)
    records_by_mode, context_by_frame, sample_intervals = {}, defaultdict(list), {}
    gaps = []
    for mode, data in modes.items():
        valid, previous = [], None
        for index, record in enumerate(data.get('records', [])):
            timestamp, frame = record.get('time'), record.get('frame')
            if (not _number(timestamp) or timestamp < 0 or type(frame) is not int or frame < 0
                    or (frames_known and frame >= frame_count)
                    or (previous and (timestamp <= previous['time'] or frame <= previous['frame']))):
                issues.append({'mode': mode, 'record_index': index, 'reason': 'invalid_or_nonmonotonic_source_frame'})
                # Whole mode is unusable for temporal interpretation.
                valid = []
                break
            valid.append(record)
            previous = record
        interval = data.get('interval')
        sample_intervals[mode] = float(interval) if _number(interval) and interval > 0 else None
        records_by_mode[mode] = valid
        for record in valid:
            for item in record.get('objects', []):
                box = _box(item.get('box'))
                score = item.get('score')
                if not box or not _number(score) or not 0 <= score <= 1:
                    continue
                context_by_frame[record['frame']].append({'label': item.get('label'), 'box': box, 'score': float(score),
                                                        'mode': mode, 'time': record['time']})
    for mode, records in records_by_mode.items():
        interval = sample_intervals[mode]
        previous, previous_observations = None, []
        for index, record in enumerate(records):
            timestamp, frame = record['time'], record['frame']
            context = [item for item in context_by_frame[frame]
                       if abs(item['time'] - timestamp) <= (1 / fps if fps else 1e-6)]
            current = []
            for object_index, obj in enumerate(record.get('objects', [])):
                kind = KINDS.get(obj.get('label'))
                box, score = _box(obj.get('box')), obj.get('score')
                if not kind or not box or not _number(score) or not 0 <= score <= 1:
                    continue
                evidence_id = _token((mode, frame, object_index, kind))
                near = []
                if people_usable:
                    for person in people.get('by_frame', {}).get(frame, []):
                        pbox = _box(person.get('box'))
                        if pbox and _center_inside(box, pbox) and type(person.get('id')) is int:
                            candidate_role = ('cashier_candidate' if role.get('status') == 'candidate' and person['id'] == role.get('cashier_track_id')
                                              else 'visitor_candidate' if role.get('status') == 'candidate' and person['id'] in role.get('visitor_track_ids', []) else 'unknown')
                            near.append({'track_id': person['id'], 'box': pbox, 'role': candidate_role,
                                         'reflection_suspected': person['id'] in reflection_ids})
                unique_people = sorted({p['track_id'] for p in near})
                assigned = unique_people[0] if len(unique_people) == 1 and unique_people[0] not in reflection_ids else None
                tables = [item for item in context if item['label'] in {'work_table', 'table', 'desk'} and _center_inside(box, item['box'])]
                counters = [item for item in context if item['label'] == 'banknote_counter' and _overlap(box, item['box']) > 0]
                conflicts = sorted({KINDS[item['label']] for item in context if item['label'] in KINDS
                                    and KINDS[item['label']] != kind and _center_inside(box, item['box']) and _center_inside(item['box'], box)})
                relations = (['near_person'] if near else []) + (['over_table'] if tables else []) + (['near_counter'] if counters else [])
                ambiguity = []
                if len(unique_people) > 1:
                    ambiguity.append('multiple_person_projections')
                if any(p['reflection_suspected'] for p in near):
                    ambiguity.append('possible_reflection_or_duplicate_person')
                if conflicts:
                    ambiguity.append('competing_object_classes')
                if not people_usable:
                    ambiguity.append('people_journal_unavailable')
                playback = record.get('playback_sec', timestamp)
                if not _number(playback) or playback < 0:
                    playback = timestamp
                observation = {'observation_id': evidence_id, 'object_type': kind, 'label': obj['label'], 'score': float(score),
                               'timestamp_sec': timestamp, 'source_frame': frame, 'mode': mode, 'playback_sec': playback,
                               'box': box, 'relations': relations or ['unknown'], 'near_people': near,
                               'single_near_track_id': assigned, 'context': {'table_candidates': tables, 'counter_candidates': counters},
                               'ambiguity': ambiguity, 'competing_types': conflicts, 'interpretation': 'spatial_hypothesis',
                               'possession': 'unknown', 'action': 'unknown', 'record_index': index}
                observations.append(observation)
                current.append(observation)
            continuous = False
            if previous:
                delta = timestamp - previous['time']
                # A single source-frame rounding allowance, not an action threshold.
                continuous = bool(interval and fps and delta <= interval + 1 / fps + 1e-9)
                if not continuous:
                    gaps.append({'mode': mode, 'start_sec': previous['time'], 'end_sec': timestamp,
                                 'reason': 'sampling_interval_unknown_or_gap'})
            edges = defaultdict(list)
            reverse = defaultdict(list)
            if continuous:
                for old_index, old in enumerate(previous_observations):
                    for new_index, new in enumerate(current):
                        if old['object_type'] == new['object_type'] and not old['competing_types'] and not new['competing_types'] and _overlap(old['box'], new['box']) > 0:
                            edges[old_index].append(new_index)
                            reverse[new_index].append(old_index)
            for new_index, observation in enumerate(current):
                old_indices = reverse[new_index]
                old = previous_observations[old_indices[0]] if len(old_indices) == 1 and len(edges[old_indices[0]]) == 1 else None
                if old is not None:
                    sequence = sequences[old['_sequence_index']]
                    observation['_sequence_index'] = old['_sequence_index']
                else:
                    if old_indices:
                        observation['ambiguity'].append('ambiguous_object_link')
                    observation['_sequence_index'] = len(sequences)
                    sequence = {'sequence_id': _token(('sequence', observation['observation_id'])), 'object_type': observation['object_type'],
                                'title': NAMES[observation['object_type']] + ': наблюдения положения', 'status': 'hypothesis',
                                'association_method': 'same_class_mutual_unique_bbox_overlap_on_adjacent_samples',
                                'start_sec': timestamp, 'end_sec': timestamp, 'observation_ids': [], 'evidence_times': [],
                                'relations_seen': [], 'action': 'unknown', 'limitations': list(COMMON)}
                    sequences.append(sequence)
                sequence['end_sec'] = timestamp
                sequence['observation_ids'].append(observation['observation_id'])
                sequence['evidence_times'].append(_evidence(observation))
                sequence['relations_seen'] = sorted(set(sequence['relations_seen']) | set(observation['relations']))
                if old and old['single_near_track_id'] is not None and observation['single_near_track_id'] is not None and old['single_near_track_id'] != observation['single_near_track_id']:
                    hypotheses.append({'id': _token(('near_person_changed', old['observation_id'], observation['observation_id'])),
                                       'kind': 'near_person_changed', 'status': 'hypothesis', 'object_type': observation['object_type'],
                                       'title': 'Изменилось положение предмета относительно траекторий людей',
                                       'start_sec': old['timestamp_sec'], 'end_sec': timestamp,
                                       'from_track_id': old['single_near_track_id'], 'to_track_id': observation['single_near_track_id'],
                                       'sequence_id': sequence['sequence_id'], 'transfer_confirmed': False,
                                       'description': 'На соседних выбранных кадрах похожие рамки предмета связаны с разными проекциями людей. Это может быть движение людей, смена номера или передача; по этим данным причина неизвестна.',
                                       'evidence_times': [_evidence(old), _evidence(observation)], 'limitations': list(COMMON)})
            previous, previous_observations = record, current
    for observation in observations:
        observation.pop('_sequence_index', None)
    observations.sort(key=lambda row: (row['timestamp_sec'], row['mode'], row['observation_id']))
    sequences.sort(key=lambda row: (row['start_sec'], row['sequence_id']))
    hypotheses.sort(key=lambda row: (row['start_sec'], row['id']))
    counts = Counter(row['object_type'] for row in observations)
    summary = ('Наблюдения предметов собраны в проверяемые цепочки положения. Действия и нарушения по ним не подтверждаются.' if observations else
               'Нет пригодных наблюдений денег и карточек для цепочки действий. Это не означает, что предметов или передачи не было.')
    return {'version': '1.0', 'status': 'partial' if observations else 'not_checked', 'summary': summary,
            'timeline': observations, 'sequences': sequences, 'hypotheses': hypotheses, 'events': [],
            'coverage': _coverage(observations), 'counts': dict(counts), 'limitations': list(COMMON),
            'temporal_coverage': {'status': 'unknown_for_short_actions', 'sample_intervals_sec': sample_intervals,
                                  'gaps': gaps, 'people_usable': people_usable,
                                  'detail': 'Объекты наблюдаются только на выбранных кадрах. Действие между ними может остаться незамеченным даже при полном журнале людей.'},
            'data_quality': {'issues': issues, 'object_association': 'hypothesis_only', 'hand_pose_available': False,
                             'counting_action_available': False, 'identity_verification_available': False},
            'settings': {'person_relation': 'object_center_inside_person_bbox', 'table_relation': 'object_center_inside_table_bbox',
                         'counter_relation': 'positive_bbox_intersection', 'score_filter': 'none_beyond_validated_input_candidates',
                         'note': 'Пространственные условия описывают изображение. Нормативные пороги и подтверждённые действия не задаются.'}}
