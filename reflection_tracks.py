"""Conservative reflection candidates from reviewed, registered scene regions.

Regions are evidence, not masks. An uninterrupted run must stay small and local
for at least two seconds while a larger physical person remains outside glass.
Only supported observations in that run are excluded. Registration loss, missing
frames and nonqualifying observations break runs; the remaining ID stays a person.
Original journals and observations are never modified.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import math
from statistics import median


def _box(value):
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)
            and value[2] > value[0] and value[3] > value[1])


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def _inside(point, polygon):
    if not isinstance(polygon, list) or len(polygon) < 3:
        return False
    if any(not isinstance(p, (list, tuple)) or len(p) != 2
           or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in p) for p in polygon):
        return False
    x, y = point
    result = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            result = not result
    return result


def _contained(box, region):
    return (_box(region.get('box')) and _inside(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), region.get('polygon'))
            and _intersection(box, region['box']) / _area(box) >= .70)


def exclusion_predicate(report: dict):
    """Compile frame-scoped exclusions, with an explicit legacy-ID fallback.

    When observation_exclusions exists it is authoritative, even when empty or
    invalid. A summary ID must never widen a partial segment to the whole track.
    Old reports without observation_exclusions retain their original semantics.
    """
    if 'observation_exclusions' not in report:
        legacy_ids = set(report.get('excluded_track_ids', []))
        return lambda identity, frame=None: identity in legacy_ids
    segments = defaultdict(list)
    for entry in report.get('observation_exclusions') or []:
        if not isinstance(entry, dict):
            continue
        first, last = entry.get('start_frame'), entry.get('end_frame')
        identity = entry.get('track_id')
        if (type(identity) is int and type(first) is int and type(last) is int
                and 0 <= first <= last and entry.get('classification') == 'reflection_candidate'):
            segments[identity].append((first, last))

    def excluded(identity, frame=None):
        return (type(frame) is int
                and any(first <= frame <= last for first, last in segments.get(identity, [])))

    return excluded


def analyze_reflections(people: dict, job: dict, scene_observations=None) -> dict:
    """Classify contiguous observation runs; scene rows are already validated.

    Expected each row: timestamp_sec, source_frame, scene.matched and
    scene.regions.reflections (projected reviewed polygons). The caller verifies
    the state journal's source hash. No source coordinates or IDs are assumed.
    """
    result = {'version': '1.1', 'status': 'not_checked', 'tracks': [], 'excluded_track_ids': [],
              'partially_excluded_track_ids': [], 'exclusion_scope': 'validated_contiguous_observations',
              'observation_exclusions': [], 'source': 'reviewed_region_and_track_evidence',
              'limitations': ['Отражение — кандидат по проверенной области и траектории, а не распознавание личности.',
                              'Исключаются только непрерывные подтверждённые участки; потеря привязки и спорные кадры сохраняются как люди.',
                              'Вся стеклянная перегородка не маскируется. Новые ракурсы и движущиеся через область люди сохраняются.']}
    if not people.get('complete'):
        result['reason'] = 'Нет полного проверенного журнала людей.'
        return result
    metadata = job.get('video', {})
    width, height = metadata.get('width', 0), metadata.get('height', 0)
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (width, height)):
        result['reason'] = 'Нет геометрии исходного видео.'
        return result
    rows = sorted((r for r in (scene_observations or []) if isinstance(r, dict)
                   and isinstance(r.get('timestamp_sec'), (int, float)) and math.isfinite(r['timestamp_sec'])),
                  key=lambda row: row['timestamp_sec'])
    times = [r['timestamp_sec'] for r in rows]

    def checked_regions(row):
        scene = row.get('scene', {})
        if scene.get('matched') is not True:
            return []
        return [r for r in scene.get('regions', {}).get('reflections', [])
                if r.get('source') == 'reviewed_reflection_region' and _box(r.get('box'))]

    def region_key(region):
        return region.get('view_id'), region.get('region_id', 'reviewed')

    def regions_at(timestamp):
        index = bisect_left(times, timestamp)
        if index < len(rows) and abs(times[index] - timestamp) < 1e-6:
            return checked_regions(rows[index])
        if index == 0 or timestamp - times[index - 1] > .3 + 1e-6:
            return []
        before = checked_regions(rows[index - 1])
        # Do not bridge into a sampled registration loss, open door or other view.
        if index < len(rows) and times[index] - timestamp <= .3 + 1e-6:
            following_keys = {region_key(r) for r in checked_regions(rows[index])}
            before = [r for r in before if region_key(r) in following_keys]
        return before

    for identity, track in people.get('by_id', {}).items():
        ordered = sorted(track, key=lambda r: r.get('frame', -1))
        valid = [r for r in ordered if _box(r.get('box'))]
        matched, inside, companions, regions_seen = 0, 0, 0, set()
        runs, current, current_region = [], [], None

        def finish_run():
            nonlocal current, current_region
            if current:
                runs.append((current, current_region))
            current, current_region = [], None

        for row in ordered:
            if not _box(row.get('box')):
                finish_run()
                continue
            regions = regions_at(row['time'])
            if not regions:
                finish_run()
                continue
            matched += 1
            box = row['box']
            region = next((r for r in regions if _contained(box, r)), None)
            if not region or _area(box) > width * height * .025 or box[3] - box[1] > height * .22:
                finish_run()
                continue
            inside += 1
            regions_seen.add(region.get('region_id', 'reviewed'))
            # A small isolated person behind glass must not disappear merely
            # because its bbox is inside a reviewed reflective area.
            outside = [other for other in people.get('by_frame', {}).get(row['frame'], [])
                       if other.get('id') != identity and _box(other.get('box'))
                       and _area(other['box']) >= _area(box) * 2.5
                       and (other['box'][1] + other['box'][3]) / 2 > region['box'][3] + height * .06
                       and not any(_contained(other['box'], r) for r in regions)]
            if outside:
                companions += 1
                key = region_key(region)
                if current and (row['frame'] != current[-1]['frame'] + 1 or key != current_region
                                or row['time'] <= current[-1]['time']):
                    finish_run()
                current.append(row)
                current_region = key
            else:
                finish_run()
        finish_run()

        segments = []
        for observations, key in runs:
            if len(observations) < 8 or observations[-1]['time'] - observations[0]['time'] < 2. - 1e-6:
                continue
            centers = [((r['box'][0] + r['box'][2]) / 2, (r['box'][1] + r['box'][3]) / 2)
                       for r in observations]
            # Scale local-motion limits to the candidate, not only the image.
            # Otherwise a real small person can traverse most of a glass region
            # while satisfying the old image-wide localization threshold.
            max_x = min(width * .09, median(r['box'][2] - r['box'][0] for r in observations) * .75)
            max_y = min(height * .07, median(r['box'][3] - r['box'][1] for r in observations) * .75)
            if (max(p[0] for p in centers) - min(p[0] for p in centers) >= max_x
                    or max(p[1] for p in centers) - min(p[1] for p in centers) >= max_y):
                continue
            segments.append({'track_id': identity, 'start_frame': observations[0]['frame'],
                             'end_frame': observations[-1]['frame'],
                             'start_sec': observations[0]['time'], 'end_sec': observations[-1]['time'],
                             'observations': len(observations), 'classification': 'reflection_candidate',
                             'view_id': key[0], 'region_id': key[1]})
        duration = valid[-1]['time'] - valid[0]['time'] if len(valid) > 1 else 0.
        excluded_count = sum(s['observations'] for s in segments)
        exclude = bool(ordered) and excluded_count == len(ordered)
        partial = bool(segments) and not exclude
        classification = ('reflection_candidate' if exclude else 'mixed_person_and_reflection' if partial
                          else 'person_or_unknown')
        detail = ('Устойчивая небольшая траектория в проверенной отражающей области при видимом человеке вне неё.' if exclude
                  else 'Исключены только подтверждённые участки отражения; остальные наблюдения сохранены как человек.' if partial
                  else 'Недостаточно непрерывных согласованных признаков отражения; траектория сохранена как человек.')
        result['tracks'].append({'track_id': identity, 'classification': classification,
                                 'excluded_from_people_rules': exclude,
                                 'partially_excluded_from_people_rules': partial,
                                 'label': 'Отражение (кандидат)' if exclude else 'Человек / отражение по участкам' if partial else 'Человек', 'reason': detail,
                                 'source_frames': [valid[0]['frame'], valid[-1]['frame']] if valid else [],
                                 'duration_sec': round(duration, 3), 'observations': len(valid),
                                 'excluded_observations': excluded_count, 'retained_observations': len(ordered) - excluded_count,
                                 'reflection_segments': len(segments),
                                 'registered_observations': matched, 'inside_region_observations': inside,
                                 'coexisting_person_observations': companions, 'region_ids': sorted(regions_seen)})
        result['observation_exclusions'].extend(segments)
        if exclude:
            result['excluded_track_ids'].append(identity)
        elif partial:
            result['partially_excluded_track_ids'].append(identity)
    result['status'] = 'partial' if any(r.get('scene', {}).get('matched') is True and
                                      r.get('scene', {}).get('regions', {}).get('reflections') for r in rows) else 'not_checked'
    result['reason'] = ('Проверенные отражающие области сопоставлены с траекториями.' if result['status'] == 'partial'
                        else 'Нет сопоставленных проверенных областей отражения для этого ракурса.')
    return result


def filter_reflections(people: dict, report: dict) -> dict:
    """Return a separate, structurally compatible journal; preserve original."""
    if not people.get('complete'):
        return {**people, 'reflections': report}
    excluded_at = exclusion_predicate(report)
    by_frame = defaultdict(list)
    by_id = defaultdict(list)
    counts = [0] * people['frames']
    for frame, rows in people.get('by_frame', {}).items():
        unique = []
        for row in rows:
            if excluded_at(row.get('id'), frame):
                continue
            by_frame[frame].append(row)
            by_id[row['id']].append(row)
            def duplicate(other):
                overlap = _intersection(row['box'], other['box'])
                return row['id'] == other['id'] or overlap / max(1e-9, _area(row['box']) + _area(other['box']) - overlap) >= .7
            if not any(duplicate(other) for other in unique):
                unique.append(row)
        counts[frame] = len(unique)
    excluded = set(people.get('by_id', {})) - set(by_id)
    partial = {identity for identity, rows in by_id.items()
               if len(rows) < len(people.get('by_id', {}).get(identity, []))}
    return {**people, 'by_frame': by_frame, 'by_id': by_id, 'counts': counts, 'reflections': report,
            'raw_track_count': len(people.get('by_id', {})), 'excluded_reflection_track_ids': sorted(excluded),
            'partially_excluded_reflection_track_ids': sorted(partial)}
