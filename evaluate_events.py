"""Offline event evaluation. No inference, training, legal thresholds or web writes.

Legacy labels contain a video verdict and one point in time, not an event type
or a complete event list. The importer preserves those limits explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from zipfile import ZipFile
import xml.etree.ElementTree as ET

VERSION = '1.0'
UNKNOWN_TYPE = 'unknown_type'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def video_id(name):
    """Only the established X / Cyrillic Х naming convention is accepted."""
    match = re.fullmatch(r'[XХxх]-?(\d+)', Path(str(name)).stem)
    if not match:
        raise ValueError(f'Unknown video name: {name!r}')
    return 'X' + str(int(match[1]))


def parse_legacy_time(value):
    text = str(value).strip().lower().replace(',', '.')
    match = re.fullmatch(r'(?:(\d+)\s*минут[аы]?\s*)?(\d+(?:\.\d+)?)\s*секунд[аы]?', text)
    if not match:
        raise ValueError(f'Unsupported timestamp; explicit review required: {value!r}')
    seconds = float(match[2])
    if match[1] and seconds >= 60:
        raise ValueError('Seconds in a minute timestamp must be below 60')
    return int(match[1] or 0) * 60 + seconds


def read_legacy_rows(path):
    """Read the source XLSX without modifying it or interpreting formulas."""
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        sheets = [n for n in archive.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', n)]
        if len(sheets) != 1:
            raise ValueError('Expected exactly one source worksheet')
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheet_name = workbook.find('s:sheets/s:sheet', ns).attrib['name']
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        rows = []
        for row in ET.fromstring(archive.read(sheets[0])).findall('.//s:row', ns):
            cells = {}
            for cell in row.findall('s:c', ns):
                if cell.find('s:f', ns) is not None:
                    raise ValueError('Source formulas require explicit interpretation')
                value = cell.find('s:v', ns)
                inline = cell.find('s:is', ns)
                value = value.text if value is not None else ''.join(inline.itertext()) if inline is not None else ''
                if cell.attrib.get('t') == 's':
                    value = strings[int(value)]
                cells[re.sub(r'\d+', '', cell.attrib['r'])] = value
            if any(cells.values()):
                rows.append((int(row.attrib['r']), cells))
    if not rows or [rows[0][1].get(k) for k in 'ABC'] != ['наименование файла', 'таймстемп', 'нарушение / без нарушения']:
        raise ValueError('Source columns do not match the reviewed legacy format')
    return sheet_name, rows[1:]


def import_labels(workbook, videos_root, inventory=None):
    workbook, videos_root = Path(workbook), Path(videos_root)
    source_hash = sha256(workbook)
    sheet, rows = read_legacy_rows(workbook)
    files = defaultdict(list)
    for path in videos_root.iterdir():
        if path.is_file() and path.suffix.lower() in {'.mov', '.mp4', '.mkv', '.avi'}:
            try:
                files[video_id(path.name)].append(path)
            except ValueError:
                pass
    durations = {}
    if inventory:
        durations = {v['sha256']: v['decoded_duration_seconds'] for v in read_json(inventory)['videos']}
    videos = []
    seen = set()
    for row_number, row in rows:
        identity = video_id(row.get('A', ''))
        if identity in seen or len(files[identity]) != 1:
            raise ValueError(f'Ambiguous or missing source video: {identity}')
        seen.add(identity)
        path = files[identity][0]
        verdict = {'нарушение': 'violation', 'без нарушения': 'no_violation'}.get(row.get('C', '').strip().lower())
        if verdict is None:
            raise ValueError(f'Unrecognized source verdict at row {row_number}')
        provenance = {'source_file': str(workbook.resolve()), 'source_sha256': source_hash,
                      'sheet': sheet, 'range': f'A{row_number}:C{row_number}', 'raw_values': row}
        record = {'video_id': identity, 'file_name': path.name, 'source_sha256': sha256(path),
                  'split': 'existing_development', 'verdict': verdict,
                  'annotation_completeness': 'positive_events_only' if verdict == 'violation' else 'exhaustive',
                  'type_annotation': 'unknown', 'events': [], 'provenance': provenance,
                  'evaluation_exclusions': []}
        if record['source_sha256'] in durations:
            record['duration_sec'] = durations[record['source_sha256']]
        if verdict == 'violation':
            timestamp = parse_legacy_time(row.get('B', ''))
            if timestamp > record.get('duration_sec', math.inf):
                raise ValueError(f'Timestamp outside video: {identity}')
            record['events'].append({'event_id': f'{identity}_legacy_row_{row_number}', 'type': UNKNOWN_TYPE,
                                     'timestamp_sec': timestamp, 'time_meaning': 'source_point_unspecified',
                                     'provenance': provenance})
        elif row.get('B', '').strip():
            raise ValueError(f'Negative row has an unexplained timestamp: {row_number}')
        videos.append(record)
    groups = defaultdict(list)
    for record in videos:
        groups[record['source_sha256']].append(record)
    duplicates = []
    for records in groups.values():
        if len(records) > 1:
            names = [v['video_id'] for v in records]
            duplicates.append(names)
            for record in records:
                record['evaluation_exclusions'].append('duplicate_video_content_requires_label_review')
    return {'schema_version': VERSION, 'kind': 'event_ground_truth', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source': {'path': str(workbook.resolve()), 'sha256': source_hash},
            'notes': ['Legacy positive rows are not exhaustive event lists. Event types are unknown.',
                      'Existing development videos are not an independent held-out benchmark.',
                      'Duplicate byte-identical videos are excluded until their labels are reconciled.'],
            'duplicate_content_groups': duplicates, 'videos': videos}


def import_web_reports(jobs_root, ground_truth):
    """Import saved reports only. Verify both filename ID and original bytes."""
    expected = {v['video_id']: v for v in ground_truth['videos']}
    candidates = defaultdict(list)
    rejected = []
    for job_file in sorted(Path(jobs_root).glob('*/job.json')):
        report_file = job_file.parent / 'report/violations.json'
        if not report_file.is_file():
            continue
        try:
            job, report = read_json(job_file), read_json(report_file)
            identity = video_id(job['name'])
            if job['status'] not in {'completed', 'partial'}:
                raise ValueError('Job is not finished')
            if video_id(report['video']) != identity or report.get('job_id') != job['id']:
                raise ValueError('Report and job identities disagree')
            source = (job_file.parent / job['_source_file']).resolve()
            if not source.is_relative_to(job_file.parent.resolve()):
                raise ValueError('Source path escapes job folder')
            source_hash = sha256(source)
            if identity not in expected or expected[identity]['source_sha256'] != source_hash:
                raise ValueError('Uploaded source bytes do not match labeled video')
            coverage = report.get('rule_coverage', [])
            checked = [r for r in coverage if r.get('status') in {'checked', 'partial'}]
            status = 'partial' if checked else 'not_checked'
            if checked and all(r.get('status') == 'checked' for r in coverage) and job['status'] == 'completed':
                status = 'full'
            events = [{'event_id': str(event.get('id', f'event_{i}')), 'type': event['rule_id'],
                       'timestamp_sec': event['start_sec'], 'time_meaning': 'candidate_start',
                       'end_sec': event.get('end_sec')} for i, event in enumerate(report['events'])]
            # no_candidates never silently becomes no_violation.
            verdict = 'violation' if events else 'no_violation' if status == 'full' and report.get('verdict') == 'no_violation' else 'unknown'
            candidates[identity].append({'video_id': identity, 'source_sha256': source_hash, 'analysis_status': status,
                                        'verdict': verdict, 'events': events, 'rule_coverage': coverage,
                                        'provenance': {'job_id': job['id'], 'job_status': job['status'],
                                                       'updated_at': job.get('updated_at', ''), 'report_path': str(report_file.resolve()),
                                                       'report_sha256': sha256(report_file), 'source_match': 'sha256_and_filename_id'}})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append({'job_file': str(job_file), 'reason': str(exc)})
    selected = [max(records, key=lambda r: (r['provenance']['updated_at'], r['provenance']['job_id'])) for records in candidates.values()]
    return {'schema_version': VERSION, 'kind': 'event_predictions', 'output_kind': 'review_candidates',
            'selection': 'Latest saved finished report per exact video ID and SHA-256, without looking at accuracy.',
            'rejected_reports': rejected, 'videos': selected}


def _finite_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _validate(data, predicted=False):
    videos = data.get('videos')
    if not isinstance(videos, list):
        raise ValueError('videos must be a list')
    ids = set()
    for video in videos:
        identity = video['video_id']
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('Video IDs must be unique nonempty strings')
        ids.add(identity)
        if not re.fullmatch('[a-f0-9]{64}', video.get('source_sha256', '')):
            raise ValueError(f'Missing/invalid source SHA-256: {identity}')
        if video.get('verdict') not in {'violation', 'no_violation', 'unknown'}:
            raise ValueError('Unknown verdict')
        if predicted:
            if video.get('analysis_status') not in {'full', 'partial', 'not_checked', 'failed'}:
                raise ValueError('Unknown analysis status')
            if video['verdict'] == 'no_violation' and video['analysis_status'] != 'full':
                raise ValueError('A partial/failed/not_checked analysis cannot assert no_violation')
            if video['analysis_status'] in {'failed', 'not_checked'} and video['verdict'] != 'unknown':
                raise ValueError('Unchecked analysis must abstain')
        elif video.get('annotation_completeness') not in {'exhaustive', 'positive_events_only', 'unreviewed'}:
            raise ValueError('Unknown annotation completeness')
        event_ids = set()
        events = video.get('events')
        if not isinstance(events, list):
            raise ValueError('events must be a list')
        if video['verdict'] == 'no_violation' and events:
            raise ValueError('Negative verdict cannot contain positive events')
        if predicted and video['analysis_status'] in {'not_checked', 'failed'} and events:
            raise ValueError('Unchecked analysis cannot supply scored events')
        for event in events:
            event_id = event.get('event_id')
            if not isinstance(event_id, str) or not event_id or event_id in event_ids:
                raise ValueError('Event IDs must be unique nonempty strings within each video')
            event_ids.add(event_id)
            if not _finite_time(event.get('timestamp_sec')):
                raise ValueError('Invalid event time')
            if event['timestamp_sec'] > video.get('duration_sec', math.inf):
                raise ValueError('Event exceeds source duration')
            if not isinstance(event.get('type'), str) or not event['type']:
                raise ValueError('Use explicit unknown_type when type is absent')


def match_events(truth, predicted, tolerance_sec):
    """Maximum cardinality, then minimum total absolute time error (1-D DP).

    Events are matched by point timestamp only. A long predicted interval does
    not receive credit simply because it contains the reference timestamp.
    Caller separates event types when type-specific scoring is possible.
    """
    if not _finite_time(tolerance_sec):
        raise ValueError('Tolerance must be finite and nonnegative')
    actual = sorted(enumerate(truth), key=lambda pair: (pair[1]['timestamp_sec'], pair[0]))
    guesses = sorted(enumerate(predicted), key=lambda pair: (pair[1]['timestamp_sec'], pair[0]))
    n, m = len(actual), len(guesses)
    dp = [[(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)]
    back = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(dp[i-1][j], 'truth'), (dp[i][j-1], 'prediction')]
            error = abs(actual[i-1][1]['timestamp_sec'] - guesses[j-1][1]['timestamp_sec'])
            if error <= tolerance_sec:
                count, cost = dp[i-1][j-1]
                options.append(((count + 1, cost + error), 'match'))
            dp[i][j], back[i, j] = max(options, key=lambda item: (item[0][0], -item[0][1], item[1] == 'match'))
    pairs = []
    i, j = n, m
    while i and j:
        move = back[i, j]
        if move == 'match':
            pairs.append((actual[i-1][0], guesses[j-1][0]))
            i -= 1
            j -= 1
        elif move == 'truth':
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _metrics(tp, fp, fn, unverified=0, timing=None):
    timing = timing or []
    return {'tp': tp, 'fp': fp, 'fn': fn, 'unverified_predictions': unverified,
            'precision': _ratio(tp, tp + fp) if not unverified else None,
            'recall': _ratio(tp, tp + fn),
            'f1': _ratio(2 * tp, 2 * tp + fp + fn) if not unverified else None,
            'median_absolute_time_error_sec': median([abs(t) for t in timing]) if timing else None,
            'median_signed_time_error_sec': median(timing) if timing else None}


def evaluate(ground_truth, predictions, tolerance_sec=2.0):
    _validate(ground_truth)
    _validate(predictions, predicted=True)
    if not _finite_time(tolerance_sec):
        raise ValueError('Tolerance must be finite and nonnegative')
    expected = {v['video_id']: v for v in ground_truth['videos']}
    predicted = {v['video_id']: v for v in predictions['videos']}
    content_groups = defaultdict(list)
    for row in expected.values():
        content_groups[row['source_sha256']].append(row['video_id'])
    duplicate_ids = {identity for group in content_groups.values() if len(group) > 1 for identity in group}
    totals = Counter()
    verdicts = Counter()
    errors = []
    per_video = []
    typed = defaultdict(Counter)
    typed_errors = defaultdict(list)
    typed_videos = 0
    for identity, reference in expected.items():
        result = {'video_id': identity, 'reference_verdict': reference['verdict']}
        exclusions = list(reference.get('evaluation_exclusions', []))
        if identity in duplicate_ids:
            exclusions.append('duplicate_video_content_requires_label_review')
        if reference['annotation_completeness'] == 'unreviewed' or reference['verdict'] == 'unknown':
            exclusions.append('reference_unreviewed')
        if exclusions:
            result.update(status='excluded', reasons=sorted(set(exclusions)))
            totals['excluded_videos'] += 1
            per_video.append(result)
            continue
        totals['eligible_reference_videos'] += 1
        totals['eligible_known_events'] += len(reference['events'])
        guess = predicted.get(identity)
        if guess is None:
            totals['missing_predictions'] += 1
            totals['unassessed_known_events'] += len(reference['events'])
            per_video.append({**result, 'status': 'missing_prediction'})
            continue
        if guess['source_sha256'] != reference['source_sha256']:
            raise ValueError(f'Source SHA-256 mismatch for {identity}')
        if any(e['timestamp_sec'] > reference.get('duration_sec', math.inf) for e in guess['events']):
            raise ValueError(f'Predicted timestamp exceeds reference video duration: {identity}')
        totals['available_predictions'] += 1
        result['analysis_status'] = guess['analysis_status']
        result['predicted_verdict'] = guess['verdict']
        if guess['verdict'] == 'unknown':
            verdicts['abstained'] += 1
        elif reference['verdict'] == 'violation':
            verdicts['tp' if guess['verdict'] == 'violation' else 'fn'] += 1
        else:
            verdicts['fp' if guess['verdict'] == 'violation' else 'tn'] += 1
        if guess['analysis_status'] in {'not_checked', 'failed'}:
            totals['not_checked_or_failed'] += 1
            totals['unassessed_known_events'] += len(reference['events'])
            per_video.append({**result, 'status': 'not_checked'})
            continue
        totals['assessed_videos'] += 1
        actual, guesses = reference['events'], guess['events']
        pairs = match_events(actual, guesses, tolerance_sec)
        matched = len(pairs)
        remaining = len(guesses) - matched
        fp = remaining if reference['annotation_completeness'] == 'exhaustive' else 0
        unverified = remaining - fp
        differences = [guesses[j]['timestamp_sec'] - actual[i]['timestamp_sec'] for i, j in pairs]
        errors.extend(differences)
        totals.update(tp=matched, fp=fp, fn=len(actual)-matched, unverified_predictions=unverified)
        result.update(status='assessed', candidate_retrieval=_metrics(matched, fp, len(actual)-matched, unverified, differences),
                      matches=[{'truth_event_id': actual[i]['event_id'], 'predicted_event_id': guesses[j]['event_id'],
                                'signed_error_sec': guesses[j]['timestamp_sec'] - actual[i]['timestamp_sec']} for i, j in pairs])
        per_video.append(result)
        if reference.get('type_annotation') == 'complete' and reference['annotation_completeness'] == 'exhaustive' and all(e['type'] != UNKNOWN_TYPE for e in actual):
            covered_types = {r['rule_id']: r.get('status') for r in guess.get('rule_coverage', [])}
            video_type_assessed = False
            for event_type in {e['type'] for e in actual + guesses}:
                refs = [e for e in actual if e['type'] == event_type]
                preds = [e for e in guesses if e['type'] == event_type]
                default_status = 'checked' if guess['analysis_status'] == 'full' else 'partial' if preds else 'not_checked'
                rule_status = covered_types.get(event_type, 'not_checked') if covered_types else default_status
                if rule_status not in {'checked', 'partial'}:
                    typed[event_type].update(unassessed_known_events=len(refs), unscored_predictions=len(preds), not_checked_videos=1)
                    continue
                video_type_assessed = True
                typed_pairs = match_events(refs, preds, tolerance_sec)
                typed[event_type].update(tp=len(typed_pairs), fp=len(preds)-len(typed_pairs), fn=len(refs)-len(typed_pairs), assessed_videos=1)
                typed_errors[event_type].extend(preds[j]['timestamp_sec'] - refs[i]['timestamp_sec'] for i, j in typed_pairs)
            typed_videos += int(video_type_assessed)
    event_metrics = _metrics(totals['tp'], totals['fp'], totals['fn'], totals['unverified_predictions'], errors)
    event_metrics['end_to_end_known_event_recall'] = _ratio(totals['tp'], totals['eligible_known_events'])
    decision_total = sum(verdicts[key] for key in ('tp', 'tn', 'fp', 'fn'))
    return {'schema_version': VERSION, 'kind': 'event_evaluation', 'tolerance_sec': tolerance_sec,
            'tolerance_source': 'Evaluation setting only; not a bank rule or legal time threshold.',
            'scope': 'Candidate retrieval on existing development data, not independent confirmed-violation accuracy.',
            'counts': dict(totals), 'candidate_retrieval_ignoring_type': event_metrics,
            'video_verdict': {**{key: verdicts[key] for key in ('tp', 'tn', 'fp', 'fn', 'abstained')},
                              'precision': _ratio(verdicts['tp'], verdicts['tp'] + verdicts['fp']),
                              'recall_on_decided_videos': _ratio(verdicts['tp'], verdicts['tp'] + verdicts['fn']),
                              'accuracy_on_decided_videos': _ratio(verdicts['tp'] + verdicts['tn'], decision_total),
                              'decision_coverage_on_available_predictions': _ratio(decision_total, totals['available_predictions'])},
            'type_quality': {'status': 'measurable_on_explicit_subset' if typed_videos else 'not_measurable',
                             'eligible_assessed_videos': typed_videos,
                             'reason': 'Requires exhaustive reference events with type_annotation=complete; unknown types never become inferred labels.',
                             'per_type': {name: {**_metrics(row['tp'], row['fp'], row['fn'], timing=typed_errors[name]),
                                                  'assessed_videos': row['assessed_videos'],
                                                  'not_checked_videos': row['not_checked_videos'],
                                                  'unassessed_known_events': row['unassessed_known_events'],
                                                  'unscored_predictions': row['unscored_predictions']}
                                          for name, row in sorted(typed.items())}},
            'unmatched_prediction_videos': sorted(set(predicted) - set(expected)), 'per_video': per_video,
            'limitations': ['One-to-one maximum-cardinality, minimum-time-error matching of point timestamps.',
                            'Model event starts are compared to legacy reference points whose onset/end meaning is unspecified.',
                            'Unmatched predictions on positive-only labels remain unverified; aggregate precision is unavailable if any exist.',
                            'Not checked, failed and missing predictions are separate from no violation and from assessed-event FN.',
                            'End-to-end recall includes missing/not-checked known reference events; assessed recall does not.',
                            'An empty partial report is an abstention, never a correct negative.',
                            'Byte-identical videos are excluded until source labels are reconciled.']}


def russian_summary(report):
    counts, events, verdict = report['counts'], report['candidate_retrieval_ignoring_type'], report['video_verdict']
    number = lambda value: 'не измерено' if value is None else f'{value:.4f}'
    return '\n'.join([
        'Оценка событий на имеющихся записях',
        'Это проверка кандидатов на знакомых данных, а не независимая оценка подтверждённых нарушений.',
        f"Допуск сравнения таймкодов: {report['tolerance_sec']} с. Это параметр оценки, не правило банка.",
        f"Эталонных видео после исключений: {counts.get('eligible_reference_videos', 0)}; оценены по отчётам: {counts.get('assessed_videos', 0)}.",
        f"Нет отчёта: {counts.get('missing_predictions', 0)}; анализ не выполнен/ошибка: {counts.get('not_checked_or_failed', 0)}; исключены: {counts.get('excluded_videos', 0)}.",
        f"Совпавшие таймкоды: {events['tp']}; пропущенные известные отметки в оценённых видео: {events['fn']}; подтверждаемые по разметке ложные кандидаты: {events['fp']}.",
        f"Кандидаты без достаточной разметки для проверки: {events['unverified_predictions']}.",
        f"Precision: {number(events['precision'])}; recall известных отметок на оценённых видео: {number(events['recall'])}.",
        f"Медиана абсолютной ошибки таймкода: {number(events['median_absolute_time_error_sec'])} с.",
        f"Отчётов без общего заключения: {verdict['abstained']}. Пустой частичный отчёт не считается ответом «нарушений нет».",
        f"Качество типов событий: {report['type_quality']['status']}.",
        'В исходной таблице нет типов нарушений и полноты перечня событий у положительных видео.',
        'Дубли видео исключены: у X4 и X47 совпадают файлы, но отметки времени различаются; X5 и X48 также одинаковы.',
        'Для независимой оценки нужны другие видео с полной разметкой событий, их типов и временных границ.',
        ''
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    importer = sub.add_parser('import-labels')
    importer.add_argument('--workbook', type=Path, default=Path('open_labels.xlsx'))
    importer.add_argument('--videos', type=Path, default=Path('videos'))
    importer.add_argument('--inventory', type=Path, default=Path('results/inventory/inventory.json'))
    importer.add_argument('--output', type=Path, default=Path('training/event_evaluation/legacy_labels.json'))
    web = sub.add_parser('import-reports')
    web.add_argument('--jobs', type=Path, default=Path('web_data/jobs'))
    web.add_argument('--truth', type=Path, default=Path('training/event_evaluation/legacy_labels.json'))
    web.add_argument('--output', type=Path, default=Path('training/event_evaluation/web_predictions.json'))
    score = sub.add_parser('score')
    score.add_argument('--truth', type=Path, default=Path('training/event_evaluation/legacy_labels.json'))
    score.add_argument('--predictions', type=Path, default=Path('training/event_evaluation/web_predictions.json'))
    score.add_argument('--tolerance-sec', type=float, default=2.0)
    score.add_argument('--output', type=Path, default=Path('training/event_evaluation/evaluation.json'))
    args = parser.parse_args()
    if args.command == 'import-labels':
        result = import_labels(args.workbook, args.videos, args.inventory if args.inventory.exists() else None)
    elif args.command == 'import-reports':
        result = import_web_reports(args.jobs, read_json(args.truth))
    else:
        result = evaluate(read_json(args.truth), read_json(args.predictions), args.tolerance_sec)
        result['input_files'] = {key: {'path': str(getattr(args, key).resolve()), 'sha256': sha256(getattr(args, key))} for key in ('truth', 'predictions')}
    write_json(args.output, result)
    if args.command == 'score':
        args.output.with_suffix('.txt').write_text(russian_summary(result), encoding='utf-8')
    print(json.dumps({'saved': str(args.output), 'videos': len(result.get('videos', [])), 'counts': result.get('counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
