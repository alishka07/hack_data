import copy
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_events import evaluate, import_labels, import_web_reports, match_events, parse_legacy_time, sha256, video_id, write_json


def event(identity, seconds, kind='unknown_type'):
    return {'event_id': identity, 'timestamp_sec': seconds, 'type': kind}


def reference(events=None, verdict='violation', complete='exhaustive', kind='unknown'):
    return {'videos': [{'video_id': 'X1', 'source_sha256': 'a' * 64, 'verdict': verdict,
                        'annotation_completeness': complete, 'type_annotation': kind,
                        'events': events or []}]}


def predictions(events=None, verdict='violation', status='full'):
    return {'videos': [{'video_id': 'X1', 'source_sha256': 'a' * 64, 'verdict': verdict,
                        'analysis_status': status, 'events': events or []}]}


class EventEvaluationTests(unittest.TestCase):
    def test_legacy_names_and_minutes(self):
        self.assertEqual(video_id('Х-17.mov'), 'X17')
        self.assertEqual(video_id('X-1.mov'), 'X1')
        self.assertEqual(parse_legacy_time('1минута 8 секунда'), 68)
        self.assertEqual(parse_legacy_time('22 секунда'), 22)
        with self.assertRaises(ValueError):
            parse_legacy_time('10 or 20 seconds')

    def test_duplicate_prediction_is_not_multiple_tp(self):
        result = evaluate(reference([event('a', 5)]), predictions([event('p', 5), event('q', 5)]))
        metric = result['candidate_retrieval_ignoring_type']
        self.assertEqual((metric['tp'], metric['fp'], metric['fn']), (1, 1, 0))
        self.assertEqual(metric['precision'], .5)

    def test_maximum_cardinality_over_nearest_greedy(self):
        actual = [event('a', 1), event('b', 2)]
        guessed = [event('p', 0), event('q', 1)]
        self.assertEqual(len(match_events(actual, guessed, 1)), 2)

    def test_minimum_timing_error_among_maximum_matchings(self):
        pairs = match_events([event('a', 5)], [event('p', 3), event('q', 5), event('r', 7)], 2)
        self.assertEqual(pairs, [(0, 1)])

    def test_matching_matches_bruteforce_small_cases(self):
        # Independent exhaustive assignment checks both optimization objectives.
        for ts, ps in [([1, 2, 5], [0, 1, 4]), ([0, 3, 8], [1, 4, 7, 8]), ([3, 3], [2, 3, 5])]:
            a = [event(str(i), t) for i, t in enumerate(ts)]
            p = [event(str(i), t) for i, t in enumerate(ps)]
            best = (0, 0)
            for k in range(1, min(len(ts), len(ps)) + 1):
                for ii in itertools.combinations(range(len(ts)), k):
                    for jj in itertools.permutations(range(len(ps)), k):
                        differences = [abs(ts[i] - ps[j]) for i, j in zip(ii, jj)]
                        if max(differences) <= 2:
                            best = max(best, (k, -sum(differences)))
            pairs = match_events(a, p, 2)
            self.assertEqual((len(pairs), -sum(abs(ts[i] - ps[j]) for i, j in pairs)), best)

    def test_partial_annotation_does_not_invent_fp(self):
        result = evaluate(reference([event('a', 5)], complete='positive_events_only'),
                          predictions([event('p', 5), event('q', 40)]))
        metric = result['candidate_retrieval_ignoring_type']
        self.assertEqual((metric['tp'], metric['fp'], metric['unverified_predictions']), (1, 0, 1))
        self.assertIsNone(metric['precision'])
        self.assertEqual(result['type_quality']['status'], 'not_measurable')

    def test_explicit_negative_can_supply_false_positives(self):
        result = evaluate(reference(verdict='no_violation'), predictions([event('p', 9)]))
        self.assertEqual(result['candidate_retrieval_ignoring_type']['fp'], 1)
        self.assertEqual(result['video_verdict']['fp'], 1)

    def test_not_checked_is_abstention_not_negative_or_fn(self):
        result = evaluate(reference([event('a', 5)]), predictions(verdict='unknown', status='not_checked'))
        self.assertEqual(result['candidate_retrieval_ignoring_type']['fn'], 0)
        self.assertIsNone(result['candidate_retrieval_ignoring_type']['recall'])
        self.assertEqual(result['candidate_retrieval_ignoring_type']['end_to_end_known_event_recall'], 0)
        self.assertEqual(result['counts']['unassessed_known_events'], 1)
        self.assertEqual(result['video_verdict']['abstained'], 1)

    def test_empty_partial_report_is_not_true_negative(self):
        result = evaluate(reference(verdict='no_violation'), predictions(verdict='unknown', status='partial'))
        self.assertEqual(result['video_verdict']['tn'], 0)
        self.assertEqual(result['video_verdict']['abstained'], 1)

    def test_missing_reports_do_not_fake_evaluation(self):
        result = evaluate(reference([event('a', 5)]), {'videos': []})
        self.assertEqual(result['counts']['missing_predictions'], 1)
        self.assertEqual(result['candidate_retrieval_ignoring_type']['fn'], 0)

    def test_source_hash_mismatch_rejected(self):
        p = predictions()
        p['videos'][0]['source_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            evaluate(reference(), p)

    def test_invalid_partial_negative_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(reference(verdict='no_violation'), predictions(verdict='no_violation', status='partial'))

    def test_bad_timestamp_and_tolerance_rejected(self):
        for bad in [-1, math.nan, math.inf, True]:
            with self.assertRaises(ValueError):
                evaluate(reference([event('a', bad)]), predictions())
            with self.assertRaises(ValueError):
                evaluate(reference(), predictions(), bad)

    def test_interval_containment_cannot_hide_wrong_timestamp(self):
        p = event('p', 0)
        p['end_sec'] = 100
        result = evaluate(reference([event('a', 50)]), predictions([p]), 2)
        self.assertEqual(result['candidate_retrieval_ignoring_type']['tp'], 0)

    def test_exact_tolerance_boundary_and_signed_error(self):
        result = evaluate(reference([event('a', 5)]), predictions([event('p', 7)]), 2)
        self.assertEqual(result['candidate_retrieval_ignoring_type']['median_signed_time_error_sec'], 2)

    def test_explicit_typed_exhaustive_subset(self):
        result = evaluate(reference([event('a', 5, 'door_open')], kind='complete'),
                          predictions([event('p', 5, 'safe_open')]))
        self.assertEqual(result['candidate_retrieval_ignoring_type']['tp'], 1)
        per_type = result['type_quality']['per_type']
        self.assertEqual(per_type['door_open']['fn'], 1)
        self.assertEqual(per_type['safe_open']['fp'], 1)

    def test_per_rule_not_checked_does_not_become_typed_fn(self):
        pred = predictions(verdict='unknown', status='partial')
        pred['videos'][0]['rule_coverage'] = [{'rule_id': 'door_open', 'status': 'not_checked'}]
        result = evaluate(reference([event('a', 5, 'door_open')], kind='complete'), pred)
        metric = result['type_quality']['per_type']['door_open']
        self.assertEqual(metric['fn'], 0)
        self.assertEqual(metric['unassessed_known_events'], 1)
        self.assertIsNone(metric['recall'])

    def test_prediction_time_must_fit_verified_source(self):
        truth = reference([event('a', 5)])
        truth['videos'][0]['duration_sec'] = 10
        with self.assertRaisesRegex(ValueError, 'exceeds reference'):
            evaluate(truth, predictions([event('p', 11)]))

    def test_import_preserves_raw_provenance_and_excludes_identical_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / 'labels.xlsx'
            workbook.write_bytes(b'placeholder; rows are independently mocked')
            video_root = root / 'videos'
            video_root.mkdir()
            (video_root / 'X-1.mov').write_bytes(b'same video')
            (video_root / 'X-2.mov').write_bytes(b'same video')
            rows = [(2, {'A': 'X-1', 'B': '5 секунда', 'C': 'нарушение'}),
                    (3, {'A': 'X-2', 'B': '', 'C': 'без нарушения'})]
            before = sha256(workbook)
            with patch('evaluate_events.read_legacy_rows', return_value=('Sheet1', rows)):
                result = import_labels(workbook, video_root)
            self.assertEqual(sha256(workbook), before)
            self.assertEqual(result['videos'][0]['events'][0]['type'], 'unknown_type')
            self.assertEqual(result['videos'][0]['provenance']['raw_values']['B'], '5 секунда')
            self.assertEqual(result['videos'][0]['annotation_completeness'], 'positive_events_only')
            self.assertEqual(result['duplicate_content_groups'], [['X1', 'X2']])

    def test_web_import_requires_exact_bytes_and_preserves_abstention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'job1'
            folder.mkdir()
            source = folder / 'input.mov'
            source.write_bytes(b'original video')
            truth = reference([event('a', 5)])
            truth['videos'][0]['source_sha256'] = sha256(source)
            write_json(folder / 'job.json', {'id': 'job1', 'name': 'X-1.mov', 'status': 'completed',
                                            'updated_at': '2026-10-17T00:00:00+00:00', '_source_file': 'input.mov'})
            write_json(folder / 'report/violations.json', {'job_id': 'job1', 'video': 'X-1.mov',
                        'verdict': 'no_candidates', 'events': [],
                        'rule_coverage': [{'rule_id': 'door_open', 'status': 'partial'}]})
            imported = import_web_reports(root, truth)
            self.assertEqual(imported['videos'][0]['analysis_status'], 'partial')
            self.assertEqual(imported['videos'][0]['verdict'], 'unknown')
            source.write_bytes(b'another video')
            rejected = import_web_reports(root, truth)
            self.assertEqual(rejected['videos'], [])
            self.assertEqual(len(rejected['rejected_reports']), 1)

    def test_duplicate_content_is_never_double_counted(self):
        truth = reference([event('a', 5)])
        other = copy.deepcopy(truth['videos'][0])
        other['video_id'] = 'X2'
        truth['videos'].append(other)
        result = evaluate(truth, predictions([event('p', 5)]))
        self.assertEqual(result['counts']['excluded_videos'], 2)
        self.assertEqual(result['candidate_retrieval_ignoring_type']['tp'], 0)


if __name__ == '__main__':
    unittest.main()
