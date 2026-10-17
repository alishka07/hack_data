"""CPU contracts for source-bound, gap-aware visible-state review events."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from state_rules import analyze_states
from rule_settings import DOOR_OPEN_THRESHOLD_SEC, DOOR_OPEN_TITLE


class StateRulesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        (self.folder / 'visual_states').mkdir()
        source = b'synthetic bytes for source-binding unit tests; no inference'
        (self.folder / 'input.mov').write_bytes(source)
        self.job = {'video': {'frames': 17, 'width': 464, 'height': 848, 'fps': 4.},
                    'stages': [{'key': 'visual_states', 'status': 'completed'}]}
        self.data = {'status': 'completed', 'frame_count_matches': True, 'decoded_frames': 17,
                     'expected_frames': 17, 'sample_seconds': .25,
                     'video': {'width': 464, 'height': 848, 'fps': 4., 'sha256': hashlib.sha256(source).hexdigest()},
                     'observations': [self.row(frame, 'open', 'closed') for frame in range(17)]}
        self.service = {'role_summary': {'status': 'unknown', 'cashier_track_id': None}, 'cashier_departures': []}

    def row(self, frame, door, safe):
        return {'frame_index': frame, 'timestamp_sec': frame / 4., 'scene': {'matched': True},
                'door': {'state': door, 'box': [10, 10, 110, 200]},
                'safe': {'state': safe, 'box': [180, 270, 260, 370]}}

    def report(self):
        (self.folder / 'visual_states/observations.json').write_text(json.dumps(self.data), encoding='utf-8')
        return analyze_states(self.folder, self.job, self.service)

    def qualified_departure(self):
        self.service = {'role_summary': {'status': 'candidate', 'cashier_track_id': 1},
                        'cashier_departures': [{'role_status': 'candidate', 'cashier_track_id': 1,
                                               'people_decode_complete': True, 'no_visible_replacement': True,
                                               'exit_sec': .75, 'start_sec': 1., 'end_sec': 4.}]}

    def test_source_mismatch_never_uses_states(self):
        self.data['video']['sha256'] = '0' * 64
        result = self.report()
        self.assertEqual(result['status'], 'not_checked')
        self.assertEqual(result['events'], [])

    def test_invalid_reflection_region_cannot_filter_people(self):
        self.data['observations'][0]['scene']['regions'] = {'reflections': [{'box': [-1, 0, 20, 30]}]}
        result = self.report()
        self.assertEqual(result['status'], 'not_checked')
        self.assertEqual(result['events'], [])

    def test_truncated_decode_never_uses_states(self):
        self.data['decoded_frames'] = 16
        self.assertEqual(self.report()['status'], 'not_checked')

    def test_missing_tail_never_claims_full_recording(self):
        self.data['observations'] = self.data['observations'][:10]
        self.assertEqual(self.report()['status'], 'not_checked')

    def test_exactly_three_point_five_observed_seconds_reaches_user_threshold(self):
        self.data['observations'] = [self.row(frame, 'open' if frame <= 14 else 'closed', 'closed') for frame in range(17)]
        events = self.report()['events']
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]['start_sec'], events[0]['end_sec']), (0., 3.5))
        self.assertEqual(events[0]['rule_id'], 'door_open')
        self.assertEqual(events[0]['context']['user_threshold_sec'], DOOR_OPEN_THRESHOLD_SEC)
        self.assertEqual(events[0]['title'], DOOR_OPEN_TITLE)

    def test_previous_two_seconds_no_longer_reaches_threshold(self):
        self.data['observations'] = [self.row(frame, 'open' if frame <= 8 else 'closed', 'closed') for frame in range(17)]
        self.assertEqual(self.report()['events'], [])

    def test_door_threshold_boundary_349_350_351(self):
        # .05-second sampling at 100 FPS permits exact hundredth-second endpoints.
        # Validated state input is used directly to isolate interval policy from decoding.
        for end, expected in [(3.49, False), (3.5, True), (3.51, True)]:
            with self.subTest(end=end):
                self.job['video'].update(fps=100., frames=401)
                times = [i / 100 for i in range(0, int(end * 100), 5)] + [end]
                rows = [self.row(0, 'open', 'closed') for _ in times]
                for row, t in zip(rows, times):
                    row.update(timestamp_sec=t, frame_index=round(t * 100))
                data = {**self.data, 'sample_seconds': .05, 'observations': rows}
                report = analyze_states(self.folder, self.job, self.service, verified_data=data)
                self.assertEqual(bool(report['events']), expected)

    def test_sample_count_does_not_round_duration_up_to_threshold(self):
        self.data['observations'] = [self.row(frame, 'open' if frame <= 13 else 'closed', 'closed') for frame in range(17)]
        self.assertEqual(self.report()['events'], [])

    def test_coverage_reports_current_user_threshold_even_without_events(self):
        self.data['observations'] = [self.row(frame, 'closed', 'closed') for frame in range(17)]
        coverage = next(row for row in self.report()['coverage'] if row['rule_id'] == 'door_open')
        self.assertEqual(coverage['user_threshold_sec'], DOOR_OPEN_THRESHOLD_SEC)
        self.assertEqual(coverage['title'], DOOR_OPEN_TITLE)

    def test_safe_departure_duration_stays_at_two_seconds(self):
        self.data['observations'] = [self.row(frame, 'closed', 'open' if frame <= 12 else 'closed') for frame in range(17)]
        self.qualified_departure()
        events = self.report()['events']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['rule_id'], 'safe_closed')
        self.assertEqual(events[0]['end_sec'] - events[0]['start_sec'], 2.)

    def test_unknown_breaks_open_interval(self):
        self.data['observations'][8]['door']['state'] = 'unknown'
        self.assertEqual(self.report()['events'], [])

    def test_missing_frame_at_low_fps_breaks_interval(self):
        self.data['observations'].pop(8)
        self.assertEqual(self.report()['events'], [])

    def test_unmatched_scene_cannot_claim_known_state(self):
        self.data['observations'][8]['scene']['matched'] = False
        result = self.report()
        self.assertEqual(result['status'], 'not_checked')
        self.assertEqual(result['events'], [])

    def test_nan_fps_is_not_valid_metadata(self):
        self.job['video']['fps'] = float('nan')
        self.assertEqual(self.report()['status'], 'not_checked')

    def test_open_safe_alone_is_not_event(self):
        self.data['observations'] = [self.row(frame, 'closed', 'open') for frame in range(17)]
        self.assertEqual(self.report()['events'], [])

    def test_open_safe_after_qualified_departure_is_review(self):
        self.data['observations'] = [self.row(frame, 'closed', 'open') for frame in range(17)]
        self.qualified_departure()
        events = self.report()['events']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['rule_id'], 'safe_closed')
        self.assertEqual(events[0]['severity'], 'review')
        self.assertIsNone(events[0]['context']['locked'])

    def test_unqualified_departure_does_not_create_safe_event(self):
        self.data['observations'] = [self.row(frame, 'closed', 'open') for frame in range(17)]
        self.qualified_departure()
        self.service['cashier_departures'][0]['no_visible_replacement'] = False
        self.assertEqual(self.report()['events'], [])

    def test_unknown_role_does_not_create_safe_event(self):
        self.data['observations'] = [self.row(frame, 'closed', 'open') for frame in range(17)]
        self.qualified_departure()
        self.service['role_summary']['status'] = 'unknown'
        self.assertEqual(self.report()['events'], [])

    def test_safe_unknown_gap_after_departure_breaks_interval(self):
        self.data['observations'] = [self.row(frame, 'closed', 'unknown' if frame == 10 else 'open') for frame in range(17)]
        self.qualified_departure()
        self.assertEqual(self.report()['events'], [])


if __name__ == '__main__':
    unittest.main()
