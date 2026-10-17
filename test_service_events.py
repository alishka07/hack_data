"""CPU tests for positional evidence and conservative event boundaries."""
from collections import defaultdict
import copy
import json
import unittest

from service_events import analyze_service


SAFE = {'label': 'safe', 'box': [760, 350, 940, 580], 'score': .9}
TABLE = {'label': 'work_table', 'box': [200, 240, 700, 520], 'score': .8}
LAPTOP = {'label': 'laptop', 'box': [480, 330, 570, 390], 'score': .9}
CASHIER = [580, 100, 690, 500]
VISITOR = [240, 90, 350, 490]


def fixture(people_at, frames=160, objects=None):
    job = {'video': {'width': 1000, 'height': 600, 'frames': frames, 'fps': 10.}}
    by_frame, by_id = defaultdict(list), defaultdict(list)
    for frame in range(frames):
        for identity, box in people_at(frame):
            row = {'id': identity, 'frame': frame, 'time': frame / 10., 'box': box}
            by_frame[frame].append(row)
            by_id[identity].append(row)
    people = {'complete': True, 'frames': frames, 'fps': 10., 'by_frame': by_frame, 'by_id': by_id}
    modes = {}
    for mode in ('trained', 'desk'):
        records = []
        for index, frame in enumerate(range(0, frames, 5)):
            found = ([SAFE, LAPTOP] if mode == 'trained' else [TABLE]) if objects is None else objects(mode, frame)
            records.append({'time': frame / 10., 'frame': frame, 'mode': mode,
                            'playback_sec': frame / 10., 'index': index, 'objects': copy.deepcopy(found)})
        modes[mode] = {'records': records, 'times': [row['time'] for row in records], 'interval': .5}
    return modes, people, job


class ServiceEventsTests(unittest.TestCase):
    def report(self, callback, **kwargs):
        return analyze_service(*fixture(callback, **kwargs))

    def test_stable_safe_side_role_is_candidate_not_identity(self):
        result = self.report(lambda f: [(1, CASHIER), (2, VISITOR)])
        self.assertEqual(result['role_summary']['cashier_track_id'], 1)
        self.assertEqual(result['role_summary']['status'], 'candidate')
        self.assertFalse(result['role_summary']['coordinate_calibration_confirmed'])
        self.assertFalse(result['role_summary']['printer_orientation_checked'])
        self.assertEqual(result['sessions'][0]['operation_end_sec'], None)
        self.assertEqual(result['events'], [])
        json.dumps(result, ensure_ascii=False)

    def test_lone_stable_laptop_workstation_can_anchor_missing_safe(self):
        # Person is clearly on one side of the table; pose and the direction of
        # the display are intentionally not inferred from rectangles.
        laptop = dict(LAPTOP, box=[450, 300, 550, 350])
        seated = [390, 340, 510, 590]
        result = self.report(lambda f: [(10, seated)],
                             objects=lambda mode, f: [laptop] if mode == 'trained' else [TABLE])
        self.assertEqual(result['role_summary']['cashier_track_id'], 10)
        self.assertEqual(result['role_summary']['role_method'], 'lone_stable_workstation_occupant')
        self.assertFalse(result['role_summary']['seated_posture_checked'])
        self.assertFalse(result['role_summary']['laptop_screen_orientation_checked'])

    def test_workstation_fallback_requires_laptop_and_unique_occupant(self):
        seated = [390, 340, 510, 590]
        for objects, people in [
                (lambda mode, f: [] if mode == 'trained' else [TABLE], lambda f: [(1, seated)]),
                (lambda mode, f: [LAPTOP] if mode == 'trained' else [TABLE],
                 lambda f: [(1, seated), (2, [550, 340, 670, 590])])]:
            result = self.report(people, objects=objects)
            self.assertIsNone(result['role_summary']['cashier_track_id'])

    def test_excluded_reflections_remain_explained_in_service_result(self):
        modes, people, job = fixture(lambda f: [(1, CASHIER)])
        people['excluded_reflection_track_ids'] = [5]
        result = analyze_service(modes, people, job)
        self.assertEqual(result['role_summary']['excluded_reflection_track_ids'], [5])
        self.assertNotIn(5, result['role_summary']['visitor_track_ids'])

    def test_workstation_fallback_does_not_turn_late_lone_visitor_into_cashier(self):
        seated = [390, 340, 510, 590]
        result = self.report(lambda f: [] if f < 40 else [(10, seated)],
                             objects=lambda mode, f: [LAPTOP] if mode == 'trained' else [TABLE])
        self.assertIsNone(result['role_summary']['cashier_track_id'])

    def test_workstation_rejects_pos_display_and_uses_laptop_with_stable_scene(self):
        table = dict(TABLE, box=[100, 300, 400, 550])
        safe = dict(SAFE, box=[450, 350, 550, 500])
        actual_laptop = dict(LAPTOP, box=[110, 350, 170, 410])
        mislabelled_pos = dict(LAPTOP, box=[300, 200, 360, 290])
        cashier = [200, 350, 300, 570]
        def objects(mode, frame):
            if mode == 'desk':
                return [table]
            return [safe, mislabelled_pos] + ([actual_laptop] if frame in [5, 80, 85, 120] else [])
        result = self.report(lambda f: [(1, cashier)] + ([(2, [230, 150, 330, 295])] if f >= 25 else []),
                             objects=objects)
        self.assertEqual(result['role_summary']['cashier_track_id'], 1)
        self.assertEqual(result['role_summary']['role_method'], 'lone_stable_workstation_occupant')
        self.assertEqual(result['role_summary']['anchors']['laptop_box'], actual_laptop['box'])
        self.assertIn(2, result['role_summary']['visitor_track_ids'])

    def test_missing_table_blocks_roles_even_when_user_description_exists(self):
        result = self.report(lambda f: [(1, CASHIER)], objects=lambda mode, f: [SAFE] if mode == 'trained' else [])
        self.assertEqual(result['status'], 'not_checked')
        self.assertIsNone(result['role_summary']['cashier_track_id'])
        self.assertEqual(result['events'], [])

    def test_unstable_safe_blocks_arbitrary_other_camera(self):
        def objects(mode, frame):
            safe = dict(SAFE, box=[20, 350, 200, 580]) if frame > 80 else SAFE
            return [safe] if mode == 'trained' else [TABLE]
        result = self.report(lambda f: [(1, CASHIER)], objects=objects)
        self.assertEqual(result['status'], 'not_checked')

    def test_short_table_gaps_interpolate_only_with_visible_stable_safe(self):
        def objects(mode, frame):
            return [SAFE] if mode == 'trained' else [TABLE] if frame % 15 == 0 else []
        result = self.report(lambda frame: [(1, CASHIER), (2, VISITOR)], objects=objects)
        self.assertEqual(result['role_summary']['cashier_track_id'], 1)
        self.assertEqual(result['events'], [])

    def test_long_table_gaps_remain_unknown(self):
        def objects(mode, frame):
            return [SAFE] if mode == 'trained' else [TABLE] if frame % 30 == 0 else []
        result = self.report(lambda frame: [(1, CASHIER), (2, VISITOR)], objects=objects)
        self.assertIsNone(result['role_summary']['cashier_track_id'])
        self.assertEqual(result['cashier_departures'], [])

    def test_both_anchors_missing_cannot_validate_each_other(self):
        def objects(mode, frame):
            return ([SAFE] if mode == 'trained' else [TABLE]) if frame % 15 == 0 else []
        result = self.report(lambda frame: [(1, CASHIER), (2, VISITOR)], objects=objects)
        self.assertIsNone(result['role_summary']['cashier_track_id'])

    def test_diagonal_ambiguous_anchors_do_not_invent_side(self):
        centered = dict(SAFE, box=[390, 330, 510, 430])
        result = self.report(lambda f: [(1, CASHIER)], objects=lambda mode, f: [centered] if mode == 'trained' else [TABLE])
        self.assertEqual(result['status'], 'not_checked')

    def test_two_cashier_side_people_from_start_are_ambiguous(self):
        result = self.report(lambda f: [(1, CASHIER), (2, [710, 100, 790, 500])])
        self.assertIsNone(result['role_summary']['cashier_track_id'])
        self.assertEqual(result['events'], [])

    def test_stationary_opposite_people_not_classed_as_reflections(self):
        result = self.report(lambda f: [(1, CASHIER), (2, VISITOR)])
        self.assertEqual(result['role_summary']['reflection_candidate_pairs'], [])

    def test_standing_visitor_at_far_table_edge_has_visible_session(self):
        # Torso centre is far from the table; the lower bbox edge is only 5px
        # away. Person 3 is much farther up in the background/glass.
        result = self.report(lambda frame: [(1, CASHIER), (2, [240, 20, 350, 235]),
                                            (3, [50, 20, 100, 120])])
        self.assertEqual(result['role_summary']['cashier_track_id'], 1)
        self.assertEqual({session['track_id'] for session in result['sessions']}, {2})
        self.assertTrue(result['sessions'][0]['sustained'])
        self.assertIsNone(result['sessions'][0]['operation_end_sec'])
        self.assertEqual(result['events'], [])

    def test_mirror_motion_blocks_role_without_deleting_tracks(self):
        def people(frame):
            offset = min(frame, 100) * .5
            return [(1, [580 + offset, 100, 680 + offset, 500]),
                    (2, [320 - offset, 100, 420 - offset, 500])]
        result = self.report(people)
        self.assertEqual(result['role_summary']['reflection_candidate_pairs'], [[1, 2]])
        self.assertIsNone(result['role_summary']['cashier_track_id'])
        self.assertEqual(result['events'], [])

    def test_concurrent_visitors_are_not_automatic_violation(self):
        result = self.report(lambda f: [(1, CASHIER), (2, VISITOR), (3, [360, 90, 420, 490])])
        self.assertTrue(result['sessions'])
        self.assertEqual(result['events'], [])

    def test_observed_later_entry_generates_only_review_candidate(self):
        def people(frame):
            rows = [(1, CASHIER), (2, VISITOR)]
            if frame >= 80:
                x = min((frame - 80) * 25, 150)
                rows.append((3, [x, 100, x + 80, 490]))
            return rows
        result = self.report(people)
        events = [event for event in result['events'] if event['rule_id'] == 'one_client_at_a_time']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['status'], 'review')
        self.assertFalse(events[0]['context']['operation_completion_known'])
        self.assertTrue(all(evidence['mode'] == 'people' for evidence in events[0]['evidence_times']))

    def test_short_visit_does_not_generate_arrival_event(self):
        def people(frame):
            rows = [(1, CASHIER), (2, VISITOR)]
            if 80 <= frame <= 92:
                x = min((frame - 80) * 50, 270)
                rows.append((3, [x, 100, x + 80, 490]))
            return rows
        result = self.report(people)
        self.assertEqual(result['events'], [])

    def test_cashier_disappears_behind_furniture_not_departure(self):
        result = self.report(lambda frame: [(2, VISITOR)] + ([(1, CASHIER)] if frame < 90 else []))
        self.assertEqual(result['events'], [])
        self.assertEqual(result['cashier_departures'], [])

    def test_edge_departure_with_visitor_remaining_can_flag_assets(self):
        def people(frame):
            rows = [(2, VISITOR)]
            if frame <= 90:
                offset = max(0, frame - 80) * 31
                rows.append((1, [580 + offset, 100, 690 + offset, 500]))
            return rows
        result = self.report(people)
        events = [event for event in result['events'] if event['rule_id'] == 'unattended_assets']
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['context']['visitors_remain'])
        self.assertEqual(events[0]['context']['cashier_track_id'], 1)
        self.assertEqual(len(result['cashier_departures']), 1)
        self.assertFalse(result['cashier_departures'][0]['operation_completion_known'])
        self.assertTrue(result['cashier_departures'][0]['no_visible_replacement'])

    def test_replacement_in_cashier_zone_blocks_unattended_event(self):
        def people(frame):
            rows = [(2, VISITOR)]
            if frame <= 90:
                offset = max(0, frame - 80) * 31
                rows.append((1, [580 + offset, 100, 690 + offset, 500]))
            elif frame >= 91:
                rows.append((3, CASHIER))
            return rows
        result = self.report(people)
        self.assertFalse(any(event['rule_id'] == 'unattended_assets' for event in result['events']))
        self.assertEqual(result['cashier_departures'], [])

    def test_money_candidate_inside_counter_is_not_left_money(self):
        def people(frame):
            rows = [(2, VISITOR)]
            if frame <= 90:
                offset = max(0, frame - 80) * 31
                rows.append((1, [580 + offset, 100, 690 + offset, 500]))
            return rows
        counter = {'label': 'banknote_counter', 'box': [420, 330, 490, 390], 'score': .9}
        money = {'label': 'cash', 'box': [430, 340, 455, 360], 'score': .9}
        result = self.report(people, objects=lambda mode, f: [SAFE, counter] if mode == 'trained' else [TABLE, money])
        self.assertEqual(result['events'], [])

    def test_incomplete_people_journal_never_supports_role_absence(self):
        modes, people, job = fixture(lambda frame: [(1, CASHIER)])
        people['complete'] = False
        result = analyze_service(modes, people, job)
        self.assertEqual(result['status'], 'not_checked')
        self.assertEqual(result['events'], [])

    def test_visitor_crosses_to_safe_side_while_cashier_present(self):
        def people(frame):
            x = 240 if frame < 60 else min(240 + (frame - 60) * 20, 700)
            return [(1, CASHIER), (2, [x, 90, x + 70, 490])]
        result = self.report(people)
        events = [event for event in result['events'] if event['rule_id'] == 'restricted_access_card_transfer']
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]['context']['card_transfer_checked'])


if __name__ == '__main__':
    unittest.main()
