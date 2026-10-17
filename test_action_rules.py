"""Action evidence tests: departures, occlusion, reflection and projected motion."""
from collections import defaultdict
import copy
import json
import unittest

from action_rules import analyze_actions

FPS = 10.
FRAMES = 160
TABLE = {'label': 'work_table', 'box': [200, 240, 700, 520], 'score': .9}
COUNTER = {'label': 'banknote_counter', 'box': [450, 290, 490, 350], 'score': .9}
CASHIER = [580, 100, 690, 500]
VISITOR = [240, 90, 350, 490]
CASH = {'label': 'cash', 'box': [410, 380, 440, 400], 'score': .9}
DOC = {'label': 'document', 'box': [450, 400, 490, 450], 'score': .9}


def fixture(people_at=None, objects_at=None, frames=FRAMES, sessions=None):
    if people_at is None:
        def people_at(frame):
            rows = [(1, CASHIER)]
            if frame <= 80:
                x = max(0, 240 - max(0, frame - 70) * 24)
                rows.append((2, [x, 90, x + 110, 490]))
            return rows
    objects_at = objects_at or (lambda frame: [TABLE, CASH])
    by_frame, by_id = defaultdict(list), defaultdict(list)
    for frame in range(frames):
        for identity, box in people_at(frame):
            row = {'id': identity, 'frame': frame, 'time': frame / FPS, 'box': list(box)}
            by_frame[frame].append(row)
            by_id[identity].append(row)
    people = {'complete': True, 'frames': frames, 'fps': FPS, 'by_frame': by_frame, 'by_id': by_id}
    records = [{'time': f / FPS, 'frame': f, 'index': i, 'mode': 'desk', 'playback_sec': f / FPS,
                'objects': copy.deepcopy(objects_at(f))} for i, f in enumerate(range(0, frames, 5))]
    modes = {'desk': {'records': records, 'times': [r['time'] for r in records], 'interval': .5,
                      'summary': {'groups': ['desk', 'cash', 'workspace']}}}
    service = {'role_summary': {'status': 'candidate', 'cashier_track_id': 1, 'visitor_track_ids': [2],
                                'anchors': {'table_box': TABLE['box'], 'safe_box': [760, 350, 940, 580]},
                                'reflection_candidate_pairs': []},
               'sessions': sessions if sessions is not None else [{'track_id': 2, 'role': 'visitor_candidate',
                            'start_sec': 0., 'end_sec': 7.2, 'sustained': True, 'reflection_ambiguous': False}],
               'cashier_departures': []}
    job = {'video': {'width': 1000, 'height': 600, 'fps': FPS, 'frames': frames}}
    return modes, people, service, job


def events(result, rule):
    return [row for row in result['events'] if row['rule_id'] == rule]


def coverage(result, rule):
    return next(row for row in result['coverage'] if row['rule_id'] == rule)


class ActionTests(unittest.TestCase):
    def test_cash_stays_after_positional_visitor_exit_not_global_count(self):
        result = analyze_actions(*fixture())
        found = events(result, 'cash_after_client')
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['context']['visitor_track_id'], 2)
        self.assertFalse(found[0]['context']['global_people_count_used'])
        self.assertFalse(found[0]['context']['operation_completion_known'])
        self.assertEqual(found[0]['status'], 'review')
        self.assertTrue(any(row['source_frame'] == 80 for row in found[0]['evidence_times']))
        json.dumps(result, ensure_ascii=False)

    def test_no_candidate_still_checked_for_supported_scope(self):
        result = analyze_actions(*fixture(objects_at=lambda f: [TABLE]))
        self.assertEqual(events(result, 'cash_after_client'), [])
        self.assertEqual(coverage(result, 'cash_after_client')['status'], 'partial')
        check = next(row for row in result['checks'] if row['check_id'] == 'cash_stays_after_visitor')
        self.assertEqual(check['status'], 'analyzed')
        self.assertEqual(check['reason_code'], 'no_candidates')

    def test_no_exit_still_available_does_not_mean_rule_passed(self):
        result = analyze_actions(*fixture(people_at=lambda f: [(1, CASHIER), (2, VISITOR)]))
        self.assertEqual(result['visitor_departures'], [])
        self.assertEqual(coverage(result, 'cash_after_client')['status'], 'partial')
        self.assertNotIn('verdict', result)

    def test_cash_already_on_table_with_lone_cashier_does_not_invent_client_departure(self):
        args = fixture(people_at=lambda f: [(1, CASHIER)], sessions=[])
        args[2]['role_summary']['visitor_track_ids'] = []
        result = analyze_actions(*args)
        self.assertEqual(result['visitor_departures'], [])
        self.assertEqual(events(result, 'cash_after_client'), [])

    def test_chocolate_conflict_cannot_become_cash_action(self):
        result = analyze_actions(*fixture(objects_at=lambda f: [TABLE, CASH, dict(CASH, label='food', score=.5)]))
        self.assertEqual(events(result, 'cash_after_client'), [])
        self.assertFalse(any(s['kind'].startswith('cash_') for s in result['sequences']))

    def test_interior_disappearance_is_not_departure(self):
        result = analyze_actions(*fixture(people_at=lambda f: [(1, CASHIER)] + ([(2, VISITOR)] if f <= 80 else [])))
        self.assertEqual(events(result, 'cash_after_client'), [])
        self.assertEqual(result['rejected_departures'][0]['reason_code'], 'no_visible_edge_exit')

    def test_cash_disappears_with_client_not_left_behind(self):
        result = analyze_actions(*fixture(objects_at=lambda f: [TABLE] + ([CASH] if f <= 80 else [])))
        self.assertEqual(events(result, 'cash_after_client'), [])

    def test_gap_in_cash_at_exit_breaks_same_object_claim(self):
        result = analyze_actions(*fixture(objects_at=lambda f: [TABLE] + ([CASH] if f != 80 else [])))
        self.assertEqual(events(result, 'cash_after_client'), [])

    def test_missing_sample_at_exit_breaks_continuity(self):
        args = fixture()
        args[0]['desk']['records'] = [r for r in args[0]['desk']['records'] if r['frame'] != 80]
        result = analyze_actions(*args)
        self.assertEqual(events(result, 'cash_after_client'), [])

    def test_object_class_conflict_cannot_be_cash_event(self):
        result = analyze_actions(*fixture(objects_at=lambda f: [TABLE, CASH, dict(CASH, label='identity_card')]))
        self.assertEqual(events(result, 'cash_after_client'), [])
        self.assertGreater(result['data_quality']['object_observation_counts']['class_conflicts'], 0)

    def test_object_on_counter_false_class_does_not_count_as_cash(self):
        fake = dict(CASH, box=COUNTER['box'])
        result = analyze_actions(*fixture(objects_at=lambda f: [TABLE, COUNTER, fake]))
        self.assertEqual(events(result, 'cash_after_client'), [])

    def test_document_stays_is_observation_not_ownership(self):
        result = analyze_actions(*fixture(objects_at=lambda f: [TABLE, DOC]))
        found = events(result, 'previous_client_documents')
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]['context']['ownership_known'])

    def test_concurrent_group_blocks_individual_operation_end(self):
        args = fixture()
        for frame in range(FRAMES):
            row = {'id': 3, 'frame': frame, 'time': frame / FPS, 'box': [360, 90, 430, 490]}
            args[1]['by_frame'][frame].append(row)
            args[1]['by_id'][3].append(row)
        args[2]['sessions'].append({'track_id': 3, 'start_sec': 0., 'end_sec': 15.9, 'sustained': True})
        args[2]['role_summary']['visitor_track_ids'].append(3)
        result = analyze_actions(*args)
        self.assertEqual(events(result, 'cash_after_client'), [])
        self.assertEqual(result['rejected_departures'][0]['reason_code'], 'visitor_group_or_ongoing_visit')

    def test_document_persists_into_separate_next_visit(self):
        args = fixture(objects_at=lambda f: [TABLE, DOC])
        for frame in range(110, FRAMES):
            x = min(240, (frame - 110) * 30)
            row = {'id': 3, 'frame': frame, 'time': frame / FPS, 'box': [x, 90, x + 110, 490]}
            args[1]['by_frame'][frame].append(row)
            args[1]['by_id'][3].append(row)
        args[2]['role_summary']['visitor_track_ids'].append(3)
        args[2]['sessions'].append({'track_id': 3, 'start_sec': 11.5, 'end_sec': 15.9,
                                    'sustained': True, 'entry_motion_observed': True})
        result = analyze_actions(*args)
        found = events(result, 'previous_client_documents')
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['context']['next_visitor_track_id'], 3)
        sequence = next(row for row in result['sequences'] if row['kind'] == 'document_persists_into_next_visit')
        self.assertFalse(sequence['context']['ownership_known'])

    def test_document_hidden_between_visitors_cannot_be_same_document(self):
        args = fixture(objects_at=lambda f: [TABLE] + ([DOC] if f != 105 else []))
        args[2]['sessions'].append({'track_id': 3, 'start_sec': 11.5, 'end_sec': 15.9,
                                    'sustained': True, 'entry_motion_observed': True})
        result = analyze_actions(*args)
        self.assertEqual([row for row in result['sequences'] if row['kind'] == 'document_persists_into_next_visit'], [])

    def test_unknown_nearby_replacement_blocks_departure(self):
        args = fixture()
        for frame in range(81, 100):
            args[1]['by_frame'][frame].append({'id': 99, 'frame': frame, 'time': frame / FPS, 'box': VISITOR})
        result = analyze_actions(*args)
        self.assertEqual(events(result, 'cash_after_client'), [])
        self.assertEqual(result['rejected_departures'][0]['reason_code'], 'person_remains_or_track_replacement')

    def test_confirmed_reflection_is_excluded_from_departure_and_person_relations(self):
        args = fixture()
        for frame in range(FRAMES):
            args[1]['by_frame'][frame].append({'id': 99, 'frame': frame, 'time': frame / FPS, 'box': VISITOR})
        args[2]['role_summary'].update(excluded_reflection_track_ids=[99], reflection_candidate_pairs=[[1, 99]])
        args[1]['reflections'] = {'excluded_track_ids': [99]}
        result = analyze_actions(*args)
        self.assertEqual(len(events(result, 'cash_after_client')), 1)
        self.assertTrue(result['data_quality']['roles_usable'])

    def test_unresolved_reflection_pair_blocks_cashier_role(self):
        args = fixture()
        args[2]['role_summary']['reflection_candidate_pairs'] = [[1, 99]]
        result = analyze_actions(*args)
        self.assertEqual(result['events'], [])
        self.assertEqual(coverage(result, 'cash_after_client')['availability'], 'insufficient_evidence')

    def test_incomplete_people_cannot_be_negative_absence(self):
        args = fixture()
        args[1]['complete'] = False
        result = analyze_actions(*args)
        self.assertEqual(result['events'], [])
        self.assertEqual(coverage(result, 'cash_after_client')['status'], 'not_checked')
        self.assertEqual(coverage(result, 'cash_after_client')['implementation_status'], 'implemented_partial')

    def test_table_missing_at_exit_blocks_surface_claim(self):
        result = analyze_actions(*fixture(objects_at=lambda f: [CASH] + ([TABLE] if f < 80 else [])))
        self.assertEqual(events(result, 'cash_after_client'), [])

    def test_cashier_zone_rule_does_not_require_assets(self):
        def people_at(frame):
            if frame > 80:
                return []
            x = 580 + max(0, frame - 70) * 31
            return [(1, [x, 100, x + 110, 500])]
        args = fixture(people_at=people_at, objects_at=lambda f: [TABLE], sessions=[])
        args[2]['cashier_departures'] = [{'cashier_track_id': 1, 'role_status': 'candidate',
                                         'people_decode_complete': True, 'no_visible_replacement': True,
                                         'start_sec': 8.1, 'end_sec': 15.9}]
        result = analyze_actions(*args)
        found = events(result, 'unattended_room')
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]['context']['asset_presence_required'])
        self.assertFalse(found[0]['context']['whole_room_visibility_confirmed'])

    def test_unqualified_cashier_departure_no_zone_event(self):
        args = fixture()
        args[2]['cashier_departures'] = [{'cashier_track_id': 1, 'role_status': 'candidate',
                                         'start_sec': 8.1, 'end_sec': 15.9}]
        self.assertEqual(events(analyze_actions(*args), 'unattended_room'), [])

    def test_declared_departure_with_cashier_still_visible_is_rejected(self):
        args = fixture()
        args[2]['cashier_departures'] = [{'cashier_track_id': 1, 'role_status': 'candidate',
                                         'people_decode_complete': True, 'no_visible_replacement': True,
                                         'start_sec': 8.1, 'end_sec': 15.9}]
        self.assertEqual(events(analyze_actions(*args), 'unattended_room'), [])

    def test_cash_move_then_settle_is_only_surface_projection(self):
        def objects_at(frame):
            x = 300 + min(frame // 5, 8) * 15
            return [TABLE, dict(CASH, box=[x, 380, x + 30, 400])]
        args = fixture(people_at=lambda f: [(1, CASHIER), (2, VISITOR)], objects_at=objects_at)
        result = analyze_actions(*args)
        found = [s for s in result['sequences'] if s['kind'] == 'cash_settled_over_table']
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]['context']['surface_contact_confirmed'])
        self.assertEqual(events(result, 'cash_placed_on_counter'), [])

    def test_stationary_cash_and_moving_people_not_handoff(self):
        def people_at(frame):
            return [(1, [380, 100, 480, 500]), (2, VISITOR)] if frame < 40 else [(1, CASHIER), (2, [380, 90, 480, 490])]
        result = analyze_actions(*fixture(people_at=people_at, objects_at=lambda f: [TABLE, CASH, COUNTER]))
        self.assertEqual([s for s in result['sequences'] if s['kind'] == 'cash_cashier_to_visitor'], [])

    def test_cash_approaches_and_leaves_counter_not_proof_of_counting(self):
        def objects_at(frame):
            x = 390 + min(frame // 5, 12) * 10
            return [TABLE, COUNTER, dict(CASH, box=[x, 310, x + 20, 330])]
        result = analyze_actions(*fixture(people_at=lambda f: [(1, CASHIER), (2, VISITOR)], objects_at=objects_at))
        found = [s for s in result['sequences'] if s['kind'] == 'cash_counter_approach_leave']
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]['context']['counting_confirmed'])
        self.assertFalse(found[0]['context']['client_line_of_sight_confirmed'])
        self.assertEqual(events(result, 'cash_counted_visible'), [])

    def test_static_cash_over_counter_is_not_counter_passage(self):
        result = analyze_actions(*fixture(objects_at=lambda f: [TABLE, COUNTER, dict(CASH, box=[455, 310, 475, 330])]))
        self.assertEqual([s for s in result['sequences'] if s['kind'] == 'cash_counter_approach_leave'], [])

    def test_cash_return_direction_can_be_reviewed_without_asserting_return_semantics(self):
        def objects_at(frame):
            x = 620 - min(frame // 5, 21) * 15
            return [TABLE, COUNTER, dict(CASH, box=[x, 380, x + 30, 400])]
        args = fixture(people_at=lambda f: [(1, CASHIER), (2, VISITOR)], objects_at=objects_at)
        result = analyze_actions(*args)
        found = events(result, 'change_through_counter')
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]['context']['return_semantics_known'])
        self.assertFalse(found[0]['context']['handoff_confirmed'])

    def test_handoff_class_gap_cannot_be_interpolated(self):
        def objects_at(frame):
            x = 620 - min(frame // 5, 21) * 15
            return [TABLE, COUNTER] + ([dict(CASH, box=[x, 380, x + 30, 400])] if frame != 50 else [])
        result = analyze_actions(*fixture(people_at=lambda f: [(1, CASHIER), (2, VISITOR)], objects_at=objects_at))
        self.assertEqual(events(result, 'change_through_counter'), [])

    def test_card_presentation_does_not_claim_authenticity_or_violation(self):
        def objects_at(frame):
            x = 300 + min(frame // 5, 21) * 15
            return [TABLE, dict(CASH, label='identity_card', box=[x, 380, x + 30, 400])]
        result = analyze_actions(*fixture(people_at=lambda f: [(1, CASHIER), (2, VISITOR)], objects_at=objects_at))
        found = [s for s in result['sequences'] if s['kind'] == 'card_visitor_to_cashier']
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]['context']['original_id_confirmed'])
        self.assertFalse(found[0]['context']['ownership_known'])
        self.assertEqual(events(result, 'original_id_verification'), [])

    def test_missing_counter_is_explicit_partial_input_failure(self):
        result = analyze_actions(*fixture())
        row = coverage(result, 'cash_counted_visible')
        self.assertEqual(row['availability'], 'insufficient_evidence')
        self.assertIn('counter_not_visible', row['reason_codes'])

    def test_invalid_video_returns_serializable_insufficient_evidence(self):
        result = analyze_actions({}, {}, {}, {'video': {}})
        self.assertEqual(result['events'], [])
        self.assertEqual(result['status'], 'not_checked')
        json.dumps(result)

    def test_timestamp_corruption_blocks_entire_mode(self):
        args = fixture()
        args[0]['desk']['records'][3]['time'] = 99.
        result = analyze_actions(*args)
        self.assertEqual(result['events'], [])
        self.assertEqual(result['data_quality']['issues'][0]['reason_code'], 'invalid_temporal_journal')


if __name__ == '__main__':
    unittest.main()
