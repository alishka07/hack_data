import copy
import unittest

from interaction_events import analyze_interactions


JOB = {'video': {'fps': 10., 'frames': 100, 'width': 300, 'height': 200}}


def obj(label='cash', box=None, score=.8):
    return {'label': label, 'box': box or [40, 40, 60, 60], 'score': score}


def mode(rows, interval=.5):
    records = [{'time': time, 'frame': round(time * 10), 'mode': 'desk', 'playback_sec': time, 'objects': objects}
               for time, objects in rows]
    return {'desk': {'records': records, 'times': [r['time'] for r in records], 'interval': interval}}


def people(by_frame=None):
    return {'complete': True, 'fps': 10., 'frames': 100, 'by_frame': by_frame or {}}


def person(identity, box):
    return {'id': identity, 'box': box}


class InteractionTests(unittest.TestCase):
    def test_empty_data_is_not_a_negative_verdict(self):
        result = analyze_interactions({}, {}, {}, JOB)
        self.assertEqual(result['status'], 'not_checked')
        self.assertEqual(result['events'], [])
        self.assertEqual(result['temporal_coverage']['status'], 'unknown_for_short_actions')
        self.assertTrue(all(row['status'] == 'not_checked' for row in result['coverage']))

    def test_relations_are_only_spatial_no_counting_proof(self):
        modes = mode([(0, [obj(), obj('work_table', [0, 30, 100, 80]), obj('banknote_counter', [50, 40, 90, 80])])])
        result = analyze_interactions(modes, people({0: [person(1, [30, 0, 65, 130])]}), {}, JOB)
        row = result['timeline'][0]
        self.assertEqual(set(row['relations']), {'near_person', 'over_table', 'near_counter'})
        self.assertEqual(row['action'], 'unknown')
        self.assertEqual(row['possession'], 'unknown')
        self.assertEqual(result['events'], [])
        self.assertTrue(all(row['status'] == 'not_checked' for row in result['coverage']))

    def test_adjacent_unique_boxes_form_typed_sequence(self):
        modes = mode([(0, [obj()]), (.5, [obj(box=[42, 40, 62, 60])]), (1, [obj(box=[44, 40, 64, 60])])])
        result = analyze_interactions(modes, people(), {}, JOB)
        self.assertEqual(len(result['sequences']), 1)
        self.assertEqual(len(result['sequences'][0]['evidence_times']), 3)
        self.assertEqual(result['sequences'][0]['object_type'], 'cash')
        self.assertEqual(result['sequences'][0]['action'], 'unknown')

    def test_empty_intervening_sample_breaks_sequence(self):
        result = analyze_interactions(mode([(0, [obj()]), (.5, []), (1, [obj()])]), people(), {}, JOB)
        self.assertEqual(len(result['sequences']), 2)
        self.assertEqual(result['hypotheses'], [])

    def test_missing_sample_gap_breaks_sequence(self):
        result = analyze_interactions(mode([(0, [obj()]), (1, [obj()])]), people(), {}, JOB)
        self.assertEqual(len(result['sequences']), 2)
        self.assertEqual(len(result['temporal_coverage']['gaps']), 1)

    def test_unknown_sample_interval_cannot_link(self):
        result = analyze_interactions(mode([(0, [obj()]), (.5, [obj()])], interval=None), people(), {}, JOB)
        self.assertEqual(len(result['sequences']), 2)

    def test_two_object_matches_are_ambiguous(self):
        result = analyze_interactions(mode([(0, [obj()]), (.5, [obj(), obj(box=[45, 40, 65, 60])])]), people(), {}, JOB)
        self.assertEqual(len(result['sequences']), 3)
        self.assertIn('ambiguous_object_link', result['timeline'][1]['ambiguity'])

    def test_competing_card_and_cash_break_association(self):
        result = analyze_interactions(mode([(0, [obj()]), (.5, [obj(), obj('card')])]), people(), {}, JOB)
        self.assertEqual(len(result['sequences']), 3)
        second = [o for o in result['timeline'] if o['timestamp_sec'] == .5]
        self.assertTrue(all('competing_object_classes' in o['ambiguity'] for o in second))

    def test_unknown_people_never_become_handoff(self):
        result = analyze_interactions(mode([(0, [obj('card')]), (.5, [obj('card')])]), {'complete': False}, {}, JOB)
        self.assertEqual(result['hypotheses'], [])
        self.assertTrue(all(o['single_near_track_id'] is None for o in result['timeline']))

    def test_change_near_tracks_is_only_hypothesis(self):
        tracks = people({0: [person(1, [0, 0, 100, 150])], 5: [person(2, [0, 0, 100, 150])]})
        result = analyze_interactions(mode([(0, [obj('identity_card')]), (.5, [obj('identity_card')])]), tracks, {}, JOB)
        self.assertEqual(len(result['hypotheses']), 1)
        guess = result['hypotheses'][0]
        self.assertFalse(guess['transfer_confirmed'])
        self.assertEqual((guess['from_track_id'], guess['to_track_id']), (1, 2))
        self.assertEqual([e['source_frame'] for e in guess['evidence_times']], [0, 5])
        self.assertEqual(result['events'], [])

    def test_multiple_people_do_not_assign_owner(self):
        tracks = people({0: [person(1, [0, 0, 100, 150]), person(2, [20, 0, 120, 150])]})
        row = analyze_interactions(mode([(0, [obj()])]), tracks, {}, JOB)['timeline'][0]
        self.assertIsNone(row['single_near_track_id'])
        self.assertIn('multiple_person_projections', row['ambiguity'])

    def test_reflection_suspicion_blocks_track_assignment(self):
        tracks = people({0: [person(1, [0, 0, 100, 150])]})
        service = {'role_summary': {'reflection_candidate_pairs': [[1, 2]]}}
        row = analyze_interactions(mode([(0, [obj()])]), tracks, service, JOB)['timeline'][0]
        self.assertIsNone(row['single_near_track_id'])
        self.assertIn('possible_reflection_or_duplicate_person', row['ambiguity'])

    def test_role_is_explicitly_candidate(self):
        tracks = people({0: [person(1, [0, 0, 100, 150])]})
        service = {'role_summary': {'status': 'candidate', 'cashier_track_id': 1, 'visitor_track_ids': [2]}}
        row = analyze_interactions(mode([(0, [obj()])]), tracks, service, JOB)['timeline'][0]
        self.assertEqual(row['near_people'][0]['role'], 'cashier_candidate')

    def test_context_requires_same_original_frame(self):
        modes = mode([(0, [obj()])])
        modes['trained'] = {'records': [{'time': .1, 'frame': 1, 'objects': [obj('banknote_counter')]}], 'interval': .5}
        row = analyze_interactions(modes, people(), {}, JOB)['timeline'][0]
        self.assertNotIn('near_counter', row['relations'])

    def test_nonmonotonic_source_invalidates_whole_mode(self):
        result = analyze_interactions(mode([(.5, [obj()]), (0, [obj()])]), people(), {}, JOB)
        self.assertEqual(result['timeline'], [])
        self.assertTrue(result['data_quality']['issues'])

    def test_invalid_boxes_and_nonfinite_scores_are_ignored(self):
        result = analyze_interactions(mode([(0, [obj(box=[1, 1, 0, 0]), obj(score=float('nan'))])]), people(), {}, JOB)
        self.assertEqual(result['timeline'], [])

    def test_does_not_mutate_input_and_ids_are_deterministic(self):
        modes = mode([(0, [obj('document')]), (.5, [obj('document')])])
        original = copy.deepcopy(modes)
        a = analyze_interactions(modes, people(), {}, JOB)
        b = analyze_interactions(modes, people(), {}, JOB)
        self.assertEqual(modes, original)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
