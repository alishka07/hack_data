"""CPU invariants: reviewed regions cannot become blanket exclusion masks."""
import copy
import unittest

from reflection_tracks import analyze_reflections, filter_reflections
from test_service_events import fixture


REGION = {'region_id': 'reviewed_glass', 'source': 'reviewed_reflection_region',
          'box': [25, 100, 125, 185], 'polygon': [[25, 100], [125, 100], [125, 185], [25, 185]]}
REFLECTION = [40, 115, 85, 165]
PHYSICAL = [300, 250, 450, 550]


def scenes(frames=160, matched=True):
    return [{'source_frame': f, 'timestamp_sec': f / 10., 'scene': {'matched': matched,
             'regions': {'reflections': [copy.deepcopy(REGION)]}}} for f in range(0, frames, 2)]


class ReflectionTests(unittest.TestCase):
    def evaluate(self, callback, regions=None, frames=160):
        _, people, job = fixture(callback, frames=frames)
        return analyze_reflections(people, job, scenes(frames) if regions is None else regions), people

    def test_local_reflection_is_separate_from_real_person(self):
        report, original = self.evaluate(lambda f: [(51, REFLECTION), (109, PHYSICAL)])
        self.assertEqual(report['excluded_track_ids'], [51])
        filtered = filter_reflections(original, report)
        self.assertEqual(filtered['counts'], [1] * 160)
        self.assertEqual(set(filtered['by_id']), {109})
        self.assertEqual(set(original['by_id']), {51, 109})
        self.assertEqual(report['tracks'][0]['label'], 'Отражение (кандидат)')

    def test_ids_are_not_a_rule(self):
        report, _ = self.evaluate(lambda f: [(2, PHYSICAL), (876, REFLECTION)])
        self.assertEqual(report['excluded_track_ids'], [876])

    def test_real_entrant_crosses_region_and_is_retained(self):
        report, original = self.evaluate(lambda f: [(3, [40 + f, 115, 85 + f, 165]), (7, PHYSICAL)])
        self.assertEqual(report['excluded_track_ids'], [])
        self.assertEqual(report['observation_exclusions'], [])
        self.assertEqual(filter_reflections(original, report)['counts'], [2] * 160)

    def test_small_real_person_without_physical_counterpart_retained(self):
        report, _ = self.evaluate(lambda f: [(2, REFLECTION)])
        self.assertEqual(report['excluded_track_ids'], [])

    def test_unmatched_scene_retains_tracks(self):
        report, _ = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], scenes(matched=False))
        self.assertEqual(report['excluded_track_ids'], [])
        self.assertEqual(report['status'], 'not_checked')

    def test_no_context_does_not_invent_reflection_region(self):
        report, _ = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], [])
        self.assertEqual(report['excluded_track_ids'], [])

    def test_short_track_retained(self):
        report, _ = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], frames=18)
        self.assertEqual(report['excluded_track_ids'], [])

    def test_missing_registration_for_most_track_retains(self):
        report, _ = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], scenes()[:20])
        self.assertEqual(report['excluded_track_ids'], [])

    def test_large_person_in_glass_retained(self):
        report, _ = self.evaluate(lambda f: [(2, [20, 50, 140, 300]), (1, PHYSICAL)])
        self.assertEqual(report['excluded_track_ids'], [])

    def test_unreviewed_mask_cannot_exclude_person(self):
        regions = scenes()
        for row in regions:
            row['scene']['regions']['reflections'][0]['source'] = 'guessed_glass'
        report, _ = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], regions)
        self.assertEqual(report['excluded_track_ids'], [])

    def test_track_identity_switch_outside_region_blocks_global_removal(self):
        report, original = self.evaluate(lambda f: [(2, REFLECTION if f < 100 else PHYSICAL), (1, PHYSICAL)])
        self.assertEqual(report['excluded_track_ids'], [])
        self.assertEqual(report['partially_excluded_track_ids'], [2])
        filtered = filter_reflections(original, report)
        self.assertEqual([r['frame'] for r in filtered['by_id'][2]], list(range(100, 160)))
        self.assertEqual(filtered['excluded_reflection_track_ids'], [])

    def test_registration_loss_does_not_remove_tail_of_id(self):
        regions = scenes()
        for row in regions:
            if row['source_frame'] >= 144:
                row['scene']['matched'] = False
        report, original = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], regions)
        # A 90% registered majority previously deleted all 160 observations.
        self.assertEqual(report['excluded_track_ids'], [])
        self.assertEqual(report['partially_excluded_track_ids'], [2])
        filtered = filter_reflections(original, report)
        self.assertEqual(filtered['counts'][144:], [2] * 16)
        self.assertEqual(filtered['counts'][:140], [1] * 140)
        self.assertEqual(len(original['by_id'][2]), 160)

    def test_internal_registration_gap_splits_exclusions(self):
        regions = scenes()
        for row in regions:
            if 60 <= row['source_frame'] <= 70:
                row['scene']['matched'] = False
        report, original = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], regions)
        intervals = report['observation_exclusions']
        self.assertEqual(len(intervals), 2)
        self.assertLess(intervals[0]['end_frame'], 60)
        self.assertGreater(intervals[1]['start_frame'], 70)
        self.assertEqual(filter_reflections(original, report)['counts'][60:71], [2] * 11)

    def test_open_or_unknown_glass_keeps_person(self):
        regions = scenes()
        for row in regions:
            if row['source_frame'] >= 144:
                row['scene']['regions']['reflections'] = []
        report, original = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], regions)
        self.assertEqual(filter_reflections(original, report)['counts'][144:], [2] * 16)
        self.assertEqual(report['excluded_track_ids'], [])

    def test_missing_companion_keeps_nonqualifying_observations(self):
        report, original = self.evaluate(lambda f: [(2, REFLECTION)] + ([(1, PHYSICAL)] if f < 144 else []))
        filtered = filter_reflections(original, report)
        self.assertEqual([r['frame'] for r in filtered['by_id'][2]], list(range(144, 160)))
        self.assertEqual(filtered['counts'], [1] * 160)
        self.assertEqual(report['tracks'][0]['retained_observations'], 16)

    def test_single_nonqualifying_frame_is_never_bridged(self):
        report, original = self.evaluate(lambda f: [(2, PHYSICAL if f == 80 else REFLECTION), (1, PHYSICAL)])
        self.assertEqual(len(report['observation_exclusions']), 2)
        filtered = filter_reflections(original, report)
        self.assertEqual([r['frame'] for r in filtered['by_id'][2]], [80])
        self.assertEqual(filtered['excluded_reflection_track_ids'], [])

    def test_track_detection_gaps_cannot_join_short_runs(self):
        report, original = self.evaluate(lambda f: [(1, PHYSICAL)] + ([(2, REFLECTION)] if f % 20 != 19 else []))
        self.assertEqual(report['observation_exclusions'], [])
        self.assertEqual(len(filter_reflections(original, report)['by_id'][2]), 152)

    def test_short_registration_islands_do_not_accumulate_evidence(self):
        regions = scenes()
        for row in regions:
            if row['source_frame'] % 20 == 18:
                row['scene']['matched'] = False
        report, _ = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)], regions)
        self.assertEqual(report['observation_exclusions'], [])

    def test_segment_filter_is_authoritative_over_summary_ids(self):
        _, original = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)])
        report = {'excluded_track_ids': [2], 'observation_exclusions': [
            {'track_id': 2, 'start_frame': 20, 'end_frame': 40, 'classification': 'reflection_candidate'}]}
        filtered = filter_reflections(original, report)
        self.assertEqual(filtered['counts'][:20], [2] * 20)
        self.assertEqual(filtered['counts'][20:41], [1] * 21)
        self.assertEqual(filtered['counts'][41:], [2] * 119)
        self.assertEqual(filtered['excluded_reflection_track_ids'], [])

    def test_empty_or_invalid_segment_list_never_uses_global_ids(self):
        _, original = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)])
        for segments in ([], [{'track_id': 2, 'start_frame': 50, 'end_frame': 20,
                               'classification': 'reflection_candidate'}]):
            filtered = filter_reflections(original, {'excluded_track_ids': [2], 'observation_exclusions': segments})
            self.assertEqual(filtered['counts'], [2] * 160)

    def test_legacy_id_only_report_keeps_supported_compatibility(self):
        _, original = self.evaluate(lambda f: [(2, REFLECTION), (1, PHYSICAL)])
        filtered = filter_reflections(original, {'excluded_track_ids': [2]})
        self.assertEqual(filtered['counts'], [1] * 160)
        self.assertEqual(filtered['excluded_reflection_track_ids'], [2])

    def test_invalid_journal_is_not_promoted_by_filter(self):
        report = analyze_reflections({'complete': False}, {}, scenes())
        self.assertFalse(filter_reflections({'complete': False}, report)['complete'])


if __name__ == '__main__':
    unittest.main()
