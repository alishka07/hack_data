import copy
import math
import unittest

from interaction_refinement import (choose_auto_crop, compare_sampling, crop_box_to_source,
                                    detect_sample, resolve_crop, select_samples, sequence_diagnostics)


class RefinementTests(unittest.TestCase):
    def test_original_frame_indices_and_coarse_subset(self):
        rows = select_samples([i / 30 for i in range(100)], 1, 2, 3, 15)
        self.assertEqual([r['source_frame'] for r in rows], list(range(30, 61, 3)))
        self.assertEqual([r['source_frame'] for r in rows if r['coarse_sample']], [30, 45, 60])

    def test_fractional_start_anchors_at_first_actual_source_frame(self):
        rows = select_samples([0., .03, .065, .10, .14, .18, .21], .04, .20, 2, 4)
        self.assertEqual([(r['source_frame'], r['timestamp_sec']) for r in rows], [(2, .065), (4, .14)])
        self.assertTrue(rows[0]['coarse_sample'])
        self.assertFalse(rows[1]['coarse_sample'])

    def test_inclusive_single_frame_window(self):
        self.assertEqual(select_samples([0, .5, 1], .5, .5, 3, 15)[0]['source_frame'], 1)

    def test_negative_container_preroll_is_preserved_but_not_selected(self):
        rows = select_samples([-.0333, .0333, .0667, .1], 0, .1, 1, 3)
        self.assertEqual([r['source_frame'] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]['timestamp_sec'], .0333)

    def test_rejects_bad_timestamps_and_sampling(self):
        for stamps in [[0, 0, 1], [0, math.nan], [0, -1], [0, math.inf]]:
            with self.assertRaises(ValueError):
                select_samples(stamps, 0, 1)
        for dense, coarse in [(0, 15), (3, 14), (True, 15), (3, 0)]:
            with self.assertRaises(ValueError):
                select_samples([0, 1], 0, 1, dense, coarse)
        with self.assertRaises(ValueError):
            select_samples([0, 1], 2, 3)

    def test_crop_offset_restores_original_coordinates(self):
        box = crop_box_to_source([3.5, 4.25, 20, 15], [100, 200, 300, 400], 500, 600)
        self.assertEqual(box, [103.5, 204.25, 120, 215])

    def test_detector_outside_crop_is_clipped_before_mapping(self):
        self.assertEqual(crop_box_to_source([-4, -6, 250, 350], [100, 200, 300, 400], 500, 600), [100, 200, 300, 400])
        with self.assertRaises(ValueError):
            crop_box_to_source([250, 250, 260, 260], [100, 200, 300, 400], 500, 600)

    def test_explicit_crop_must_not_silently_clip(self):
        self.assertEqual(resolve_crop('1,2,30,40', 50, 60), [1, 2, 30, 40])
        self.assertEqual(resolve_crop('full', 50, 60), [0, 0, 50, 60])
        for value in ['-1,2,30,40', '1.5,2,30,40', '1,2,70,40', [1, 2, 1, 10]]:
            with self.assertRaises(ValueError):
                resolve_crop(value, 50, 60)

    def test_auto_ambiguity_falls_back_to_full_frame(self):
        table = {'box': [20, 20, 80, 60]}
        for anchors in [[], [[], [], []], [[table, table], [table], [table]]]:
            crop, provenance = choose_auto_crop(anchors, 100, 100)
            self.assertEqual(crop, [0, 0, 100, 100])
            self.assertEqual(provenance['status'], 'fallback_full_frame')

    def test_stable_auto_crop_has_context_margin(self):
        anchors = [[{'box': [20, 20, 80, 60]}]] * 3
        crop, provenance = choose_auto_crop(anchors, 100, 100)
        self.assertEqual(crop, [8, 12, 92, 68])
        self.assertEqual(provenance['status'], 'model_proposed')

    def test_auto_unstable_table_falls_back(self):
        anchors = [[{'box': [0, 0, 20, 20]}], [{'box': [70, 70, 90, 90]}], [{'box': [0, 0, 20, 20]}]]
        self.assertEqual(choose_auto_crop(anchors, 100, 100)[1]['status'], 'fallback_full_frame')

    def test_injected_detector_keeps_empty_frame_and_never_loads_model(self):
        from PIL import Image
        sample = {'source_frame': 60, 'timestamp_sec': 2., 'coarse_sample': True, 'timestamp_source': 'video_pts'}
        calls = []
        def empty(image, prompt, threshold):
            calls.append((image.size, prompt, threshold))
            return []
        row = detect_sample(Image.new('RGB', (200, 300)), sample, [10, 20, 100, 200], empty, {'cash': 'banknotes.'}, .28)
        self.assertEqual(row['source_frame'], 60)
        self.assertEqual(row['candidates'], [])
        self.assertEqual(row['raw_detections'], [])
        self.assertEqual(row['action'], 'unknown')
        self.assertEqual(calls[0][0], (90, 180))

    def test_detection_provenance_survives_filtering(self):
        from PIL import Image
        sample = {'source_frame': 60, 'timestamp_sec': 2., 'coarse_sample': True}
        detector = lambda image, prompt, threshold: [{'label': 'an identity card', 'box': [5, 5, 25, 15], 'score': .8}]
        row = detect_sample(Image.new('RGB', (200, 300)), sample, [10, 20, 100, 200], detector, {'desk': 'an identity card.'}, .28)
        self.assertEqual(row['raw_detections'][0]['box'], [15, 25, 35, 35])
        self.assertEqual(row['raw_detections'][0]['model_box_crop'], [5, 5, 25, 15])
        self.assertEqual(row['candidates'][0]['coordinate_space'], 'original_frame_pixels')

    def test_comparison_never_calls_extra_detections_an_accuracy_gain(self):
        records = [{'source_frame': 0, 'timestamp_sec': 0., 'coarse_sample': True, 'candidates': [], 'review': []},
                   {'source_frame': 3, 'timestamp_sec': .1, 'coarse_sample': False,
                    'candidates': [{'label': 'cash'}], 'review': []}]
        result = compare_sampling(records)
        self.assertEqual(result['dense']['frames_with_target_candidates_or_review'], 1)
        self.assertEqual(result['coarse']['frames_with_target_candidates_or_review'], 0)
        self.assertEqual(result['quality'], 'not_measured')
        self.assertEqual(result['action'], 'unknown')

    def test_spatial_diagnostics_are_not_action_proof(self):
        records = [{'source_frame': index, 'timestamp_sec': index / 30, 'coarse_sample': True,
                    'candidates': [{'label': 'cash', 'score': .8, 'box': [10, 10, 25, 25]}]}
                   for index in (0, 3, 6)]
        original = copy.deepcopy(records)
        result = sequence_diagnostics(records, {'fps': 30, 'source_frames': 100, 'width': 100, 'height': 100}, 3)
        self.assertEqual(result['sequence_count'], 1)
        self.assertEqual(result['maximum_sequence_observations'], 3)
        self.assertEqual(result['action'], 'unknown')
        self.assertEqual(records, original)


if __name__ == '__main__':
    unittest.main()
