import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dense_review import (detect_roi_sample, plan_dense_windows, run_stage,
                          select_dense_samples, _select_windows)


class DenseReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'input.mov').write_bytes(b'test-video')
        self.job = {'_source_file': 'input.mov', 'video': {'width': 400, 'height': 500,
                    'duration': 20., 'fps': 30., 'frames': 600},
                    'stages': [{'key': 'desk', 'status': 'completed'}]}
        self.rows = [self.frame(t) for t in range(20)]
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def frame(t):
        return {'source_frame': t*30, 'timestamp_sec': float(t), 'raw_detections': [
            {'label': 'an office desk', 'score': .9, 'box': [70, 150, 330, 350]},
            {'label': 'banknotes', 'score': .8, 'box': [180, 220, 210, 235]}]}

    def save(self):
        (self.root/'job.json').write_text(json.dumps(self.job), encoding='utf-8')
        desk = self.root/'desk/input'
        desk.mkdir(parents=True, exist_ok=True)
        (desk/'summary.json').write_text(json.dumps({'status': 'completed', 'frame_count_matches': True,
            'signature': {'video_size': (self.root/'input.mov').stat().st_size,
                          'video_mtime': (self.root/'input.mov').stat().st_mtime_ns},
            'decoded_frames': 600,
            'image_size': [400, 500]}), encoding='utf-8')
        (desk/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in self.rows), encoding='utf-8')

    def test_plan_is_bounded_and_has_source_provenance(self):
        plan = plan_dense_windows(self.root, self.job)
        self.assertEqual(plan['status'], 'planned')
        self.assertFalse(plan['human_labels_used'])
        self.assertEqual(plan['maximum_samples'], 60)
        self.assertLessEqual(plan['planned_window_seconds'], 12)
        self.assertLessEqual(plan['window_coverage_fraction'], .6)
        self.assertEqual(len(plan['source_sha256']), 64)
        self.assertEqual(len(plan['input_files']), 2)
        self.assertEqual(plan['violations'], [])
        windows = plan['windows']
        self.assertTrue(all(a['end_sec'] <= b['start_sec'] for a, b in zip(windows, windows[1:])))

    def test_corrected_cash_is_never_an_automatic_trigger(self):
        for row in self.rows:
            row['raw_detections'] = row['raw_detections'][:1]
            row['candidates'] = [{'label': 'cash', 'score': 1, 'box': [180,220,210,235],
                                  'source': 'human_confirmed_reference_match'}]
            row['reviewed_object_corrections'] = [{'confirmed_label': 'cash'}]
        self.save()
        plan = plan_dense_windows(self.root, self.job)
        self.assertEqual(plan['status'], 'not_checked')
        self.assertEqual(plan['reason'], 'no_automatic_cash_card_or_motion_proposals')

    def test_reference_match_even_in_raw_is_excluded(self):
        for row in self.rows:
            row['raw_detections'][1]['source'] = 'human_confirmed_reference_match'
        self.save()
        self.assertEqual(plan_dense_windows(self.root, self.job)['status'], 'not_checked')

    def test_moving_table_and_multiple_tables_are_rejected(self):
        for row in self.rows[10:]:
            row['raw_detections'][0]['box'] = [0, 0, 150, 150]
        self.save()
        self.assertEqual(plan_dense_windows(self.root, self.job)['reason'], 'table_position_changed')
        self.rows = [self.frame(t) for t in range(20)]
        self.rows[5]['raw_detections'].append({'label': 'a table', 'score': .8, 'box': [0,0,50,50]})
        self.save()
        self.assertEqual(plan_dense_windows(self.root, self.job)['reason'], 'multiple_table_candidates')

    def test_incomplete_stage_is_not_checked(self):
        self.job['stages'][0]['status'] = 'partial'
        self.assertEqual(plan_dense_windows(self.root, self.job)['reason'], 'desk_stage_not_completed')

    def test_source_metadata_mismatch_is_not_checked(self):
        self.job['video']['frames'] = 601
        self.assertEqual(plan_dense_windows(self.root, self.job)['reason'], 'desk_source_metadata_mismatch')

    def test_money_outside_roi_does_not_propose_window(self):
        for row in self.rows:
            row['raw_detections'][1]['box'] = [0,450,20,465]
        self.save()
        self.assertEqual(plan_dense_windows(self.root, self.job)['status'], 'not_checked')

    def test_raw_nearby_track_motion_can_propose_review(self):
        for row in self.rows:
            row['raw_detections'] = row['raw_detections'][:1]
        self.job['stages'].append({'key': 'people', 'status': 'completed'})
        self.save()
        people = self.root/'people'
        people.mkdir()
        (people/'review.json').write_text(json.dumps({'frame_count_matches': True}), encoding='utf-8')
        (people/'review.csv').write_text('source_frame,timestamp_sec,track_id,class,x1,y1,x2,y2\n'
            '0,0,2,person,60,80,150,250\n6,0.2,2,person,110,80,200,250\n', encoding='utf-8')
        plan = plan_dense_windows(self.root, self.job)
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['windows'][0]['trigger']['reason'], 'raw_person_motion_near_table')
        self.assertEqual(plan['violations'], [])

    def test_actual_pts_half_open_and_budget(self):
        stamps = [-.03] + [i/30 for i in range(600)]
        samples = select_dense_samples(stamps, [{'start_sec':0., 'end_sec':12.}], 5, 60)
        self.assertEqual(len(samples), 60)
        self.assertEqual(samples[0]['source_frame'], 1)
        self.assertEqual(samples[-1]['timestamp_sec'], 11.8)
        self.assertTrue(all(s['timestamp_source'] == 'video_pts' for s in samples))
        self.assertTrue(all(s['timestamp_sec'] < 12 for s in samples))

    def test_variable_pts_never_infers_frame_time(self):
        stamps = [0., .07, .21, .3, .44, .5, .9]
        samples = select_dense_samples(stamps, [{'start_sec':0., 'end_sec':1.}], 5, 60)
        self.assertEqual([s['timestamp_sec'] for s in samples], [0., .21, .44, .9])
        with self.assertRaises(ValueError):
            select_dense_samples([0, 0, 1], [], 5, 60)
        with self.assertRaises(ValueError):
            select_dense_samples([0, 1, 2], [{'start_sec':0, 'end_sec':2}, {'start_sec':1, 'end_sec':3}], 5, 60)

    def test_priority_selects_movement_before_static_cash(self):
        triggers = [{'timestamp_sec': 1., 'priority':1}, {'timestamp_sec': 9., 'priority':3}]
        windows = _select_windows(triggers, 20., 3.)
        self.assertEqual(windows[0]['trigger']['timestamp_sec'], 9.)

    def test_crop_detector_preserves_original_coordinates_and_empty_frames(self):
        from PIL import Image
        class Detector:
            def detect_groups(self, image, prompts, threshold):
                self.size = image.size
                return {'targets': [{'label':'banknotes', 'score':.8, 'box':[10,20,50,40]}]}
        detector = Detector()
        sample = {'source_frame': 30, 'timestamp_sec':1., 'timestamp_source':'video_pts'}
        result = detect_roi_sample(Image.new('RGB', (400,500)), sample, [50,100,300,350], detector)
        self.assertEqual(detector.size, (250,250))
        self.assertEqual(result['raw_detections'][0]['box'], [60,120,100,140])
        self.assertEqual(result['candidates'][0]['label'], 'cash')
        self.assertEqual(result['action'], 'unknown')
        self.assertEqual(result['violations'], [])
        class Empty:
            def detect_groups(self, *args):
                return {'targets': []}
        result = detect_roi_sample(Image.new('RGB', (400,500)), sample, [50,100,300,350], Empty())
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['source_frame'], 30)
        self.assertEqual(result['interpretation'], 'unverified_closeup_object_observations')

    def test_crop_invalid_box_is_recorded_not_silently_accepted(self):
        from PIL import Image
        class Detector:
            def detect_groups(self, *args):
                return {'targets': [{'label':'banknotes', 'score':.8, 'box':[300,20,350,40]}]}
        result = detect_roi_sample(Image.new('RGB', (400,500)), {}, [50,100,300,350], Detector())
        self.assertEqual(result['candidates'], [])
        self.assertEqual(len(result['invalid_detections']), 1)

    def test_plan_only_never_loads_model_or_decodes(self):
        with patch('dense_review.inspect_source', side_effect=AssertionError('Unexpected decode')):
            result = run_stage(self.root/'input.mov', self.root, self.root/'dense', plan_only=True)
        self.assertEqual(result['status'], 'planned_only')
        self.assertFalse(result['model_loaded'])
        self.assertTrue((self.root/'dense/summary.json').is_file())

    def test_failed_decode_writes_not_checked_not_a_clean_result(self):
        with patch('dense_review.inspect_source', side_effect=ValueError('Incomplete source decode')):
            result = run_stage(self.root/'input.mov', self.root, self.root/'dense')
        self.assertEqual(result['status'], 'not_checked')
        self.assertFalse(result['all_planned_samples_analyzed'])
        self.assertEqual(result['violations'], [])

    def test_partial_inference_preserves_evidence_and_partial_status(self):
        from PIL import Image
        stamps = [i/30 for i in range(600)]
        info = {'source_dimensions': [400,500], 'frame_count_matches':True}
        class Detector:
            def detect_groups(self, *args):
                return {'targets': []}
        def frames(video, samples):
            yield samples[0], Image.new('RGB', (400,500))
            raise ValueError('Second pass decode stopped')
        with patch('dense_review.inspect_source', return_value=(stamps, info)), \
             patch('dense_review._model_signature', return_value={'fake':'test'}), \
             patch('interaction_refinement.iter_selected_frames', frames):
            result = run_stage(self.root/'input.mov', self.root, self.root/'dense', detector=Detector())
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['analyzed_samples'], 1)
        self.assertEqual(len((self.root/'dense/observations.jsonl').read_text().splitlines()), 1)

    def test_complete_injected_run_labels_test_provenance_and_saves_empty_samples(self):
        from PIL import Image
        stamps = [i/30 for i in range(600)]
        info = {'source_dimensions': [400,500], 'frame_count_matches':True}
        class Detector:
            def detect_groups(self, *args):
                return {'targets': []}
        def frames(video, samples):
            for sample in samples:
                yield sample, Image.new('RGB', (400,500))
        with patch('dense_review.inspect_source', return_value=(stamps, info)), \
             patch('dense_review._model_signature', return_value={'fake':'test'}), \
             patch('interaction_refinement.iter_selected_frames', frames), \
             patch('builtins.print'):
            result = run_stage(self.root/'input.mov', self.root, self.root/'dense', detector=Detector())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['analyzed_samples'], 60)
        self.assertEqual(result['inference_kind'], 'injected_test_detector')
        self.assertEqual(result['candidate_observations_by_class'], {})
        self.assertTrue(result['sampling_complete_for_selected_windows_only'])


if __name__ == '__main__':
    unittest.main()
