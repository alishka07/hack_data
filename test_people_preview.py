"""CPU end-to-end encoding and cache/failure preservation checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from people_preview import prepare_people_preview, existing_people_preview, _labels, _draw_frame, _font


class PeoplePreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        (self.folder / 'people').mkdir()
        (self.folder / 'people/review.csv').write_text('test validated via loader mock', encoding='utf-8')
        self.source = self.folder / 'input.mp4'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'mp4v'), 10., (160, 120))
        for frame in range(12):
            writer.write(np.full((120, 160, 3), 60 + frame, np.uint8))
        writer.release()
        self.job = {'_source_file': 'input.mp4', 'video': {'width': 160, 'height': 120, 'frames': 12, 'fps': 10.}}
        self.people = {'complete': True, 'frames': 12, 'fps': 10., 'by_frame':
                       {i: [{'frame': i, 'time': i / 10., 'id': 1, 'box': [-5.2, 10.4, 100.6, 130.2]}] for i in range(12)}}
        self.report = {'service': {'role_summary': {'status': 'candidate', 'cashier_track_id': 1, 'visitor_track_ids': [2]}},
                       'reflections': {'excluded_track_ids': [5]}}
        self.patch_loader = patch('rule_engine._load_people', return_value=self.people)
        self.patch_loader.start()

    def tearDown(self):
        self.patch_loader.stop()
        self.temp.cleanup()

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_render_source_frames_and_cache_then_invalidate_role(self):
        original = self.digest(self.source)
        result = prepare_people_preview(self.folder, self.job, self.report)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['frames'], 12)
        target = existing_people_preview(self.folder, self.job, self.report)
        self.assertIsNotNone(target)
        self.assertEqual(self.digest(self.source), original)
        cached = prepare_people_preview(self.folder, self.job, self.report)
        self.assertTrue(cached['cached'])
        report = copy.deepcopy(self.report)
        report['reflections']['excluded_track_ids'] = [1, 5]
        self.assertIsNone(existing_people_preview(self.folder, self.job, report))
        self.assertEqual(len(list((self.folder / 'people').glob('roles.encoding.webm'))), 0)

    def test_cancel_keeps_previous_preview_and_manifest(self):
        first = prepare_people_preview(self.folder, self.job, self.report)
        self.assertEqual(first['status'], 'completed')
        target = Path(first['path'])
        old = self.digest(target)
        old_manifest = target.with_suffix('.json').read_bytes()
        report = copy.deepcopy(self.report)
        report['reflections']['excluded_track_ids'] = [1]
        count = [0]
        def cancelled():
            count[0] += 1
            return count[0] > 5
        result = prepare_people_preview(self.folder, self.job, report, cancelled=cancelled)
        self.assertEqual(result['status'], 'cancelled', result)
        self.assertEqual(self.digest(target), old)
        self.assertEqual(target.with_suffix('.json').read_bytes(), old_manifest)

    def test_truncated_source_does_not_publish_or_replace_previous(self):
        first = prepare_people_preview(self.folder, self.job, self.report)
        self.assertEqual(first['status'], 'completed')
        target = Path(first['path'])
        old = self.digest(target)
        invalid_job = copy.deepcopy(self.job)
        invalid_job['video']['frames'] = 20
        result = prepare_people_preview(self.folder, invalid_job, self.report)
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(self.digest(target), old)

    def test_lightweight_lookup_detects_csv_or_file_tamper(self):
        result = prepare_people_preview(self.folder, self.job, self.report)
        self.assertEqual(result['status'], 'completed')
        (self.folder / 'people/review.csv').write_text('changed', encoding='utf-8')
        self.assertIsNone(existing_people_preview(self.folder, self.job, self.report))

    def test_incomplete_journal_does_not_render(self):
        with patch('rule_engine._load_people', return_value={'complete': False, 'reason': 'incomplete'}):
            result = prepare_people_preview(self.folder, self.job, self.report)
        self.assertEqual(result['status'], 'unavailable')
        self.assertFalse((self.folder / 'people/roles.webm').exists())

    def test_reflection_label_takes_priority_over_role_and_clip_is_finite(self):
        report = copy.deepcopy(self.report)
        report['reflections']['excluded_track_ids'] = [1]
        label = _labels(report)
        self.assertIn('Отражение', label(1)[0])
        self.assertIn('Посетитель', label(2)[0])
        self.assertIn('Человек', label(9)[0])
        frame = np.zeros((120, 160, 3), np.uint8)
        rows = self.people['by_frame'][0] + [{'id': 9, 'box': [float('nan'), 1, 5, 10]}]
        result = _draw_frame(frame, rows, label, _font(12), 0.)
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(np.any(result))

    def test_frame_scoped_reflection_label_keeps_surviving_cashier(self):
        report = copy.deepcopy(self.report)
        report['reflections'] = {'excluded_track_ids': [1], 'observation_exclusions': [
            {'track_id': 1, 'start_frame': 2, 'end_frame': 6, 'classification': 'reflection_candidate'}]}
        label = _labels(report)
        self.assertIn('Кассир', label(1, 1)[0])
        self.assertIn('Отражение', label(1, 2)[0])
        self.assertIn('Отражение', label(1, 6)[0])
        self.assertIn('Кассир', label(1, 7)[0])
        self.assertIn('Кассир', label(1)[0])
        calls = []
        def observed(identity, frame):
            calls.append((identity, frame))
            return label(identity, frame)
        _draw_frame(np.zeros((120, 160, 3), np.uint8), self.people['by_frame'][7], observed, _font(12), .7)
        self.assertEqual(calls, [(1, 7)])

    def test_partial_reflection_preview_keeps_all_source_frames_and_manifest(self):
        self.report['reflections'] = {'excluded_track_ids': [], 'partially_excluded_track_ids': [1],
                                     'observation_exclusions': [
            {'track_id': 1, 'start_frame': 2, 'end_frame': 6, 'classification': 'reflection_candidate'}]}
        result = prepare_people_preview(self.folder, self.job, self.report)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['frames'], 12)
        self.assertEqual(result['partially_excluded_reflection_track_ids'], [1])
        self.assertEqual(result['reflection_observation_exclusions'], self.report['reflections']['observation_exclusions'])


if __name__ == '__main__':
    unittest.main()
