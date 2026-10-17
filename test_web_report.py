"""Evidence must match the source frame, and exports must preserve limitations."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from web_report import save_rule_report


class ReportTests(unittest.TestCase):
    def test_exact_frame_export_and_escaped_title(self):
        with tempfile.TemporaryDirectory(prefix='rule_evidence_') as temporary:
            folder = Path(temporary)
            writer = cv2.VideoWriter(str(folder / 'input.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 64))
            self.assertTrue(writer.isOpened())
            for value in [10, 50, 100, 150, 220]:
                writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
            writer.release()
            report = {'events': [{'source_frame': 3, 'rule_id': 'test', 'start_sec': .6, 'end_sec': 1,
                                  'title': '<script>unsafe</script>', 'limitations': ['Проверить человеком'],
                                  'steps': [{'title': 'Первый кадр', 'timestamp_sec': .6, 'detail': '<script>observed</script>'}]}],
                      'rule_coverage': [{'title': 'Дверь', 'status': 'not_checked', 'detail': 'Нет модели состояния',
                                         'procedure_status': 'planned', 'procedure_steps': ['<script>planned</script>']}],
                      'summary': 'Возможный эпизод, не подтверждение'}
            save_rule_report(folder, {'id': 'a' * 32, 'name': '<video>.avi'}, report)
            picture = cv2.imread(str(folder / 'report' / report['events'][0]['evidence_file']))
            self.assertAlmostEqual(float(picture.mean()), 150, delta=4)
            page = (folder / 'report/review.html').read_text(encoding='utf-8')
            self.assertNotIn('<script>', page)
            self.assertIn('&lt;script&gt;', page)
            self.assertIn('Нет модели состояния', page)
            self.assertIn('Что произошло по шагам', page)
            self.assertIn('ещё не реализовано', page)
            self.assertIn('&lt;script&gt;observed', page)
            saved = json.loads((folder / 'report/violations.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['events'][0]['limitations'], ['Проверить человеком'])
            import csv
            with (folder / 'report/violations.csv').open(encoding='utf-8-sig', newline='') as stream:
                exported = next(csv.DictReader(stream))
            self.assertEqual(json.loads(exported['steps_json']), saved['events'][0]['steps'])

    def test_no_source_does_not_invent_evidence(self):
        with tempfile.TemporaryDirectory(prefix='rule_no_source_') as temporary:
            report = {'events': [{'source_frame': 4, 'title': 'Признак'}], 'rule_coverage': []}
            save_rule_report(Path(temporary), {'id': 'b' * 32}, report)
            self.assertNotIn('thumbnail_url', report['events'][0])


if __name__ == '__main__':
    unittest.main()
