import json
from pathlib import Path
import tempfile
import unittest
from detect_cash_roi import calibrated_threshold, map_cash_predictions, make_experiment_index

class CashRuntimeTests(unittest.TestCase):
    def test_missing_calibration_does_not_silently_choose_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'missing.json'
            with self.assertRaises(ValueError):calibrated_threshold(path,'abc')
            self.assertEqual(calibrated_threshold(path,'abc',.21),.21)

    def test_calibration_is_bound_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calibration.json'
            path.write_text(json.dumps({'checkpoint_sha256':'abc','thresholds':{'cash':.21}}))
            self.assertEqual(calibrated_threshold(path,'abc'),.21)
            with self.assertRaises(ValueError):calibrated_threshold(path,'other')

    def test_outside_center_rejected_and_overlaps_suppressed(self):
        rows=[{'label':'cash','box':[-20,5,4,15],'score':.9},
              {'label':'cash','box':[10,10,40,30],'score':.8},
              {'label':'cash','box':[11,10,41,30],'score':.7}]
        kept=map_cash_predictions(rows,[20,280,400,520],464,848)
        self.assertEqual(len(kept),1)
        self.assertEqual(kept[0]['score'],.8)

    def test_index_preserves_previous_run_and_escapes_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['X26','X46 & demo']:
                (root/name).mkdir();(root/name/'summary.json').write_text('{}')
            make_experiment_index(root)
            html=(root/'index.html').read_text(encoding='utf-8')
            self.assertIn('X26',html);self.assertIn('X46 &amp; demo',html)
            self.assertNotIn('2 секунд',html)

if __name__=='__main__':unittest.main()
