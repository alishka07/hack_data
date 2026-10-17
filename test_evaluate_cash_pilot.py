"""CPU-only regression checks for the cash pilot's evaluation contract."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import evaluate_cash_pilot as pilot


class CashPilotTests(unittest.TestCase):
    def setUp(self):
        self.data = {'images': [{'id': 1, 'width': 64, 'height': 48},
                                {'id': 2, 'width': 64, 'height': 48}],
                     'categories': [{'id': 1, 'name': 'cash'}],
                     'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                      'bbox': [2, 3, 8, 10], 'area': 80, 'iscrowd': 0}]}
        self.item = {'id': 1, 'width': 64, 'height': 48, 'roi_xyxy': [10, 20, 74, 68]}

    def prediction(self, image_id=1, score=.8, bbox=None):
        return {'image_id': image_id, 'category_id': 1, 'score': score,
                'bbox': bbox or [2, 3, 8, 10]}

    def test_full_frame_cash_is_clipped_and_translated_to_roi(self):
        rows = [{'label': 'banknotes', 'score': .7, 'box': [8, 22, 22, 35]}]
        self.assertEqual(pilot.cash_predictions(rows, self.item, 1, 'gdino_fullframe'),
                         [{'image_id': 1, 'category_id': 1, 'score': .7, 'bbox': [0., 2., 12., 13.]}])

    def test_overlap_without_inside_center_does_not_count(self):
        rows = [{'label': 'money', 'score': .7, 'box': [0, 22, 12, 35]}]
        self.assertEqual(pilot.cash_predictions(rows, self.item, 1, 'gdino_fullframe'), [])

    def test_roi_clipping_and_exact_cash_alias_filter(self):
        rows = [{'label': 'banknotes money', 'score': .9, 'box': [-1, 3, 10, 13]},
                {'label': 'paper document', 'score': .99, 'box': [2, 3, 10, 13]},
                {'label': 'money paper document', 'score': .99, 'box': [2, 3, 10, 13]},
                {'label': 'money', 'score': .0009, 'box': [2, 3, 10, 13]}]
        predictions = pilot.cash_predictions(rows, self.item, 1, 'gdino_roi')
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]['bbox'], [0., 3., 10., 10.])

    def test_negative_images_report_false_positive_frames_and_boxes(self):
        predictions = [self.prediction(), self.prediction(2),
                       self.prediction(2, .7, [30, 20, 8, 10])]
        report = pilot.fixed_metrics(self.data, predictions, .5)
        self.assertEqual((report['tp'], report['fp'], report['fn']), (1, 2, 0))
        self.assertEqual(report['annotated_negative_images'], 1)
        self.assertEqual(report['negative_images_with_false_positive'], 1)
        self.assertEqual(report['false_positives_on_negative_images'], 2)

    def test_empty_predictions_preserve_false_negatives_and_negative_denominator(self):
        report = pilot.fixed_metrics(self.data, [], .5)
        self.assertEqual((report['tp'], report['fp'], report['fn']), (0, 0, 1))
        self.assertEqual(report['annotated_negative_images'], 1)
        self.assertEqual(report['negative_images_with_false_positive'], 0)
        self.assertEqual(pilot.raw_ap(self.data, [])['map_50'], 0.)

    def test_all_negative_ap_is_undefined_not_perfect_accuracy(self):
        data = {**self.data, 'annotations': []}
        self.assertIsNone(pilot.raw_ap(data, [])['map_50'])
        self.assertEqual(pilot.fixed_metrics(data, [], .5)['annotated_negative_images'], 2)

    def test_val_threshold_is_not_refitted_on_lower_scoring_test(self):
        fit = pilot.fit_validation(self.data, [self.prediction(score=.8)])
        self.assertEqual(fit['thresholds']['cash'], .8)
        with patch.object(pilot, 'fit_thresholds', side_effect=AssertionError('Test must not fit')):
            result = pilot.fixed_metrics(self.data, [self.prediction(score=.6)], fit['thresholds']['cash'])
        self.assertEqual(result['fn'], 1)
        self.assertEqual(result['tp'], 0)

    def test_low_score_cash_head_can_be_calibrated_on_validation(self):
        predictions = [self.prediction(score=.019), self.prediction(image_id=2, score=.013)]
        fit = pilot.fit_validation(self.data, predictions)
        self.assertEqual(fit['thresholds']['cash'], .019)
        self.assertEqual((fit['classes']['cash']['tp'], fit['classes']['cash']['fp']), (1, 0))
        self.assertEqual(min(pilot.THRESHOLD_GRID), .001)
        self.assertEqual(max(pilot.THRESHOLD_GRID), .9)
        self.assertEqual(len(pilot.THRESHOLD_GRID), len(set(pilot.THRESHOLD_GRID)))
        rows = [{'label': 'cash', 'score': .013, 'box': [2, 3, 10, 13]}]
        for mode in ('rtdetr_roi', 'gdino_roi'):
            self.assertEqual(len(pilot.cash_predictions(rows, self.item, 1, mode)), 1)

    def test_duplicate_sources_cannot_cross_splits(self):
        def split(video, sha, source):
            return {'images': [{'source_video': video, 'source_sha256': sha * 64,
                                'source_frame_path': source}]}
        with self.assertRaisesRegex(ValueError, 'share source_sha256'):
            pilot.check_split_leakage({'train': split('X4', 'a', 'a.jpg'),
                                      'test': split('X47', 'a', 'b.jpg')})
        with self.assertRaisesRegex(ValueError, 'share source_video'):
            pilot.check_split_leakage({'val': split('Х4', 'a', 'a.jpg'),
                                      'test': split('X4', 'b', 'b.jpg')})

    def test_prediction_inputs_exclude_ground_truth_and_reviewed_labels(self):
        image = {**self.item, 'file_name': 'roi.jpg', 'source_frame_path': '/original.jpg',
                 'source_sha256': 'a' * 64, 'source_video': 'X1',
                 'reviewed_cash_box': [1, 2, 3, 4], 'confirmed_cash': True}
        result = pilot.prediction_images({'images': [image], 'annotations': self.data['annotations']})
        self.assertNotIn('reviewed_cash_box', result[0])
        self.assertNotIn('confirmed_cash', result[0])
        self.assertNotIn('annotations', result[0])

    def make_fixture(self, directory):
        directory = Path(directory)
        root = directory / 'dataset'
        root.mkdir()
        splits = {}
        for index, name in enumerate(('train', 'val', 'test'), 1):
            source = directory / f'{name}_full.png'
            Image.new('RGB', (100, 100), color=(index * 20, 100, 80)).save(source)
            Image.new('RGB', (64, 48), color=(index * 20, 100, 80)).save(root / f'{name}.png')
            item = {**self.item, 'file_name': f'{name}.png', 'source_frame_path': str(source.resolve()),
                    'source_sha256': str(index) * 64, 'source_video': f'X{index}',
                    'complete_for_cash_only': True}
            data = {**copy.deepcopy(self.data), 'images': [item]}
            pilot.write_json(root / f'{name}.json', data)
            splits[name] = data
        return root, splits

    def test_prediction_cache_requires_unchanged_image_bytes_and_identity(self):
        with TemporaryDirectory() as temporary:
            root, splits = self.make_fixture(temporary)
            output = Path(temporary) / 'predictions.json'
            images = pilot.prediction_images(splits['val'])
            predictor = lambda image: [{'label': 'cash', 'score': .8, 'box': [2, 3, 10, 13]}]
            first = pilot.cached_inference(images, 1, root, 'rtdetr_roi', {'model': 'a'}, output, predictor)
            cached = pilot.cached_inference(images, 1, root, 'rtdetr_roi', {'model': 'a'}, output)
            self.assertEqual(first, cached)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                pilot.cached_inference(images, 1, root, 'rtdetr_roi', {'model': 'b'}, output)
            Image.new('RGB', (64, 48), 'red').save(root / 'val.png')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                pilot.cached_inference(images, 1, root, 'rtdetr_roi', {'model': 'a'}, output)

    def test_prediction_cache_payload_tampering_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root, splits = self.make_fixture(temporary)
            output = Path(temporary) / 'predictions.json'
            images = pilot.prediction_images(splits['val'])
            pilot.cached_inference(images, 1, root, 'rtdetr_roi', {}, output, lambda image: [])
            cache = pilot.read_json(output)
            cache['records'][0]['predictions'] = [self.prediction()]
            pilot.write_json(output, cache)
            with self.assertRaisesRegex(ValueError, 'payload hash'):
                pilot.cached_inference(images, 1, root, 'rtdetr_roi', {}, output)

    def test_dataset_roi_geometry_is_verified(self):
        with TemporaryDirectory() as temporary:
            root, splits = self.make_fixture(temporary)
            pilot.validate_dataset(splits['val'], root)
            splits['val']['images'][0]['roi_xyxy'][2] = 75
            with self.assertRaisesRegex(ValueError, 'dimensions differ'):
                pilot.validate_dataset(splits['val'], root)

    def test_unreviewed_empty_labels_are_not_known_negative_images(self):
        with TemporaryDirectory() as temporary:
            root, splits = self.make_fixture(temporary)
            data = splits['val']
            data['annotations'] = []
            data['images'][0].pop('complete_for_cash_only')
            with self.assertRaisesRegex(ValueError, 'missing boxes are not known negatives'):
                pilot.validate_dataset(data, root)

    def test_test_inference_occurs_after_all_thresholds_are_locked(self):
        with TemporaryDirectory() as temporary:
            root, splits = self.make_fixture(temporary)
            output = Path(temporary) / 'evaluation'
            identities = {'rtdetr_roi': {'checkpoint_sha256': 'candidate'}, 'gdino_roi': {'sha': 'baseline'}}
            events = []

            def infer(data, dataset, split, modes, destination, *args):
                events.append(split)
                if split == 'test':
                    locked = pilot.read_json(output / 'locked_thresholds.json')
                    self.assertEqual(set(locked['models']), set(identities))
                    self.assertEqual(locked['fit_split'], 'val')
                    calibration = pilot.read_json(output / 'calibration.json')
                    self.assertEqual(calibration['checkpoint_sha256'], 'candidate')
                    self.assertEqual(calibration['thresholds'], {'cash': .8})
                    self.assertEqual(calibration['fit_split'], 'val')
                    self.assertFalse(calibration['test_labels_used'])
                return {mode: [self.prediction(score=.8 if split == 'val' else .6)] for mode in modes}

            args = SimpleNamespace(dataset=root, output=output, weights=Path(temporary) / 'dummy.pth',
                                   dino_root=Path(temporary), skip_baseline=False, device='cpu')
            with patch.object(pilot, 'build_identities', return_value=identities), \
                 patch.object(pilot, 'infer_split', side_effect=infer), \
                 patch.object(pilot, 'raw_ap', return_value={'map_50': 0., 'map_50_95': 0.}), \
                 patch.dict('sys.modules', {'torch': SimpleNamespace(set_num_threads=lambda count: None)}):
                report = pilot.run(args)
            self.assertEqual(events, ['val', 'test'])
            self.assertTrue(report['thresholds_locked_before_test_inference'])
            self.assertEqual(report['models']['rtdetr_roi']['threshold'], .8)
            self.assertEqual(report['models']['rtdetr_roi']['test']['fixed_threshold']['fn'], 1)
            self.assertFalse(report['independent_ground_truth'])


if __name__ == '__main__':
    unittest.main()
