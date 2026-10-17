"""Small checks for the data errors that can silently invalidate training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from train_objects import check_data, classification_initialization


class TrainingDataChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ["a.jpg", "b.jpg"]:
            (self.root / name).write_bytes(b"exists; image decode is validated by the dataset loader")
        self.train = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100, "source_video": "train.mov"}],
                      "categories": [{"id": 7, "name": "safe"}],
                      "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [5, 5, 40, 50]}]}
        self.val = copy.deepcopy(self.train)
        self.val["images"][0].update(file_name="b.jpg", source_video="val.mov")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        for name, value in [("train.json", self.train), ("val.json", self.val)]:
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        return check_data(self.root / "train.json", self.root / "val.json", self.root)

    def test_arbitrary_coco_category_id_and_video_split(self):
        report = self.check()
        self.assertEqual(report["train_boxes"], {"safe": 1})
        self.assertTrue(report["video_split_verified"])
        self.assertEqual(report['source_group_split_checks'], {'source_sha256': False, 'source_group': False})

    def test_rejects_duplicate_source_hash_under_different_video_names(self):
        self.train['images'][0]['source_sha256'] = 'a' * 64
        self.val['images'][0]['source_sha256'] = 'A' * 64
        with self.assertRaisesRegex(ValueError, 'share source_sha256'):
            self.check()

    def test_requires_declared_hash_on_every_image(self):
        self.train['images'][0]['source_sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'All images must declare nonempty source_sha256'):
            self.check()

    def test_rejects_invalid_hash_instead_of_claiming_verified_split(self):
        self.train['images'][0]['source_sha256'] = 'not-a-sha256'
        self.val['images'][0]['source_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'Invalid source_sha256'):
            self.check()

    def test_rejects_overlapping_source_group_even_with_distinct_hashes(self):
        for data, digest in ((self.train, 'a'), (self.val, 'b')):
            data['images'][0].update(source_sha256=digest * 64, source_group='same-recording')
        with self.assertRaisesRegex(ValueError, 'share source_group'):
            self.check()

    def test_requires_group_on_every_image_when_only_validation_declares_it(self):
        self.val['images'][0]['source_group'] = 'validation-group'
        with self.assertRaisesRegex(ValueError, 'All images must declare nonempty source_group'):
            self.check()

    def test_distinct_source_groups_are_reported_as_checked(self):
        self.train['images'][0].update(source_sha256='a' * 64, source_group='train-group')
        self.val['images'][0].update(source_sha256='b' * 64, source_group='val-group')
        self.assertEqual(self.check()['source_group_split_checks'], {'source_sha256': True, 'source_group': True})

    def test_group_only_contract_is_supported(self):
        self.train['images'][0]['source_group'] = 'recording-1'
        self.val['images'][0]['source_group'] = 'recording-2'
        self.assertEqual(self.check()['source_group_split_checks'], {'source_sha256': False, 'source_group': True})

    def test_sha_group_alias_is_case_insensitive(self):
        self.train['images'][0]['source_group'] = 'sha256:' + 'a' * 64
        self.val['images'][0]['source_group'] = 'SHA256:' + 'A' * 64
        with self.assertRaisesRegex(ValueError, 'share source_group'):
            self.check()

    def test_rejects_same_video_in_both_splits(self):
        self.val["images"][0]["source_video"] = "train.mov"
        with self.assertRaisesRegex(ValueError, "share source videos"):
            self.check()

    def test_rejects_box_outside_image(self):
        self.train["annotations"][0]["bbox"] = [80, 5, 40, 50]
        with self.assertRaisesRegex(ValueError, "outside-image"):
            self.check()

    def test_rejects_class_without_training_examples(self):
        for data in [self.train, self.val]:
            data["categories"].append({"id": 8, "name": "large_bag"})
        with self.assertRaisesRegex(ValueError, "No training boxes"):
            self.check()

    def test_validation_never_applies_requested_augmentation(self):
        from PIL import Image
        from unittest.mock import patch
        import torch
        from train_objects import LocalCocoDataset
        Image.new('RGB', (100, 100), (50, 100, 150)).save(self.root / 'a.jpg')
        path = self.root / 'validation.json'
        path.write_text(json.dumps(self.train), encoding='utf-8')
        dataset = LocalCocoDataset(path, self.root, 128, augment=False, augmentation='bag_focus')
        with patch('object_augmentation.augment_sample', side_effect=AssertionError('validation augmented')):
            first, target = dataset[0]
            second, _ = dataset[0]
        self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(target['boxes'], torch.tensor([[.25, .30, .40, .50]]))

    def test_evaluation_accepts_utf8_russian_metadata(self):
        import torch
        from train_objects import evaluate
        data = copy.deepcopy(self.train)
        data['info'] = {'description': 'Изменённый мешок — проверка цвета'}
        data['annotations'][0].update(area=2000, iscrowd=0)
        path = self.root / 'проверка.json'
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8-sig')

        class ExactModel(torch.nn.Module):
            def forward(self, images):
                return None

        def exact_prediction(output, sizes):
            return [{'labels': torch.tensor([0]),
                     'boxes': torch.tensor([[5., 5., 45., 55.]]),
                     'scores': torch.tensor([0.99])}]

        loader = [(torch.zeros(1, 3, 16, 16),
                   [{'orig_size': torch.tensor([100, 100]), 'image_id': torch.tensor(1)}])]
        metrics = evaluate(ExactModel(), exact_prediction, loader, data['categories'], path,
                           torch.device('cpu'), self.root / 'predictions.json')
        self.assertAlmostEqual(metrics['map_50'], 1.0)


class ClassificationInitializationChecks(unittest.TestCase):
    def test_matching_names_keep_head_even_if_coco_ids_differ(self):
        report = classification_initialization({'categories': [{'id': 1, 'name': 'cash'}]},
                                               [{'id': 7, 'name': 'cash'}])
        self.assertEqual(report['mode'], 'reuse_matching_class_order')

    def test_same_count_different_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'names/order differ'):
            classification_initialization({'categories': [{'name': 'bag'}]}, [{'name': 'cash'}])

    def test_same_names_reordered_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'names/order differ'):
            classification_initialization({'categories': [{'name': 'cash'}, {'name': 'bag'}]},
                                          [{'name': 'bag'}, {'name': 'cash'}])

    def test_changed_count_records_head_reset(self):
        report = classification_initialization({'categories': [{'name': 'cash'}]},
                                               [{'name': 'cash'}, {'name': 'bag'}])
        self.assertEqual(report['mode'], 'reset_changed_class_count')
        self.assertEqual(report['source_category_names'], ['cash'])

    def test_base_coco_checkpoint_without_label_metadata_remains_supported(self):
        report = classification_initialization(None, [{'name': 'cash'}])
        self.assertEqual(report['label_source'], 'pretrained_checkpoint_without_category_metadata')

    def test_malformed_custom_labels_are_rejected(self):
        for categories in [[], ['cash'], [{'name': ''}], [{'name': 'cash'}, {'name': 'cash'}]]:
            with self.subTest(categories=categories), self.assertRaises(ValueError):
                classification_initialization({'categories': categories}, [{'name': 'cash'}])


if __name__ == "__main__":
    unittest.main()
