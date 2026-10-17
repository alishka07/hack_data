"""Geometry, label retention and reproducibility checks; no model/GPU required."""
import json
import random
import unittest

import numpy as np
from PIL import Image, ImageDraw

from object_augmentation import (affine_matrix, apply_affine, augment_sample,
                                 horizontal_flip_boxes, recolor_masked_bag, transform_boxes)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 80), (20, 40, 60))
        ImageDraw.Draw(self.image).rectangle((10, 20, 29, 39), fill=(255, 0, 0))
        self.boxes = [[10, 20, 30, 40], [60, 40, 90, 70]]
        self.labels = [7, 42]

    def test_flip_edges_and_involution(self):
        flipped = horizontal_flip_boxes(self.boxes, 100)
        self.assertEqual(flipped, [[70, 20, 90, 40], [10, 40, 40, 70]])
        self.assertEqual(horizontal_flip_boxes(flipped, 100), self.boxes)
        self.assertEqual(horizontal_flip_boxes([[0, 0, 100, 80]], 100), [[0, 0, 100, 80]])

    def test_translation_matches_pixels(self):
        matrix = affine_matrix(100, 80, translate_x=5, translate_y=3)
        image, boxes, info = apply_affine(self.image, [[10, 20, 30, 40]], matrix)
        self.assertTrue(info["accepted"])
        self.assertEqual(boxes, [[15, 23, 35, 43]])
        self.assertEqual(image.getpixel((15, 23)), (255, 0, 0))
        self.assertEqual(image.getpixel((34, 42)), (255, 0, 0))
        self.assertEqual(image.getpixel((35, 43)), (20, 40, 60))

    def test_scale_matches_edge_boxes(self):
        matrix = affine_matrix(100, 80, scale=0.5)
        image, boxes, info = apply_affine(self.image, [[10, 20, 30, 40]], matrix)
        self.assertTrue(info["accepted"])
        self.assertEqual(boxes, [[30, 30, 40, 40]])
        self.assertEqual(image.getpixel((35, 35)), (255, 0, 0))

    def test_crop_rejects_entire_proposal_without_dropping_labels(self):
        # The first object would retain only 25% of its width.
        matrix = affine_matrix(100, 80, translate_x=-25)
        image, boxes, info = apply_affine(self.image, self.boxes, matrix)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["reason"], "box_visibility")
        self.assertEqual(boxes, self.boxes)
        self.assertEqual(image.tobytes(), self.image.tobytes())

    def test_visible_sixty_percent_is_kept_but_less_is_rejected(self):
        matrix = affine_matrix(100, 80, translate_x=-18)
        boxes, info = transform_boxes([[10, 20, 30, 40]], matrix, (100, 80))
        self.assertTrue(info["accepted"])
        self.assertEqual(boxes, [[0, 20, 12, 40]])
        boxes, info = transform_boxes([[10, 20, 30, 40]], affine_matrix(100, 80, translate_x=-18.1), (100, 80))
        self.assertIsNone(boxes)
        self.assertFalse(info["accepted"])

    def test_tiny_or_fully_outside_proposals_rejected(self):
        for matrix in [affine_matrix(100, 80, scale=0.1), affine_matrix(100, 80, translate_x=200)]:
            boxes, info = transform_boxes(self.boxes, matrix, (100, 80))
            self.assertIsNone(boxes)
            self.assertFalse(info["accepted"])

    def test_rotation_clipped_polygon_visibility(self):
        # Rotating a square 45 degrees about its center inflates its bounding box;
        # visibility is still 100% when the actual rotated polygon is in frame.
        matrix = affine_matrix(100, 80, angle_degrees=45)
        boxes, info = transform_boxes([[40, 30, 60, 50]], matrix, (100, 80))
        self.assertTrue(info["accepted"])
        self.assertAlmostEqual(info["visibility"][0], 1.0)
        self.assertAlmostEqual(boxes[0][2] - boxes[0][0], 20 * 2 ** 0.5)

    def test_seed_repeatability_including_noise_and_jpeg(self):
        # Multiple seeds cover all degradation branches and optional transforms.
        names = set()
        for seed in range(30):
            first = augment_sample(self.image, self.boxes, self.labels, 42, random.Random(seed))
            second = augment_sample(self.image, self.boxes, self.labels, 42, random.Random(seed))
            self.assertEqual(first[0].tobytes(), second[0].tobytes())
            self.assertEqual(first[1:], second[1:])
            json.dumps(first[3], allow_nan=False)
            names.update(t["name"] for t in first[3]["transforms"])
        self.assertTrue({"sensor_noise", "jpeg_compression", "affine", "color", "horizontal_flip"} <= names)

    def test_all_labels_and_bounds_preserved_many_random_proposals(self):
        for seed in range(150):
            image, boxes, labels, meta = augment_sample(self.image, self.boxes, self.labels, 42, random.Random(seed))
            self.assertEqual(image.size, self.image.size)
            self.assertEqual(labels, [7, 42])
            self.assertEqual(len(boxes), 2)
            for x1, y1, x2, y2 in boxes:
                self.assertTrue(0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 80)
            self.assertTrue(meta["bag_frame"])
        self.assertEqual(self.boxes, [[10, 20, 30, 40], [60, 40, 90, 70]])
        self.assertEqual(self.labels, [7, 42])

    def test_empty_boxes_and_bag_label_zero(self):
        image, boxes, labels, meta = augment_sample(self.image, [], [], 0, random.Random(7))
        self.assertEqual((boxes, labels), ([], []))
        self.assertFalse(meta["bag_frame"])
        self.assertEqual(image.size, (100, 80))
        _, _, _, meta = augment_sample(self.image, [[10, 20, 30, 40]], [0], 0, random.Random(7))
        self.assertTrue(meta["bag_frame"])

    def test_none_profile_is_exact_copy(self):
        image, boxes, labels, meta = augment_sample(self.image, self.boxes, self.labels, 42, random.Random(4), "none")
        self.assertEqual(image.tobytes(), self.image.tobytes())
        self.assertEqual(boxes, self.boxes)
        self.assertEqual(labels, self.labels)
        self.assertEqual(meta["transforms"], [])
        self.assertIsNot(image, self.image)

    def test_mask_recolors_only_foreground_and_preserves_background_exactly(self):
        mask = Image.new("L", (100, 80))
        ImageDraw.Draw(mask).ellipse((12, 22, 26, 36), fill=255)
        result = recolor_masked_bag(self.image, mask, hue_turns=0.33, saturation=1.3)
        before, after, foreground = np.asarray(self.image), np.asarray(result), np.asarray(mask) > 0
        self.assertTrue(np.array_equal(before[~foreground], after[~foreground]))
        self.assertFalse(np.array_equal(before[foreground], after[foreground]))

    def test_supplied_mask_is_audited_and_deterministic(self):
        mask = Image.new("L", (100, 80))
        ImageDraw.Draw(mask).ellipse((12, 22, 26, 36), fill=255)
        first = augment_sample(self.image, self.boxes, self.labels, 7, random.Random(11), bag_mask=mask)
        second = augment_sample(self.image, self.boxes, self.labels, 7, random.Random(11), bag_mask=mask)
        self.assertEqual(first[0].tobytes(), second[0].tobytes())
        self.assertEqual(first[1:], second[1:])
        self.assertTrue(first[3]["bag_mask_applied"])
        self.assertEqual(len(first[3]["bag_mask_sha256"]), 64)
        self.assertEqual(first[2], self.labels)

    def test_unrelated_or_wrong_size_mask_rejected(self):
        for mask in [Image.new("L", (99, 80), 255), Image.new("L", (100, 80), 255), Image.new("L", (100, 80))]:
            with self.assertRaises(ValueError):
                augment_sample(self.image, self.boxes, self.labels, 7, random.Random(11), bag_mask=mask)
        with self.assertRaises(ValueError):
            augment_sample(self.image, self.boxes, self.labels, 8, random.Random(11), bag_mask=Image.new("L", (100, 80), 255))

    def test_invalid_annotations_fail_instead_of_silently_disappearing(self):
        for invalid in [[[-1, 0, 10, 20]], [[0, 0, 101, 20]], [[4, 4, 4, 9]], [[0, 0, float("nan"), 4]]]:
            with self.assertRaises(ValueError):
                augment_sample(self.image, invalid, [1], 1, random.Random(1))
        with self.assertRaises(ValueError):
            augment_sample(self.image, self.boxes, [1], 1, random.Random(1))


if __name__ == "__main__":
    unittest.main()
