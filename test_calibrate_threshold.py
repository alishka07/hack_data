"""CPU-only regression checks for validation threshold fitting."""
import copy
import unittest

from calibrate_threshold import fit_thresholds


class ThresholdCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.annotations = {"images": [{"id": 1}], "categories": [{"id": 7, "name": "safe"}],
                            "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 40, 40]}]}

    def prediction(self, box, score, category=7, image=1):
        return {"image_id": image, "category_id": category, "bbox": box, "score": score}

    def test_selects_higher_threshold_in_tie_and_removes_false_positive(self):
        result = fit_thresholds(self.annotations, [self.prediction([10, 10, 40, 40], .8), self.prediction([80, 80, 10, 10], .4)])
        self.assertEqual(result["thresholds"]["safe"], .8)
        self.assertEqual(result["classes"]["safe"]["tp"], 1)
        self.assertEqual(result["classes"]["safe"]["fp"], 0)
        self.assertFalse(result["independent_test"])

    def test_same_class_nms_removes_duplicate(self):
        result = fit_thresholds(self.annotations, [self.prediction([10, 10, 40, 40], .8), self.prediction([11, 11, 40, 40], .75)], thresholds=[.7])
        self.assertEqual(result["classes"]["safe"]["predictions_after_nms"], 1)
        self.assertEqual(result["classes"]["safe"]["fp"], 0)

    def test_one_ground_truth_cannot_match_two_predictions(self):
        # Both boxes have IoU .5 with truth, but IoU 0 with each other, so NMS keeps both.
        result = fit_thresholds(self.annotations, [self.prediction([10, 10, 20, 40], .8), self.prediction([30, 10, 20, 40], .7)], thresholds=[.6])
        self.assertEqual(result["classes"]["safe"]["tp"], 1)
        self.assertEqual(result["classes"]["safe"]["fp"], 1)

    def test_matching_and_nms_do_not_cross_images_or_classes(self):
        annotations = copy.deepcopy(self.annotations)
        annotations["images"].append({"id": 2})
        annotations["categories"].append({"id": 8, "name": "bag"})
        predictions = [self.prediction([10, 10, 40, 40], .8, image=2), self.prediction([10, 10, 40, 40], .8, category=8)]
        result = fit_thresholds(annotations, predictions, thresholds=[.7])
        self.assertEqual(result["classes"]["safe"]["tp"], 0)
        self.assertEqual(result["classes"]["safe"]["fn"], 1)
        self.assertEqual(result["classes"]["bag"]["fp"], 1)
        self.assertEqual(result["classes"]["bag"]["support_status"], "no_positive_validation_examples")

    def test_missing_predictions_are_false_negatives(self):
        result = fit_thresholds(self.annotations, [])
        self.assertEqual(result["thresholds"]["safe"], .9)
        self.assertEqual(result["classes"]["safe"]["fn"], 1)
        self.assertEqual(result["classes"]["safe"]["f1"], 0)


if __name__ == "__main__":
    unittest.main()
