"""Small regressions for observed counter/bag/outlet failure modes; no model run."""
import copy
import json
import math
from pathlib import Path
import unittest

from candidate_filters import canonical_label, filter_candidates


def detection(label, box, score=0.5, query="test"):
    return dict(label=label, box=box, score=score, query=query)


class CandidateFilterTests(unittest.TestCase):
    def test_chocolate_cash_conflict_is_not_resolved_by_higher_cash_score(self):
        result = filter_candidates([detection('money', [20, 20, 50, 35], .8),
                                    detection('a chocolate bar', [20, 20, 50, 35], .35)], (100, 100))
        self.assertEqual(result['candidates'], [])
        self.assertEqual({row['label'] for row in result['review']}, {'cash', 'food'})

    def test_food_near_cash_is_not_the_same_object(self):
        result = filter_candidates([detection('banknotes', [20, 20, 50, 35], .5),
                                    detection('a coffee cup', [60, 20, 80, 45], .6)], (100, 100))
        self.assertEqual({row['label'] for row in result['candidates']}, {'cash', 'drink'})
        self.assertFalse(result['review'])

    def test_partial_cash_food_overlap_keeps_both_ambiguous(self):
        result = filter_candidates([detection('banknotes', [10, 10, 30, 30], .45),
                                    detection('a paper cup', [18, 10, 38, 30], .80)], (100, 100))
        self.assertFalse(result['candidates'])
        self.assertEqual({row['label'] for row in result['review']}, {'cash', 'drink'})

    def test_nested_incompatible_hypotheses_are_reviewed_without_relabelling(self):
        rows = [detection("banknote_counter", [296, 353, 428, 592], 0.33),
                detection("desktop computer tower", [303, 354, 414, 498], 0.63)]
        result = filter_candidates(rows, (464, 848))
        self.assertEqual(result["candidates"], [])
        self.assertEqual({x["label"] for x in result["review"]}, {"banknote_counter", "desktop_computer"})
        self.assertEqual(len(result["review"]), 2)
        self.assertTrue(all("conflicting_class_predictions" in x["reasons"] for x in result["review"]))

    def test_containment_does_not_merge_tiny_item_with_huge_region(self):
        rows = [detection("banknote_counter", [0, 0, 100, 100]),
                detection("plastic_bag", [20, 20, 40, 40])]
        result = filter_candidates(rows, (100, 100))
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["review"], [])

    def test_actual_x1_nested_safe_boxes_leave_no_unambiguous_counter_there(self):
        path = Path(__file__).parent / "regression_inputs" / "X1_10_pipeline_raw.json"
        if not path.is_file():
            self.skipTest("Optional private case fixture is not included in the portable repository")
        fixture = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["timestamp_sec"], 10.0)
        result = filter_candidates(fixture["raw_detections"], tuple(fixture["image_size"]))
        # Right-hand cabinet hypotheses are all ambiguous. This deliberately
        # does not label the physical cabinet as a safe on the model's behalf.
        cabinet_candidates = [row for row in result["candidates"]
                              if row["label"] == "banknote_counter" and row["box"][0] > 290]
        self.assertEqual(cabinet_candidates, [])
        cabinet_review = [row for row in result["review"] if row["box"][0] > 290]
        self.assertEqual({row["label"] for row in cabinet_review}, {"banknote_counter", "desktop_computer"})
        self.assertTrue(any(row["label"] == "banknote_counter" and row["box"][3] > 580 for row in cabinet_review))
        # The separate, left-hand genuine counter remains a model candidate.
        self.assertTrue(any(row["label"] == "banknote_counter" and row["box"][0] < 40
                            and row["box"][1] > 390 for row in result["candidates"]))

    def test_chair_bag_same_box_is_reviewed_when_both_predictions_exist(self):
        rows = [detection("plastic_bag", [2, 716, 203, 846], 0.455),
                detection("office chair", [0, 715, 204, 848], 0.40)]
        result = filter_candidates(rows, (464, 848))
        self.assertEqual(result["candidates"], [])
        self.assertEqual({x["label"] for x in result["review"]}, {"plastic_bag", "chair"})

    def test_counter_bag_conflict_does_not_trust_larger_score(self):
        rows = [detection("a plastic bag.", [35, 400, 114, 489], 0.604, "plastic bag"),
                detection("a banknote counting machine.", [33, 398, 115, 490], 0.419, "equipment")]
        result = filter_candidates(rows, (464, 848))
        self.assertEqual(result["candidates"], [])
        self.assertEqual({x["label"] for x in result["review"]}, {"plastic_bag", "banknote_counter"})
        self.assertTrue(all("conflicting_class_predictions" in x["reasons"] for x in result["review"]))

    def test_outlet_exclusion_keeps_counter_and_real_bag_elsewhere(self):
        rows = [detection("banknote_counter", [28, 494, 75, 545], 0.304),
                detection("banknote_counter", [33, 398, 115, 490], 0.419),
                detection("plastic_bag", [132, 370, 207, 406], 0.573)]
        regions = [{"kind": "exclude", "label": "banknote_counter", "source": "manual",
                    "box": [25 / 464, 491 / 848, 78 / 464, 548 / 848]}]
        result = filter_candidates(rows, (464, 848), regions)
        self.assertEqual([x["detection_index"] for x in result["candidates"]], [1, 2])
        self.assertEqual(result["suppressed"][0]["reasons"], ["manual_exclusion_region"])

    def test_equipment_hint_marks_bag_ambiguous_but_never_relabels(self):
        regions = [{"kind": "equipment", "label": "banknote_counter", "source": "manual",
                    "box": [0.1, 0.1, 0.3, 0.3]}]
        result = filter_candidates([detection("plastic_bag", [10, 10, 30, 30], 0.9)], (100, 100), regions)
        self.assertEqual(result["review"][0]["label"], "plastic_bag")
        self.assertIn("overlaps_reviewed_equipment", result["review"][0]["reasons"])
        self.assertEqual(filter_candidates([], (100, 100), regions)["candidates"], [])

    def test_partial_overlap_does_not_suppress_huge_or_neighbouring_object(self):
        regions = [{"kind": "exclude", "label": "banknote_counter", "source": "manual",
                    "box": [0.1, 0.1, 0.3, 0.3]}]
        result = filter_candidates([detection("banknote_counter", [0, 0, 100, 100]),
                                    detection("banknote_counter", [25, 20, 50, 40])], (100, 100), regions)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["suppressed"], [])

    def test_huge_equipment_region_does_not_mark_every_tiny_bag(self):
        regions = [{"kind": "equipment", "label": "banknote_counter", "source": "manual",
                    "box": [0, 0, 1, 1]}]
        result = filter_candidates([detection("plastic_bag", [10, 10, 20, 20])], (100, 100), regions)
        self.assertEqual(len(result["candidates"]), 1)

    def test_invalid_scores_and_boxes_cannot_enter_candidates_or_nonstandard_json(self):
        rows = [detection("plastic_bag", [1, 1, 20, 20], score)
                for score in [math.nan, math.inf, -0.1, 1.1, True, "0.8"]]
        rows += [detection("plastic_bag", box) for box in
                 [[1, 1, math.inf, 10], [10, 1, 1, 20], [101, 101, 110, 110], None]]
        result = filter_candidates(rows, (100, 100))
        self.assertFalse(result["candidates"] or result["review"])
        self.assertEqual(len(result["suppressed"]), len(rows))
        json.dumps(result, allow_nan=False)

    def test_fragment_and_combined_labels_do_not_become_safe_or_cash(self):
        rows = [detection(label, [1, 1, 20, 20]) for label in
                ["a metal", "a stack", "a laptop a computer monitor", "black object"]]
        result = filter_candidates(rows, (100, 100))
        self.assertEqual(result["candidates"], [])
        self.assertEqual(len(result["suppressed"]), 4)
        self.assertIsNone(canonical_label("a metal"))
        self.assertEqual(canonical_label("a bag."), "bag")

    def test_cash_synonyms_can_merge_but_mixed_classes_still_cannot(self):
        self.assertEqual(canonical_label('banknotes money'), 'cash')
        self.assertEqual(canonical_label('money banknotes'), 'cash')
        self.assertIsNone(canonical_label('banknotes a snack'))
        result = filter_candidates([detection('banknotes money', [10,10,30,30], .41),
                                    detection('a snack', [10,10,30,30], .7)], (100,100))
        self.assertEqual(result['candidates'], [])
        self.assertEqual({row['label'] for row in result['review']}, {'cash', 'food'})

    def test_same_class_duplicates_use_score_and_keep_prompt_provenance(self):
        rows = [detection("a plastic bag.", [1, 1, 20, 20], 0.5, "first"),
                detection("plastic_bag", [2, 2, 20, 20], 0.7, "second")]
        original = copy.deepcopy(rows)
        result = filter_candidates(rows, (100, 100))
        self.assertEqual(result["candidates"][0]["query"], "second")
        self.assertEqual(result["suppressed"][0]["duplicate_of"], 1)
        self.assertEqual(rows, original)

    def test_regions_must_be_manual_and_normalized(self):
        for region in [dict(kind="equipment", label="safe", source="model", box=[0, 0, 1, 1]),
                       dict(kind="equipment", label="safe", source="manual", box=[0, 0, 100, 100])]:
            with self.assertRaises(ValueError):
                filter_candidates([], (100, 100), [region])


if __name__ == "__main__":
    unittest.main(verbosity=2)
