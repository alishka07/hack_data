import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from reviewed_objects import ReviewedObjectMatcher, _same_cash_object


class ReviewedObjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.video = self.folder / 'X-46.avi'
        rng = np.random.default_rng(41)
        self.frame = np.full((128, 160, 3), 105, dtype=np.uint8)
        self.frame[50:70, 50:82] = rng.integers(0, 256, (20, 32, 3), dtype=np.uint8)
        self._write_video(self.video, self.frame)
        capture = cv2.VideoCapture(str(self.video))
        ok, self.decoded = capture.read()
        capture.release()
        self.assertTrue(ok)
        self.record = {
            'number': 1, 'label_status': 'human_corrected_class',
            'confirmed_label': 'chocolate', 'correction_id': 'review-1',
            'negative_for_labels': ['cash', 'banknotes'],
            'source_video_sha256': hashlib.sha256(self.video.read_bytes()).hexdigest(),
            'source_frame_zero_based': 0, 'frame_size': [160, 128],
            'integer_box_xyxy': [50, 50, 82, 70],
            'crop_box_xyxy': [40, 40, 92, 80],
        }
        self.annotation = self.folder / 'annotations.json'
        self._save()
        self.matcher = ReviewedObjectMatcher.for_video(self.video, self.annotation)
        self.assertIsNotNone(self.matcher)

    @staticmethod
    def _write_video(path, frame):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (frame.shape[1], frame.shape[0]))
        if not writer.isOpened():
            raise RuntimeError('MJPG encoder required for source-binding tests')
        writer.write(frame)
        writer.release()

    def _save(self):
        self.annotation.write_text(json.dumps({'candidates': [self.record]}), encoding='utf-8')

    def test_true_object_and_provenance(self):
        result, notes = self.matcher.apply(self.decoded, [])
        self.assertEqual(len(result), 1)
        item = result[0]
        self.assertEqual(item['label'], 'food')
        self.assertEqual(item['confirmed_class'], 'chocolate')
        self.assertEqual(item['source'], 'human_confirmed_reference_match')
        self.assertEqual(item['reference_video_sha256'], self.record['source_video_sha256'])
        self.assertEqual(item['reference_frame_zero_based'], 0)
        self.assertEqual(item['score'], item['correlation'])
        self.assertEqual(item['score_kind'], 'reference_similarity_not_probability')
        self.assertGreaterEqual(item['correlation'], .999)
        self.assertEqual(len(notes), 1)

    def test_small_translation_matches(self):
        shifted = cv2.warpAffine(self.decoded, np.float32([[1, 0, 7], [0, 1, -5]]), (160, 128), borderValue=(105, 105, 105))
        result, _ = self.matcher.apply(shifted, [])
        self.assertEqual(result[0]['box'], [57, 45, 89, 65])

    def test_same_filename_other_source_never_matches(self):
        elsewhere = self.folder / 'another'
        elsewhere.mkdir()
        path = elsewhere / self.video.name
        other = self.frame.copy()
        other[0:30, 0:30] = 15
        self._write_video(path, other)
        self.assertIsNone(ReviewedObjectMatcher.for_video(path, self.annotation))

    def test_sha_binding_ignores_annotation_filename(self):
        self.record['source_name'] = 'wrong-name.mov'
        self._save()
        self.assertIsNotNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))
        self.record['source_video_sha256'] = '0' * 64
        self._save()
        self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def test_overlap_suppressed_other_cash_preserved_without_mutation(self):
        raw = [{'label': 'cash', 'box': [50, 50, 82, 70], 'score': .44},
               {'label': 'banknotes', 'box': [51, 50, 82, 70], 'score': .32},
               {'label': 'cash', 'box': [92, 52, 116, 72], 'score': .63},
               {'label': 'food', 'box': [50, 50, 82, 70], 'score': .5}]
        before = copy.deepcopy(raw)
        result, notes = self.matcher.apply(self.decoded, raw)
        self.assertEqual(raw, before)
        self.assertEqual(len(result), 2)
        self.assertEqual(sum(item['label'] == 'food' for item in result), 1)
        self.assertIn(before[2], result)
        self.assertEqual(notes[0]['suppressed_cash_count'], 2)

    def test_wrong_patch_is_not_matched(self):
        frame = self.decoded.copy()
        frame[48:72, 48:84] = np.random.default_rng(19).integers(0, 256, (24, 36, 3), dtype=np.uint8)
        self.assertEqual(self.matcher.apply(frame, []), ([], []))

    def test_occluded_object_is_not_matched(self):
        frame = self.decoded.copy()
        frame[50:70, 50:73] = (180, 165, 140)
        self.assertEqual(self.matcher.apply(frame, []), ([], []))

    def test_broad_constant_background_is_not_matched(self):
        self.assertEqual(self.matcher.apply(np.full_like(self.decoded, 105), []), ([], []))

    def test_large_translation_is_not_matched(self):
        shifted = cv2.warpAffine(self.decoded, np.float32([[1, 0, 35], [0, 1, 0]]), (160, 128), borderValue=(105, 105, 105))
        self.assertEqual(self.matcher.apply(shifted, []), ([], []))

    def test_same_shape_different_color_rejected(self):
        frame = self.decoded.copy()
        frame[50:70, 50:82] = np.clip(frame[50:70, 50:82].astype(np.int16)+80, 0, 255).astype(np.uint8)
        self.assertEqual(self.matcher.apply(frame, []), ([], []))

    def test_wrong_dimensions_rejected(self):
        self.assertEqual(self.matcher.apply(cv2.resize(self.decoded, (320, 256)), []), ([], []))
        self.record['frame_size'] = [161, 128]
        self._save()
        self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def test_original_cash_confirmations_are_not_loaded(self):
        self.record.update(confirmed_label='cash', label_status='human_confirmed_class')
        self._save()
        self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def _cash_matcher(self):
        self.record.update(confirmed_label='cash', label_status='human_confirmed_class',
                           reference_matching_enabled=True, number=6, correction_id='cash-review-6')
        self.record.pop('negative_for_labels', None)
        self._save()
        matcher = ReviewedObjectMatcher.for_video(self.video, self.annotation)
        self.assertIsNotNone(matcher)
        return matcher

    def test_cash_reference_requires_explicit_boolean_opt_in_and_confirmed_status(self):
        matcher = self._cash_matcher()
        result, notes = matcher.apply(self.decoded, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'cash')
        self.assertEqual(result[0]['confirmed_class'], 'cash')
        self.assertEqual(result[0]['score_kind'], 'reference_similarity_not_probability')
        self.assertEqual(result[0]['source'], 'human_confirmed_reference_match')
        self.assertEqual(result[0]['annotation_number'], 6)
        self.assertEqual(notes[0]['action'], 'apply_human_confirmed_reference')
        for enabled in (False, None, 1, 'true'):
            with self.subTest(reference_matching_enabled=enabled):
                self.record['reference_matching_enabled'] = enabled
                self._save()
                self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))
        self.record['reference_matching_enabled'] = True
        for status in ('human_corrected_class', 'unconfirmed', 'model_prediction', None, ['human_confirmed_class']):
            with self.subTest(label_status=status):
                self.record['label_status'] = status
                self._save()
                self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def test_cash_reference_does_not_enable_older_cash_confirmations(self):
        self._cash_matcher()
        older = copy.deepcopy(self.record)
        older.update(number=2, correction_id='older-cash', source_frame_zero_based=999)
        older.pop('reference_matching_enabled')
        self.annotation.write_text(json.dumps({'candidates': [older, self.record]}), encoding='utf-8')
        matcher = ReviewedObjectMatcher.for_video(self.video, self.annotation)
        self.assertIsNotNone(matcher)
        self.assertEqual([item.annotation_number for item in matcher.references], [6])

    def test_malformed_class_rows_do_not_disable_valid_cash_reference(self):
        self._cash_matcher()
        malformed = copy.deepcopy(self.record)
        malformed.update(label_status='human_corrected_class', confirmed_label=['chocolate'])
        self.annotation.write_text(json.dumps({'candidates': [None, malformed, self.record]}), encoding='utf-8')
        matcher = ReviewedObjectMatcher.for_video(self.video, self.annotation)
        self.assertIsNotNone(matcher)
        self.assertEqual([item.annotation_number for item in matcher.references], [6])

    def test_cash_reference_suppresses_only_same_object_conflicts_and_preserves_raw(self):
        matcher = self._cash_matcher()
        conflicts = ['cash', 'banknotes', 'snack', 'paper cup', 'identity card', 'document', 'notebook']
        raw = [{'label': label, 'box': [50, 50, 82, 70], 'score': .5} for label in conflicts]
        keep = [
            {'label': 'cash', 'box': [92, 52, 116, 72], 'score': .6},
            {'label': 'food', 'box': [92, 52, 116, 72], 'score': .6},
            {'label': 'document', 'box': [30, 40, 94, 80], 'score': .6},
            {'label': 'table', 'box': [40, 40, 92, 80], 'score': .8},
            {'label': 'safe', 'box': [50, 50, 82, 70], 'score': .7},
        ]
        raw.extend(keep)
        before = copy.deepcopy(raw)
        result, notes = matcher.apply(self.decoded, raw)
        self.assertEqual(raw, before)
        self.assertEqual(len(result), len(keep)+1)
        for item in keep:
            self.assertIn(item, result)
        self.assertEqual(notes[0]['suppressed_cash_count'], 2)
        self.assertEqual(notes[0]['suppressed_conflicting_count'], len(conflicts))
        self.assertEqual(notes[0]['suppressed_conflicting_labels'],
                         ['cash', 'document', 'drink', 'food', 'identity_card', 'notebook'])
        self.assertEqual(result[-1]['label'], 'cash')

    def test_cash_reference_uses_coverage_for_shifted_conflict_box(self):
        matcher = self._cash_matcher()
        raw = [{'label': 'food', 'box': [26, 48, 74, 73], 'score': .6},
               {'label': 'document', 'box': [26, 48, 67, 73], 'score': .6}]
        result, notes = matcher.apply(self.decoded, raw)
        self.assertNotIn(raw[0], result)  # 75% cash coverage, IoU below .45.
        self.assertIn(raw[1], result)  # Insufficient overlap: possibly another object.
        self.assertEqual(notes[0]['suppressed_conflicting_count'], 1)

    def test_cash_reference_accepts_centered_broad_duplicates_from_x26_tail(self):
        # Actual detector boxes at 18.0, 18.5, 19.0 and 19.5 seconds: the
        # reference interior occupies only 44–49% of the detector's outer box.
        examples = [
            ([208, 397, 235, 410], [204.7593, 390.9294, 238.4424, 413.5575]),
            ([208, 397, 235, 410], [204.8869, 391.1578, 238.3779, 413.2060]),
            ([208, 397, 235, 410], [204.7938, 391.1147, 238.0196, 412.8197]),
            ([208, 397, 235, 410], [205.1145, 391.1731, 238.1494, 412.7288]),
            ([208, 396, 235, 409], [204.1333, 390.3906, 238.6632, 413.2230]),
        ]
        for reference, candidate in examples:
            with self.subTest(candidate=candidate):
                self.assertTrue(_same_cash_object(candidate, reference))

    def test_cash_reference_broad_duplicate_branch_is_bounded(self):
        reference = [50, 50, 82, 70]
        negatives = {
            'area_above_three_times': [38, 40, 94, 80],
            'center_displaced_despite_containing_cash': [28, 45, 82, 75],
            'elongated_containing_box': [30, 50, 102, 70],
            'less_than_ninety_percent_coverage': [34, 52.5, 98, 75.5],
        }
        for reason, candidate in negatives.items():
            with self.subTest(reason=reason):
                self.assertFalse(_same_cash_object(candidate, reference))

    def test_cash_reference_broad_match_still_preserves_equipment_and_raw(self):
        matcher = self._cash_matcher()
        broad_box = [44, 47, 86, 79]
        raw = [{'label': 'identity card', 'box': broad_box, 'score': .31},
               {'label': 'safe', 'box': broad_box, 'score': .7},
               {'label': 'table', 'box': broad_box, 'score': .8}]
        before = copy.deepcopy(raw)
        result, notes = matcher.apply(self.decoded, raw)
        self.assertEqual(raw, before)
        self.assertNotIn(raw[0], result)
        self.assertIn(raw[1], result)
        self.assertIn(raw[2], result)
        self.assertEqual(notes[0]['suppressed_conflicting_labels'], ['identity_card'])

    def test_cash_reference_cannot_add_evidence_when_occluded(self):
        matcher = self._cash_matcher()
        occluded = self.decoded.copy()
        occluded[50:70, 50:73] = (180, 165, 140)
        raw = [{'label': 'food', 'box': [50, 50, 82, 70], 'score': .6}]
        result, notes = matcher.apply(occluded, raw)
        self.assertEqual(result, raw)
        self.assertEqual(notes, [])

    def test_cash_reference_is_bound_to_exact_source_bytes(self):
        self._cash_matcher()
        other = self.folder / 'X-26.avi'
        changed = self.frame.copy()
        changed[:20, :20] = 12
        self._write_video(other, changed)
        self.assertIsNone(ReviewedObjectMatcher.for_video(other, self.annotation))
        self.record['source_video_sha256'] = '0' * 64
        self._save()
        self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def _bag_matcher(self, regions=None):
        self.record.update(
            confirmed_label='bag', label_status='human_corrected_class',
            reference_matching_enabled=True, number=4, correction_id='bag-review-4',
            negative_for_labels=['cash', 'food'],
            rejected_class_regions=regions if regions is not None else [
                {'label': 'cash', 'box_xyxy': [50, 50, 82, 56]}])
        self._save()
        matcher = ReviewedObjectMatcher.for_video(self.video, self.annotation)
        self.assertIsNotNone(matcher)
        return matcher

    def test_bag_reference_requires_explicit_opt_in_and_corrected_status(self):
        matcher = self._bag_matcher()
        result, notes = matcher.apply(self.decoded, [])
        self.assertEqual(result[0]['label'], 'bag')
        self.assertEqual(result[0]['confirmed_class'], 'bag')
        self.assertEqual(result[0]['box'], [50, 50, 82, 70])
        self.assertEqual(result[0]['annotation_number'], 4)
        self.assertEqual(result[0]['source'], 'human_confirmed_reference_match')
        self.assertEqual(result[0]['score_kind'], 'reference_similarity_not_probability')
        self.assertEqual(notes[0]['action'], 'apply_human_corrected_reference')
        self.assertEqual(matcher.min_correlation, .90)
        self.assertEqual(matcher.max_color_mae, .08)
        for enabled in (False, None, 1, 'true'):
            with self.subTest(reference_matching_enabled=enabled):
                self.record['reference_matching_enabled'] = enabled
                self._save()
                self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))
        self.record['reference_matching_enabled'] = True
        for status in ('human_confirmed_class', 'unconfirmed', 'model_prediction', None):
            with self.subTest(label_status=status):
                self.record['label_status'] = status
                self._save()
                self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def test_bag_reference_requires_both_rejected_classes(self):
        self._bag_matcher()
        for negatives in ([], ['cash'], ['food'], None, 'cash food'):
            with self.subTest(negative_for_labels=negatives):
                self.record['negative_for_labels'] = negatives
                self._save()
                self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def test_bag_reference_is_bound_to_exact_source_bytes(self):
        self._bag_matcher()
        other = self.folder / 'same-frame-other-source.avi'
        changed = self.frame.copy()
        changed[:20, :20] = 12
        self._write_video(other, changed)
        self.assertIsNone(ReviewedObjectMatcher.for_video(other, self.annotation))
        self.record['source_video_sha256'] = '0' * 64
        self._save()
        self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def test_bag_reference_replaces_full_object_conflicts_and_reviewed_cash_strip_only(self):
        matcher = self._bag_matcher()
        full_object = [{'label': label, 'box': [50, 50, 82, 70], 'score': .6}
                       for label in ('bag', 'cash', 'snack', 'paper cup')]
        reviewed_strip = {'label': 'banknotes', 'box': [50, 50, 82, 56], 'score': .4}
        keep = [
            {'label': 'cash', 'box': [53, 60, 65, 68], 'score': .7},
            {'label': 'cash', 'box': [92, 52, 116, 72], 'score': .6},
            {'label': 'food', 'box': [53, 60, 65, 68], 'score': .7},
            {'label': 'table', 'box': [50, 50, 82, 70], 'score': .8},
        ]
        raw = full_object + [reviewed_strip] + keep
        before = copy.deepcopy(raw)
        result, notes = matcher.apply(self.decoded, raw)
        self.assertEqual(raw, before)
        self.assertEqual(len(result), len(keep) + 1)
        for item in keep:
            self.assertIn(item, result)
        self.assertEqual(result[-1]['label'], 'bag')
        self.assertEqual(notes[0]['suppressed_cash_count'], 2)
        self.assertEqual(notes[0]['suppressed_conflicting_count'], 5)
        self.assertEqual(notes[0]['suppressed_conflicting_labels'], ['bag', 'cash', 'drink', 'food'])
        self.assertEqual(result[-1]['reference_rejected_class_regions'],
                         [{'label': 'cash', 'box_xyxy': [50, 50, 82, 56]}])

    def test_bag_cash_strip_suppression_tracks_reference_translation(self):
        matcher = self._bag_matcher()
        shifted = cv2.warpAffine(self.decoded, np.float32([[1, 0, 7], [0, 1, -5]]),
                                (160, 128), borderValue=(105, 105, 105))
        old_location = {'label': 'cash', 'box': [50, 50, 82, 56], 'score': .6}
        shifted_strip = {'label': 'cash', 'box': [57, 45, 89, 51], 'score': .6}
        result, notes = matcher.apply(shifted, [old_location, shifted_strip])
        self.assertIn(old_location, result)
        self.assertNotIn(shifted_strip, result)
        self.assertEqual(notes[0]['suppressed_cash_count'], 1)
        self.assertEqual(result[-1]['matched_rejected_class_regions'],
                         [{'label': 'cash', 'box_xyxy': [57, 45, 89, 51]}])

    def test_bag_without_reviewed_strip_never_suppresses_cash_by_containment(self):
        matcher = self._bag_matcher(regions=[])
        raw = [{'label': 'cash', 'box': [50, 50, 82, 56], 'score': .6},
               {'label': 'cash', 'box': [53, 60, 65, 68], 'score': .6}]
        result, notes = matcher.apply(self.decoded, raw)
        for item in raw:
            self.assertIn(item, result)
        self.assertEqual(notes[0]['suppressed_cash_count'], 0)

    def test_bag_invalid_or_outside_negative_regions_are_ignored(self):
        matcher = self._bag_matcher(regions=[
            None, {'label': 'food', 'box_xyxy': [50, 50, 82, 56]},
            {'label': 'cash', 'box_xyxy': [50, 50, 170, 56]},
            {'label': 'cash', 'box_xyxy': [40, 50, 82, 56]},
            {'label': 'cash', 'box_xyxy': ['bad', 50, 82, 56]},
        ])
        raw = [{'label': 'cash', 'box': [50, 50, 82, 56], 'score': .6}]
        result, notes = matcher.apply(self.decoded, raw)
        self.assertIn(raw[0], result)
        self.assertEqual(result[-1]['reference_rejected_class_regions'], [])
        self.assertEqual(notes[0]['suppressed_cash_count'], 0)

    def test_bag_correction_requires_visible_reference_and_preserves_raw_on_failure(self):
        matcher = self._bag_matcher()
        frame = self.decoded.copy()
        frame[50:70, 50:73] = (180, 165, 140)
        raw = [{'label': 'cash', 'box': [50, 50, 82, 56], 'score': .6}]
        before = copy.deepcopy(raw)
        result, notes = matcher.apply(frame, raw)
        self.assertEqual(result, before)
        self.assertEqual(raw, before)
        self.assertEqual(notes, [])

    def test_blank_reference_is_not_loaded(self):
        self.record.update(integer_box_xyxy=[115, 95, 145, 115], crop_box_xyxy=[105, 85, 155, 125])
        self._save()
        self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))

    def test_missing_or_malformed_inputs_are_optional(self):
        self.assertIsNone(ReviewedObjectMatcher.for_video(self.folder/'missing.avi', self.annotation))
        self.annotation.write_text('{broken', encoding='utf-8')
        self.assertIsNone(ReviewedObjectMatcher.for_video(self.video, self.annotation))
        self.assertEqual(self.matcher.apply(None, []), ([], []))


if __name__ == '__main__':
    unittest.main()
