"""CPU-only pipeline tests using an actual tiny video and a fake detector.

No model weights, torch, CUDA, network, or H.264 preview encoder are needed.
"""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import run as pipeline


class RunPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="case_pipeline_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "synthetic_sampling.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 80))
        if not writer.isOpened():
            writer.release()
            self.fail("Cannot create the CPU MJPG test fixture; pipeline test was not executed")
        try:
            for index in range(20):
                frame = np.full((80, 96, 3), index * 10, dtype=np.uint8)
                cv2.rectangle(frame, (20, 40), (55, 70), (240, 240, 240), -1)
                writer.write(frame)
        finally:
            writer.release()
        self.calls = []
        self.args = SimpleNamespace(output=self.root / "results", interval=0.5,
                                    groups=["equipment", "bags"], threshold=0.28,
                                    max_samples=0, no_video=True, resume=False)

    def detect(self, image, loaded, prompt, threshold):
        self.calls.append((prompt, threshold, image.size, loaded))
        if prompt == pipeline.PROMPTS["equipment"]:
            return [{"label": "a banknote counting machine", "box": [20, 40, 55, 70], "score": 0.42},
                    {"label": "a banknote counting machine", "box": None, "score": 0.40}]
        if prompt == pipeline.PROMPTS["bags"]:
            return [{"label": "a plastic bag", "box": [20, 40, 55, 70], "score": 0.60}]
        raise AssertionError(f"Unexpected prompt {prompt!r}")

    def analyze(self):
        return pipeline.analyze(self.video, self.args, (None, None, "test"), self.detect)

    def records(self):
        path = self.args.output / self.video.stem / "frames.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_reviewed_food_keeps_original_cash_output_and_provenance_separate(self):
        self.args.groups = ['food_drink']
        raw_cash = dict(label='banknotes', box=[20, 40, 55, 70], score=.44)
        corrected = dict(label='food', box=[20, 40, 55, 70], score=.96,
                         source='human_confirmed_reference_match', score_kind='reference_similarity_not_probability',
                         reference_frame_zero_based=10)
        matcher = SimpleNamespace(apply=lambda frame, rows: ([dict(corrected)], [{'action':'apply_human_corrected_reference'}]))
        with patch('reviewed_objects.ReviewedObjectMatcher.for_video', return_value=matcher):
            pipeline.analyze(self.video, self.args, (None, None, 'test'), lambda *args: [dict(raw_cash)])
        frame = self.records()[0]
        self.assertEqual(frame['raw_detections'][0]['label'], 'banknotes')
        self.assertEqual([r['label'] for r in frame['candidates']], ['food'])
        self.assertEqual(frame['candidates'][0]['source'], 'human_confirmed_reference_match')
        with (self.args.output / self.video.stem / 'detections.csv').open(encoding='utf-8-sig') as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['score_kind'], 'reference_similarity_not_probability')
        self.assertEqual(row['reference_frame'], '10')
        # Rebuilding saved journals must preserve the correction and provenance,
        # rather than turning the original false cash prediction back into cash.
        import refilter
        before_raw = [record['raw_detections'] for record in self.records()]
        with patch('reviewed_objects.ReviewedObjectMatcher.for_video', return_value=matcher):
            refilter.process(self.args.output / self.video.stem)
        rebuilt = self.records()
        self.assertEqual([record['raw_detections'] for record in rebuilt], before_raw)
        self.assertTrue(all([row['label'] for row in record['candidates']] == ['food'] for record in rebuilt))
        self.assertTrue(all(record['reviewed_object_corrections'] for record in rebuilt))
        with (self.args.output / self.video.stem / 'detections.csv').open(encoding='utf-8-sig') as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['source'], 'human_confirmed_reference_match')
        self.assertEqual(row['score_kind'], 'reference_similarity_not_probability')
        self.assertEqual(row['reference_frame'], '10')

    def test_whole_video_decoded_but_only_half_second_samples_inferred(self):
        summary = self.analyze()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["decoded_frames"], 20)
        self.assertEqual(summary["metadata_frames"], 20)
        self.assertTrue(summary["frame_count_matches"])
        self.assertEqual(summary["analyzed_samples"], 4)
        self.assertEqual(len(self.calls), 8)
        self.assertAlmostEqual(summary["last_timestamp_sec"], 1.9, places=5)
        records = self.records()
        self.assertEqual([row["source_frame"] for row in records], [0, 5, 10, 15])
        for row, timestamp in zip(records, [0, 0.5, 1, 1.5]):
            self.assertAlmostEqual(row["timestamp_sec"], timestamp, places=5)
            self.assertIn(row["timestamp_source"], {"video_pts", "frame_index_fps_estimate"})
            self.assertEqual(len(row["raw_detections"]), 3)
            self.assertEqual({item["query"] for item in row["raw_detections"]}, {"equipment", "bags"})
            self.assertEqual(row["candidates"], [])
            self.assertEqual({item["label"] for item in row["review"]}, {"banknote_counter", "plastic_bag"})
            self.assertEqual(len(row["suppressed"]), 1)
            self.assertIn("invalid_box", row["suppressed"][0]["reasons"])
        # Even apparent conflict detections must not become a disciplinary claim.
        self.assertEqual(summary["verdict"], "not_evaluated")
        self.assertEqual(summary["violations"], [])
        self.assertIsNone(summary["video_preview"])
        output = self.args.output / self.video.stem
        self.assertFalse((output / "review.mp4").exists())
        with (output / "detections.csv").open(encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 12)
        invalid_rows = [row for row in csv_rows if row["status"] == "suppressed"]
        self.assertEqual(len(invalid_rows), 4)
        self.assertTrue(all(row["x1"] == row["y1"] == row["x2"] == row["y2"] == "" for row in invalid_rows))
        for thumbnail in summary["thumbnails"]:
            self.assertTrue((output / thumbnail["file"]).is_file())

    def test_positive_cash_reference_enabled_without_food_prompt(self):
        self.args.groups = ['cash']
        reference = dict(label='cash', box=[20, 40, 55, 70], score=.96,
                         source='human_confirmed_reference_match', score_kind='reference_similarity_not_probability',
                         reference_frame_zero_based=10)
        matcher = SimpleNamespace(apply=lambda frame, rows: ([dict(reference)], [{'action':'apply_human_confirmed_reference'}]))
        with patch('reviewed_objects.ReviewedObjectMatcher.for_video', return_value=matcher):
            pipeline.analyze(self.video, self.args, (None,None,'test'), lambda *args: [])
            import refilter
            refilter.process(self.args.output / self.video.stem)
        records = self.records()
        self.assertTrue(all(not row['raw_detections'] for row in records))
        self.assertTrue(all(row['candidates'][0]['label'] == 'cash' for row in records))
        self.assertTrue(all(row['candidates'][0]['source'] == 'human_confirmed_reference_match' for row in records))

    def test_max_samples_does_not_stop_whole_video_decoding(self):
        self.args.max_samples = 1
        summary = self.analyze()
        self.assertEqual(summary["decoded_frames"], 20)
        self.assertEqual(summary["analyzed_samples"], 1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.records()), 1)
        self.assertAlmostEqual(summary["last_timestamp_sec"], 1.9, places=5)
        self.assertEqual(summary["verdict"], "not_evaluated")

    def test_matching_resume_skips_inference_but_changed_sampling_does_not(self):
        first = self.analyze()
        self.calls.clear()
        self.args.resume = True
        cached = self.analyze()
        self.assertEqual(cached, first)
        self.assertEqual(self.calls, [])
        self.args.max_samples = 1
        changed = self.analyze()
        self.assertEqual(changed["analyzed_samples"], 1)
        self.assertEqual(len(self.calls), 2)

    def test_metadata_count_mismatch_is_warning_and_not_resume_skipped(self):
        original_capture = cv2.VideoCapture

        class WrongMetadataCapture:
            def __init__(self, path):
                self.capture = original_capture(path)

            def __getattr__(self, name):
                return getattr(self.capture, name)

            def get(self, prop):
                value = self.capture.get(prop)
                return value + 3 if prop == cv2.CAP_PROP_FRAME_COUNT else value

        with patch.object(cv2, "VideoCapture", WrongMetadataCapture):
            summary = self.analyze()
            self.assertEqual(summary["status"], "completed_with_warnings")
            self.assertFalse(summary["frame_count_matches"])
            self.assertTrue(summary["decoding_warning"])
            self.assertEqual(summary["decoded_frames"], 20)
            self.assertEqual(summary["metadata_frames"], 23)
            self.calls.clear()
            self.args.resume = True
            self.analyze()
            self.assertEqual(len(self.calls), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
