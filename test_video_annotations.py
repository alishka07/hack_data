import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from video_annotations import (build_catalog, candidate_rules, decode_video, duplicate_groups,
                               import_notes, normalize_video_id, parse_timepoint)


def source(rows):
    return {"document_id": "doc", "document_url": "https://example.invalid/doc", "revision_id": "revision",
            "tabs": [{"tab_id": "t.0", "title": "Original tab", "tables": [{"start_index": 100,
                     "rows": [{"row_index": 0, "cells": ["header"]}] +
                             [{"row_index": i + 1, "cells": row} for i, row in enumerate(rows)]}]}]}


class VideoAnnotationTests(unittest.TestCase):
    def test_normalize_latin_cyrillic_and_spaces(self):
        for name in ("X - 17", "х-17.mov", "x17", "Х -17"):
            self.assertEqual(normalize_video_id(name), "X17")
        with self.assertRaises(ValueError):
            normalize_video_id("X17 or X18")

    def test_point_parsing_preserves_raw_and_no_interval(self):
        for text, expected in (("1 min 8 sec", 68), ("1 минута 8 секунда", 68), ("1:08", 68), ("3,5", 3.5), ("22", 22)):
            value = parse_timepoint(text)
            self.assertEqual(value["timestamp_sec"], expected)
            self.assertEqual(value["raw"], text)
            self.assertIsNone(value["event_start_sec"])
            self.assertIsNone(value["event_end_sec"])

    def test_ambiguous_missing_not_negative_or_zero(self):
        for text in ("n\\a", "n/a", "", "10 or 20", "1:80", "-1", "NaN", "1 min 80 sec"):
            self.assertIsNone(parse_timepoint(text)["timestamp_sec"])

    def test_note_provenance_and_na(self):
        original = source([["n\\a", "Х-3", "n\\a", "n\\a", "scenario", ""],
                           ["сейф оставила открытым", "X-20", "5", "", "", ""], ["", "", "", "", "", ""]])
        before = json.dumps(original)
        notes, ignored = import_notes(original)
        self.assertEqual(json.dumps(original), before)
        self.assertEqual(len(notes), 2)
        self.assertEqual(len(ignored), 1)
        self.assertEqual(notes[0]["verdict"], "unknown")
        self.assertEqual(notes[1]["candidate_rule_ids"], ["safe_closed"])
        for note in notes:
            self.assertEqual(note["origin"], "HUMAN_SOURCE")
            self.assertFalse(note["verified"])
            self.assertEqual(note["review_state"], "pending")
            self.assertEqual(note["provenance"]["revision_id"], "revision")
            self.assertEqual(note["id"], note["note_id"])

    def test_mapping_does_not_invent_customer_departure_or_contents(self):
        cash = candidate_rules("оставила сумму денег на столе и ушла с место работы", "")
        self.assertIn("unattended_assets", cash["candidate_rule_ids"])
        self.assertNotIn("cash_after_client", cash["candidate_rule_ids"])
        bag = candidate_rules("Клиент просто пришел и пакет денег отдал и ушел не показав документы", "")
        self.assertIn("bag_handover_claim", bag["candidate_claim_types"])
        self.assertIn("original_id_verification", bag["candidate_rule_ids"])
        self.assertEqual(candidate_rules("", "Personal belongings in the cash unit")["candidate_rule_ids"], ["permitted_desk_equipment"])

    def test_duplicate_note_difference_requires_review_not_verified_conflict(self):
        videos = [{"video_id": "X4", "file_name": "X-4.mov", "source_sha256": "same", "notes":
                   [{"timepoint": {"timestamp_sec": 57}, "description": "a", "category": ""}]},
                  {"video_id": "X47", "file_name": "X-47.mov", "source_sha256": "same", "notes":
                   [{"timepoint": {"timestamp_sec": 68}, "description": "b", "category": ""}]}]
        group = duplicate_groups(videos)[0]
        self.assertEqual(group["label_conflict_status"], "different_notes_require_review")
        self.assertFalse(group["confirmed_contradiction"])

    def test_sequential_decode_all_frames_and_nearest_note_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "X-1.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
            self.assertTrue(writer.isOpened())
            for i in range(35):
                frame = np.full((64, 96, 3), i * 5, dtype=np.uint8)
                cv2.putText(frame, str(i), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
                writer.write(frame)
            writer.release()
            notes, _ = import_notes(source([["door", "X1", "1.2", "", "", ""], ["late", "X1", "99", "", "", ""],
                                            ["last frame interval", "X1", "3.45", "", "", ""]]))
            output = root / "review"
            result = decode_video(video, notes, output)
            self.assertEqual(result["decode_proof"]["decoded_frames"], 35)
            self.assertEqual(result["samples"][0]["frame_index"], 0)
            self.assertEqual(result["samples"][-1]["frame_index"], 34)
            ledger = [json.loads(line) for line in (output / result["decode_proof"]["frame_ledger"]).read_text().splitlines()]
            self.assertEqual([line["frame_index"] for line in ledger], list(range(35)))
            self.assertAlmostEqual(notes[0]["timepoint"]["nearest_sample"]["timestamp_sec"], 1.2)
            self.assertEqual(notes[1]["timepoint"]["validity"], "out_of_range")
            self.assertNotIn("nearest_sample", notes[1]["timepoint"])
            self.assertEqual(notes[2]["timepoint"]["validity"], "in_range")
            self.assertEqual(notes[2]["timepoint"]["nearest_sample"]["frame_index"], 34)
            for sample in result["samples"]:
                for field in ("full_frame", "thumbnail", "desk_roi"):
                    self.assertTrue((output / sample[field]).is_file())
            self.assertFalse(result["model_inference_performed"])

    def test_catalog_preserves_original_notes_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos"
            videos.mkdir()
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source([["n/a", "X3", "n/a", "n/a", "", ""]])), encoding="utf-8")
            output = root / "review"
            result = build_catalog(source_path, videos, output)
            self.assertEqual(source_path.read_bytes(), (output / "source_notes.original.json").read_bytes())
            self.assertEqual(result["policy_overrides"]["door_open_threshold_sec"], 3.5)
            self.assertEqual(result["policy_overrides"]["document_threshold_sec"], 30)
            self.assertEqual(result["split"], "development")
            self.assertEqual(len(result["unmatched_notes"]), 1)
            self.assertFalse(result["semantic_review_complete"])


if __name__ == "__main__":
    unittest.main()
