"""Build a source-bound development review catalog from human video notes.

All source frames are decoded in order; selected frames are saved for review.
No detector runs, human notes are not verified findings, and a time point is not
an event boundary. Original video files and the imported notes stay unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from video_inventory import VIDEO_SUFFIXES, digest_file, finite, font, natural_key, save_frame

ROOT = Path(__file__).resolve().parent
VERSION = "1.0"
UNKNOWN_MARKS = {"", "n/a", "n\\a", "na", "неясно", "неизвестно", "—", "-"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_video_id(value):
    text = Path(str(value).strip()).stem
    match = re.fullmatch(r"[xх]\s*-?\s*(\d+)", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized video reference: {value!r}")
    return f"X{int(match.group(1))}"


def parse_timepoint(value):
    """Strict parsing: ambiguity/missing values are not zero or negative labels."""
    raw = str(value)
    text = raw.strip().lower().replace(",", ".")
    result = {"raw": raw, "timestamp_sec": None, "status": "ambiguous", "kind": "point",
              "event_start_sec": None, "event_end_sec": None}
    if text in UNKNOWN_MARKS:
        result["status"] = "unspecified"
        return result
    seconds = None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        seconds = float(text)
    elif re.fullmatch(r"\d+:\d{1,2}(?:\.\d+)?", text):
        minute, second = text.split(":")
        if float(second) < 60:
            seconds = int(minute) * 60 + float(second)
    else:
        match = re.fullmatch(r"(?:(\d+)\s*(?:min(?:ute)?s?|мин(?:ут[аы]?)?)\s*)?"
                             r"(\d+(?:\.\d+)?)\s*(?:sec(?:ond)?s?|сек(?:унд[аы]?)?)", text)
        if match and (not match.group(1) or float(match.group(2)) < 60):
            seconds = int(match.group(1) or 0) * 60 + float(match.group(2))
    if seconds is not None and math.isfinite(seconds) and seconds >= 0:
        result.update(timestamp_sec=seconds, status="parsed")
    return result


def candidate_rules(description, category):
    """Conservative search routing, not a conversion to verified event labels."""
    text = (description + " " + category).strip().lower()
    rules, claims, cautions = [], [], []

    def add(rule, claim, caution=None):
        if rule not in rules:
            rules.append(rule)
        claims.append(claim)
        if caution and caution not in cautions:
            cautions.append(caution)

    if any(s in text for s in ("без пересчета", "без пересчёта")) and any(s in text for s in ("обратно", "возврат", "остат")):
        add("change_through_counter", "return_without_counting_claim")
    elif any(s in text for s in ("не посчитала", "потом посчитала", "сразу не посчитала")):
        add("cash_counted_visible", "cash_counting_sequence_claim")
    if any(s in text for s in ("документ не предоставил", "не показав документы", "не показал документ")):
        add("original_id_verification", "id_not_presented_claim",
            "Video alone cannot establish originality, ownership, or successful identity verification.")
    if any(s in text for s in ("начала обслуживать другого", "след клиенту", "не сохранении очереди")):
        add("one_client_at_a_time", "overlapping_service_claim")
    if "ключ" in text and "оставила" in text:
        add("safe_closed", "key_left_in_safe_claim", "A closed safe door does not establish a locked safe.")
    elif "ключ" in text and any(s in text for s in ("не закрыла", "вытащила")):
        add("safe_closed", "safe_not_locked_claim", "Key movement alone does not establish lock state.")
    elif "сейф" in text and any(s in text for s in ("открытым", "открыт")):
        add("safe_closed", "safe_open_claim")
    if "свет не отключила" in text:
        add("lights_after_departure", "light_left_on_claim", "No verified lighting policy is established by this note.")
    if "двер" in text and "сейф" not in text and "открыт" in text:
        add("door_open", "door_open_claim", "The note gives a time point, not an opening duration; active user threshold is 3.5 seconds.")
    if any(s in text for s in ("ушла из кабинета", "вышла с кабинета", "ушла с место работы", "without supervision")):
        add("unattended_assets", "assets_or_processing_without_attendance_claim",
            "Cashier departure is distinct from customer departure; supervision outside the view is unknown.")
    if "ноутбук" in text and any(s in text for s in ("доступ", "включенный")):
        add("unattended_assets", "workstation_access_claim", "A visible screen does not establish database access or session lock state.")
    if any(s in text for s in ("второй человек в рабочее", "посторонний человек")):
        add("restricted_access_card_transfer", "additional_person_at_cashier_claim",
            "This note alone does not assert a card transfer, card ownership, or the person's access permission.")
    if "personal belongings" in text:
        add("permitted_desk_equipment", "personal_belongings_claim", "The note does not specify the item class or establish ownership.")
    if "unsecured" in text or "деньги первого клиента в сейф не поставила" in text:
        claims.append("previous_cash_unsecured_claim")
        cautions.append("Unsecured cash does not by itself establish customer departure or unattended assets.")
    if "пакет денег отдал" in text:
        claims.append("bag_handover_claim")
        cautions.append("The content of the bag is a source claim, not visually established by the note.")
    return {"candidate_rule_ids": rules, "candidate_claim_types": claims,
            "mapping_status": "candidate_search_routing" if claims else "unmapped_or_unspecified",
            "mapping_limitations": cautions}


def import_notes(source):
    records, ignored = [], []
    for tab in source.get("tabs", []):
        for table_index, table in enumerate(tab.get("tables", [])):
            for row in table.get("rows", []):
                cells = list(row.get("cells", []))
                if row.get("row_index") == 0:
                    continue
                if not any(str(c).strip() for c in cells):
                    ignored.append({"tab_id": tab.get("tab_id"), "table_index": table_index,
                                    "row_index": row.get("row_index"), "reason": "empty_row", "raw_cells": cells})
                    continue
                padded = cells + [""] * max(0, 6 - len(cells))
                try:
                    video_id = normalize_video_id(padded[1])
                except ValueError:
                    ignored.append({"tab_id": tab.get("tab_id"), "table_index": table_index,
                                    "row_index": row.get("row_index"), "reason": "ambiguous_video_reference", "raw_cells": cells})
                    continue
                identity = f"{tab.get('tab_id')}:{table_index}:{row.get('row_index')}"
                row_id = hashlib.sha256((source.get("document_id", "") + ":" + identity).encode()).hexdigest()[:16]
                record = {"note_id": row_id, "video_id": video_id, "origin": "HUMAN_SOURCE",
                          "review_state": "pending", "verified": False, "split": "development",
                          "annotation_completeness": "non_exhaustive_source_notes",
                          "verdict": "unverified_source_claim", "description": padded[0], "category": padded[3],
                          "scenario": padded[4], "suggested_detection": padded[5], "timepoint": parse_timepoint(padded[2]),
                          "provenance": {"document_id": source.get("document_id"), "document_url": source.get("document_url"),
                                         "revision_id": source.get("revision_id"), "tab_id": tab.get("tab_id"),
                                         "tab_title": tab.get("title"), "table_index": table_index,
                                         "table_start_index": table.get("start_index"), "row_index": row.get("row_index"),
                                         "raw_cells": cells}}
                record["id"] = row_id
                record["timestamp_sec"] = record["timepoint"]["timestamp_sec"]
                if padded[0].strip().lower() in UNKNOWN_MARKS and padded[3].strip().lower() in UNKNOWN_MARKS:
                    record["verdict"] = "unknown"
                record.update(candidate_rules(padded[0], padded[3]))
                records.append(record)
    return records, ignored


def _relative(path, output):
    return Path(path).resolve().relative_to(Path(output).resolve()).as_posix()


def _save_assets(frame, index, timestamp, output, video_dir, reason, notes):
    prefix = f"frame_{index:07d}"
    frame_dir = video_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    full_path = frame_dir / f"{prefix}.jpg"
    thumb_path = frame_dir / f"{prefix}_thumb.jpg"
    crop_path = frame_dir / f"{prefix}_roi.jpg"
    save_frame(full_path, frame)
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    thumb = ImageOps.contain(rgb, (240, 320))
    thumb.save(thumb_path, quality=83)
    width, height = rgb.size
    box = [int(width * .04), int(height * .30), int(width * .96), int(height * .77)]
    rgb.crop(box).save(crop_path, quality=91)
    return {"frame_index": index, "timestamp_sec": timestamp, "timestamp_source": "opencv_source_pos_msec",
            "full_frame": _relative(full_path, output), "thumbnail": _relative(thumb_path, output),
            "desk_roi": _relative(crop_path, output), "roi_xyxy": box,
            "roi_kind": "geometric_middle_lower_review_crop_not_detected_desk",
            "selection_reasons": [reason], "note_ids": list(notes)}


def make_contact_sheets(samples, output, video_dir, video_id, page_size=12):
    paths = []
    for start in range(0, len(samples), page_size):
        subset = samples[start:start + page_size]
        cw, ch, columns = 440, 340, 3
        canvas = Image.new("RGB", (columns * cw, math.ceil(len(subset) / columns) * ch), "#eef1f5")
        draw = ImageDraw.Draw(canvas)
        for i, item in enumerate(subset):
            x, y = i % columns * cw, i // columns * ch
            for kind, offset, bounds in (("thumbnail", 4, (205, 290)), ("desk_roi", 218, (210, 290))):
                with Image.open(output / item[kind]) as original:
                    thumb = ImageOps.contain(original.convert("RGB"), bounds)
                canvas.paste(thumb, (x + offset, y + 5))
            second = f"{item['timestamp_sec']:.3f}s" if item["timestamp_sec"] is not None else "PTS unknown"
            label = f"{video_id} | #{item['frame_index']} | {second}"
            draw.text((x + 5, y + 298), label, font=font(15), fill="#172334")
            reason = "note window" if item["note_ids"] else ", ".join(item["selection_reasons"])
            draw.text((x + 5, y + 319), reason + " | right: geometric ROI", font=font(12), fill="#435468")
        path = video_dir / f"contact_{start // page_size + 1:03d}.jpg"
        canvas.save(path, quality=87)
        paths.append(_relative(path, output))
    return paths


def decode_video(path, notes, output, base_interval=2., dense_interval=.5, window=2.):
    identity = normalize_video_id(path.name)
    video_dir = output / "videos" / identity
    video_dir.mkdir(parents=True, exist_ok=True)
    digest = digest_file(path)
    record = {"video_id": identity, "file_name": path.name, "source_path": str(path.resolve()),
              "source_sha256": digest, "source_bytes": path.stat().st_size, "split": "development",
              "review_state": "pending", "model_inference_performed": False,
              "notes": notes, "samples": [], "contact_sheets": [], "warnings": [],
              "manifest_path": _relative(video_dir / "manifest.json", output)}
    for note in notes:
        note["source_sha256"] = digest
    cap = cv2.VideoCapture(str(path))
    ledger = video_dir / "frames.jsonl"
    if not cap.isOpened():
        record["metadata"] = {"width": 0, "height": 0, "fps": None, "frames": 0, "duration": None}
        record["decode_proof"] = {"status": "unreadable", "decoded_frames": 0, "sequential_decode": True,
                                  "reported_frames": 0, "last_pts": None, "frame_count_matches": False}
        for note in notes:
            note["timepoint"].update(validity="unknown", alignment_reliable=False)
        record["warnings"].append("OpenCV could not open the source.")
        cap.release()
        write_json(video_dir / "manifest.json", record)
        return record
    fps = finite(cap.get(cv2.CAP_PROP_FPS))
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata = {"fps": fps, "frame_count": reported_count,
                "duration_sec": reported_count / fps if fps and fps > 0 else None,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "rotation_degrees": finite(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
                "decoder_backend": cap.getBackendName()}
    metadata["frames"] = metadata["frame_count"]
    metadata["duration"] = metadata["duration_sec"]
    record["metadata"] = metadata
    targets = defaultdict(set)
    for note in notes:
        center = note["timepoint"]["timestamp_sec"]
        if center is None:
            continue
        for step in range(-math.floor(window / dense_interval), math.floor(window / dense_interval) + 1):
            target = max(0., round(center + step * dense_interval, 6))
            targets[target].add(note["note_id"])
    note_targets = sorted(targets)
    cursor = 0
    grid = 0.
    selected = {}
    previous = None
    timestamps = []
    dimensions = set()
    invalid = repeated = regressions = 0
    decode_hash = hashlib.sha256()
    frames = 0
    error = None

    def keep(item, reason, associated=()):
        index, timestamp, frame = item
        if index not in selected:
            selected[index] = _save_assets(frame, index, timestamp, output, video_dir, reason, associated)
        else:
            saved = selected[index]
            if reason not in saved["selection_reasons"]:
                saved["selection_reasons"].append(reason)
            saved["note_ids"] = sorted(set(saved["note_ids"]) | set(associated))

    def nearest(current, target):
        if previous is not None and previous[1] is not None and abs(previous[1] - target) <= abs(current[1] - target):
            return previous
        return current

    try:
        with ledger.open("w", encoding="utf-8") as stream:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or not frame.size:
                    error = "Decoder returned an empty frame; index continuity cannot be guaranteed."
                    break
                pts = finite(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
                current = (frames, pts, frame)
                dimensions.add((int(frame.shape[1]), int(frame.shape[0])))
                decode_hash.update(frame.tobytes())
                stream.write(json.dumps({"frame_index": frames, "timestamp_sec": pts,
                                         "timestamp_source": "opencv_source_pos_msec"}, allow_nan=False) + "\n")
                if pts is None:
                    invalid += 1
                else:
                    if timestamps:
                        delta = pts - timestamps[-1]
                        repeated += abs(delta) <= 1e-6
                        regressions += delta < -1e-6
                    timestamps.append(pts)
                if frames == 0:
                    keep(current, "first_frame")
                if pts is not None:
                    while grid <= pts:
                        keep(nearest(current, grid), "regular_grid")
                        grid = round(grid + base_interval, 6)
                    while cursor < len(note_targets) and note_targets[cursor] <= pts:
                        target = note_targets[cursor]
                        keep(nearest(current, target), "source_note_window", targets[target])
                        cursor += 1
                previous = current
                frames += 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        cap.release()
    if previous is not None:
        keep(previous, "last_frame")
    positive_deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    interval = float(np.median(positive_deltas)) if positive_deltas else (1 / fps if fps and fps > 0 else None)
    duration = timestamps[-1] + interval if timestamps and interval else None
    # A valid time can fall in the final frame's display interval, after its PTS.
    # Attach that point to the final decoded frame rather than an earlier sample.
    if previous is not None and duration is not None:
        while cursor < len(note_targets) and note_targets[cursor] < duration:
            target = note_targets[cursor]
            keep(previous, "source_note_window", targets[target])
            cursor += 1
    timing_ok = frames > 1 and not (invalid or repeated or regressions)
    if frames != reported_count:
        record["warnings"].append("Decoded frame count differs from metadata; EOF may be premature or metadata may be inaccurate.")
    if not timing_ok:
        record["warnings"].append("Timestamps are not all strictly increasing and valid; source-time alignment requires review.")
    if len(dimensions) != 1:
        record["warnings"].append("Changing or unavailable decoded dimensions require review.")
    if error:
        record["warnings"].append(error)
    record["decode_proof"] = {"status": "decoded" if frames and not record["warnings"] else "review_required" if frames else "unreadable",
                              "sequential_decode": True, "frame_skipping": False, "decoded_frames": frames,
                              "reported_frame_count": reported_count, "frame_count_matches": frames == reported_count,
                              "decoded_pixels_sha256": decode_hash.hexdigest(), "frame_ledger": _relative(ledger, output),
                              "frame_ledger_sha256": digest_file(ledger), "first_timestamp_sec": timestamps[0] if timestamps else None,
                              "last_timestamp_sec": timestamps[-1] if timestamps else None, "duration_sec": duration,
                              "source_timeline_end_sec": duration,
                              "decoded_span_sec": (timestamps[-1] - timestamps[0] + interval) if timestamps and interval else None,
                              "negative_preroll_timestamps": sum(t < 0 for t in timestamps),
                              "median_positive_interval_sec": interval, "invalid_timestamps": invalid,
                              "duplicate_timestamps": int(repeated), "timestamp_regressions": int(regressions),
                              "timestamps_strictly_increasing": timing_ok, "decoded_dimensions": [list(d) for d in sorted(dimensions)],
                              "timestamp_semantics": "Zero-based sequential decoded frame index and OpenCV-reported source position in seconds; no FPS-derived replacement. Negative preroll PTS are preserved. duration_sec is the timeline endpoint, while decoded_span_sec includes preroll.",
                              "limit": "Full decoding is technical coverage, not semantic review of every frame. OpenCV EOF cannot rule out codec concealment or every container error."}
    record["decode_proof"]["reported_frames"] = reported_count
    record["decode_proof"]["last_pts"] = record["decode_proof"]["last_timestamp_sec"]
    record["samples"] = sorted(selected.values(), key=lambda row: row["frame_index"])
    for note in notes:
        point = note["timepoint"]
        seconds = point["timestamp_sec"]
        point["validity"] = "unknown" if seconds is None or duration is None else "in_range" if 0 <= seconds < duration else "out_of_range"
        point["alignment_reliable"] = timing_ok and point["validity"] == "in_range"
        point["event_start_sec"] = point["event_end_sec"] = None
        samples = [s for s in record["samples"] if note["note_id"] in s["note_ids"]]
        note["review_window"] = {"requested_start_sec": max(0, seconds - window) if seconds is not None else None,
                                 "requested_end_sec": seconds + window if seconds is not None else None,
                                 "sampling_interval_sec": dense_interval, "kind": "review_context_not_event_boundaries",
                                 "sample_frame_indices": [s["frame_index"] for s in samples]}
        if seconds is not None and samples and point["validity"] == "in_range":
            close = min(samples, key=lambda s: abs(s["timestamp_sec"] - seconds))
            point["nearest_sample"] = {key: close[key] for key in ("frame_index", "timestamp_sec", "full_frame", "thumbnail", "desk_roi")}
            point["nearest_sample"]["time_error_sec"] = close["timestamp_sec"] - seconds
    record["contact_sheets"] = make_contact_sheets(record["samples"], output, video_dir, identity)
    write_json(video_dir / "manifest.json", record)
    return record


def duplicate_groups(videos):
    groups = defaultdict(list)
    for video in videos:
        groups[video["source_sha256"]].append(video)
    result = []
    for sha, rows in groups.items():
        if len(rows) < 2:
            continue
        signatures = []
        for row in rows:
            signatures.append(sorted((n["timepoint"]["timestamp_sec"], n["description"], n["category"])
                                     for n in row["notes"] if n["timepoint"]["timestamp_sec"] is not None))
        different = any(signature != signatures[0] for signature in signatures[1:])
        result.append({"source_sha256": sha, "video_ids": [r["video_id"] for r in rows],
                       "file_names": [r["file_name"] for r in rows], "split_group": "sha256:" + sha,
                       "label_conflict_status": "different_notes_require_review" if different else "no_source_note_difference",
                       "confirmed_contradiction": False, "review_state": "pending",
                       "note": "Byte-identical files must stay in one split. Different points/descriptions are not automatically contradictory; labels are not copied between aliases."})
    return result


def build_catalog(source_path, videos_root, output, base_interval=2., dense_interval=.5, window=2.):
    if any(not math.isfinite(x) or x <= 0 for x in (base_interval, dense_interval)) or not math.isfinite(window) or window < 0:
        raise ValueError("Sampling intervals must be positive and review window nonnegative.")
    source_path, videos_root, output = Path(source_path), Path(videos_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    untouched_path = output / "source_notes.original.json"
    if source_path.resolve() != untouched_path.resolve():
        shutil.copyfile(source_path, untouched_path)
    notes, ignored = import_notes(source)
    by_video = defaultdict(list)
    for note in notes:
        by_video[note["video_id"]].append(note)
    paths = sorted([p for p in videos_root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES], key=natural_key)
    identities = [normalize_video_id(p.name) for p in paths]
    if len(set(identities)) != len(identities):
        raise ValueError("Multiple source files have the same normalized video ID; disambiguate before catalog generation.")
    records = []
    for i, path in enumerate(paths, 1):
        record = decode_video(path, by_video.get(normalize_video_id(path.name), []), output, base_interval, dense_interval, window)
        records.append(record)
        write_json(output / "progress.json", {"completed": len(records), "total": len(paths), "video_ids": [r["video_id"] for r in records]})
        print(f"[{i}/{len(paths)}] {record['video_id']}: {record['decode_proof']['decoded_frames']} frames; {len(record['samples'])} review images", flush=True)
    result = {"schema_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "development_annotation_review", "review_state": "pending", "split": "development",
              "model_inference_performed": False, "semantic_review_complete": False,
              "source": {"document_id": source.get("document_id"), "document_url": source.get("document_url"),
                         "revision_id": source.get("revision_id"), "original_notes": _relative(untouched_path, output),
                         "original_notes_sha256": digest_file(untouched_path)},
              "policy_overrides": {"door_open_threshold_sec": 3.5, "threshold_origin": "explicit_user_conversation",
                                   "document_threshold_sec": 30, "not_a_verified_legal_requirement": True,
                                   "cash_on_desk_rule": "after_customer_departure_only; cashier absence is a separate observation",
                                   "food_rule": "eating_while_cash_present; presence_of_food_drinks_is_separate"},
              "sampling": {"regular_interval_sec": base_interval, "note_window_half_width_sec": window,
                           "note_interval_sec": dense_interval, "first_and_last_frame": True},
              "annotation_limits": ["Imported notes are human source claims, not independently verified labels or model findings.",
                                    "A source timestamp is a point for review, not event start/end or an opening duration.",
                                    "n/a and missing notes are unknown, never negative examples.",
                                    "All available footage is development data; this catalog creates no held-out evaluation set.",
                                    "ROI crops are geometric review aids, not confirmed desk or object annotations.",
                                    "Sequential decoding of every frame does not mean semantic review of every frame."],
              "video_count": len(records), "note_count": len(notes), "decoded_frame_count": sum(r["decode_proof"]["decoded_frames"] for r in records),
              "sample_count": sum(len(r["samples"]) for r in records), "videos": records,
              "duplicate_groups": duplicate_groups(records), "ignored_source_rows": ignored,
              "unmatched_notes": [n for n in notes if n["video_id"] not in identities]}
    write_json(output / "catalog.json", result)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "annotations" / "source_notes_2026_10_17.json")
    parser.add_argument("--videos", type=Path, default=ROOT / "videos")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "video_review")
    parser.add_argument("--interval", type=float, default=2.)
    parser.add_argument("--dense-interval", type=float, default=.5)
    parser.add_argument("--window", type=float, default=2.)
    args = parser.parse_args()
    result = build_catalog(args.source, args.videos, args.output, args.interval, args.dense_interval, args.window)
    print(json.dumps({key: result[key] for key in ("video_count", "note_count", "decoded_frame_count", "sample_count", "duplicate_groups")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
