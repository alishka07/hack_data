"""Read every frame locally and create an inventory of prerecorded videos.

Run: python video_inventory.py
This script performs no model inference and makes no conclusions about violations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mov", ".mp4", ".mkv", ".avi", ".m4v", ".webm"}


def natural_key(path):
    return [int(part) if part.isdigit() else part.casefold().replace("х", "x")
            for part in re.split(r"(\d+)", path.name)]


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return float(value) if math.isfinite(value) else None


def save_frame(path, frame):
    # imencode/tofile supports Windows Unicode filenames reliably.
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"Unable to encode frame: {path}")
    encoded.tofile(str(path))


def inspect_video(path, output):
    result = {"file": path.name, "path": str(path.resolve()),
              "bytes": path.stat().st_size, "sha256": digest_file(path),
              "decoded_frames": 0, "samples": [], "warnings": []}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        result.update(status="unreadable", error="OpenCV could not open this file")
        cap.release()
        return result
    fps = finite(cap.get(cv2.CAP_PROP_FPS))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata_duration = reported_frames / fps if fps and fps > 0 else None
    result.update(reported_fps=fps, reported_frames=reported_frames,
                  metadata_duration_seconds=metadata_duration,
                  reported_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  reported_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  rotation_metadata_degrees=finite(cap.get(cv2.CAP_PROP_ORIENTATION_META)))
    fractions = (0.15, 0.50, 0.85)
    # Retain only three representative decoded images; no seeking or frame skipping.
    targets = [max(0, int(max(0, reported_frames - 1) * q)) for q in fractions]
    samples = {}
    last_frame = None
    first_pts = last_pts = None
    regressions = duplicate_pts = nonfinite_pts = 0
    intervals = deque(maxlen=10000)
    dimensions = set()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                result["warnings"].append(f"Empty decoded frame at index {frames}")
                continue
            pts = finite(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if pts is None:
                nonfinite_pts += 1
            else:
                if first_pts is None:
                    first_pts = pts
                if last_pts is not None:
                    delta = pts - last_pts
                    if delta < -1e-6:
                        regressions += 1
                    elif abs(delta) <= 1e-6:
                        duplicate_pts += 1
                    else:
                        intervals.append(delta)
                last_pts = pts
            dimensions.add((int(frame.shape[1]), int(frame.shape[0])))
            for sample_index, target in enumerate(targets):
                if frames == target:
                    samples[sample_index] = (frames, pts, frame.copy())
            last_frame = (frames, pts, frame.copy())
            frames += 1
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cap.release()
    if last_frame:
        for sample_index in range(len(targets)):
            samples.setdefault(sample_index, last_frame)
    sample_dir = output / "frames" / path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample_index in sorted(samples):
        frame_index, pts, frame = samples[sample_index]
        sample_path = sample_dir / f"sample_{sample_index + 1}_{frame_index:06d}.jpg"
        save_frame(sample_path, frame)
        result["samples"].append({"fraction_target": fractions[sample_index],
                                  "frame_index": frame_index,
                                  "timestamp_seconds": pts,
                                  "image": str(sample_path.resolve())})
    interval = float(np.median(intervals)) if intervals else (1 / fps if fps and fps > 0 else None)
    decoded_duration = (last_pts - first_pts + interval
                        if last_pts is not None and first_pts is not None and interval else None)
    result.update(decoded_frames=frames, decoded_dimensions=[list(x) for x in sorted(dimensions)],
                  first_timestamp_seconds=first_pts, last_timestamp_seconds=last_pts,
                  decoded_duration_seconds=decoded_duration,
                  median_positive_frame_interval_seconds=interval,
                  timestamp_regressions=regressions, duplicate_timestamps=duplicate_pts,
                  nonfinite_timestamps=nonfinite_pts,
                  timestamps_strictly_increasing=bool(frames > 1 and not (regressions or duplicate_pts or nonfinite_pts)),
                  frame_count_difference=frames - reported_frames)
    if frames != reported_frames:
        result["warnings"].append("Decoded frame count differs from metadata; possible early termination or inaccurate metadata.")
    if regressions or duplicate_pts or nonfinite_pts:
        result["warnings"].append("Source timestamps are not strictly increasing/valid; downstream analysis must handle timestamp uncertainty.")
    if len(dimensions) != 1:
        result["warnings"].append("Decoded dimensions change within this video or no image could be decoded.")
    result["status"] = "unreadable" if not frames else ("review" if result["warnings"] or result.get("error") else "decoded")
    return result


def font(size):
    for candidate in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def contact_sheets(results, output, page_size=14):
    sheets = []
    cell_width, cell_height, columns = 224, 390, 4
    label_font = font(15)
    for start in range(0, len(results), page_size):
        page = results[start:start + page_size]
        canvas = Image.new("RGB", (columns * cell_width, math.ceil(len(page) / columns) * cell_height), "#f4f4f4")
        draw = ImageDraw.Draw(canvas)
        for index, result in enumerate(page):
            x, y = (index % columns) * cell_width, (index // columns) * cell_height
            samples = result["samples"]
            if samples:
                sample = samples[min(1, len(samples) - 1)]
                with Image.open(sample["image"]) as original:
                    thumb = ImageOps.contain(original.convert("RGB"), (cell_width - 12, cell_height - 60))
                canvas.paste(thumb, (x + (cell_width - thumb.width) // 2, y + 4))
                pts = sample["timestamp_seconds"]
                second = f"{pts:.2f}s" if pts is not None else "unknown PTS"
            else:
                second = "unreadable"
            duration = result.get("decoded_duration_seconds")
            duration_text = f"{duration:.2f}s" if duration is not None else "?"
            draw.text((x + 7, y + cell_height - 52), f"{result['file']} @ {second}", font=label_font, fill="black")
            draw.text((x + 7, y + cell_height - 30), f"{result['decoded_frames']} frames | {duration_text}", font=label_font, fill="black")
        path = output / f"contact_sheet_{start // page_size + 1:02d}.jpg"
        canvas.save(path, quality=92)
        sheets.append(str(path.resolve()))
    return sheets


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=ROOT / "videos")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "inventory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted((p for p in args.videos.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES), key=natural_key)
    if not paths:
        raise SystemExit(f"No supported videos in {args.videos}")
    results = []
    for index, path in enumerate(paths, 1):
        result = inspect_video(path, args.output)
        results.append(result)
        print(f"[{index}/{len(paths)}] {path.name}: {result['decoded_frames']} frames, {result['status']}", flush=True)
        (args.output / "inventory_progress.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    groups = {}
    for item in results:
        groups.setdefault(item["sha256"], []).append(item["file"])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "All files decoded sequentially. Three representative frames saved per video. No object or violation inference.",
              "corruption_limit": "A successful decode and matching frame count do not prove absence of codec concealment, corrupt pixels, or all container errors. OpenCV read(False) cannot distinguish normal EOF from every decoding error.",
              "video_count": len(results),
              "decoded_frames": sum(x["decoded_frames"] for x in results),
              "total_decoded_duration_seconds": sum(x.get("decoded_duration_seconds") or 0 for x in results),
              "statuses": {state: sum(x["status"] == state for x in results) for state in ("decoded", "review", "unreadable")},
              "identical_file_groups": [names for names in groups.values() if len(names) > 1],
              "contact_sheets": contact_sheets(results, args.output), "videos": results}
    (args.output / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["file", "status", "bytes", "decoded_frames", "reported_frames", "reported_fps", "decoded_duration_seconds", "reported_width", "reported_height", "timestamp_regressions", "duplicate_timestamps", "nonfinite_timestamps", "sha256"]
    with (args.output / "inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({key: report[key] for key in ("video_count", "decoded_frames", "total_decoded_duration_seconds", "statuses", "identical_file_groups", "contact_sheets")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
