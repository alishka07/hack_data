"""Decode only video, with authoritative source presentation timestamps."""
from __future__ import annotations
from dataclasses import dataclass
import bisect
import json
import os
from pathlib import Path
import subprocess
import time


class AnalysisError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Budget:
    def __init__(self, deadline: float):
        self.deadline = deadline

    def remaining(self, reserve: float = 0) -> float:
        remaining = self.deadline - time.time() - reserve
        if remaining <= 0:
            raise AnalysisError("deadline_exceeded", "The video could not be fully processed before its deadline")
        return remaining


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    pts: tuple[float, ...]
    first_pts: float


@dataclass(frozen=True)
class FrameRef:
    index: int
    timestamp: float
    path: Path
    view: str = "full scene"
    crop_bbox: tuple[float, float, float, float] | None = None


def _run(command: list[str], budget: Budget, limit: float = 30) -> bytes:
    try:
        result = subprocess.run(command, capture_output=True, timeout=min(limit, budget.remaining(2)))
    except subprocess.TimeoutExpired as exc:
        raise AnalysisError("decode_timeout", "Local video decoding exceeded its time budget") from exc
    except FileNotFoundError as exc:
        raise AnalysisError("decoder_missing", "FFmpeg/ffprobe is not available") from exc
    if result.returncode:
        raise AnalysisError("decode_failed", "The video stream could not be decoded locally")
    return result.stdout


def probe_video(path: Path, budget: Budget) -> VideoInfo:
    probe = os.environ.get("FORTE_FFPROBE", "ffprobe")
    raw = _run([probe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], budget)
    try:
        data = json.loads(raw)
        stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
        duration = float(stream.get("duration") or data["format"]["duration"])
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, ValueError, StopIteration, TypeError) as exc:
        raise AnalysisError("invalid_video", "No valid video stream was found") from exc
    if not (0 < duration < 86400 and 0 < width <= 16384 and 0 < height <= 16384):
        raise AnalysisError("invalid_video", "Invalid video dimensions or duration")
    raw = _run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "frame=best_effort_timestamp_time", "-of", "json", str(path)], budget, 35)
    try:
        frames = json.loads(raw)["frames"]
        absolute = tuple(float(f["best_effort_timestamp_time"]) for f in frames)
    except (KeyError, ValueError, TypeError) as exc:
        raise AnalysisError("timestamps_missing", "Source frame timestamps are unavailable") from exc
    if not absolute or any(b <= a for a, b in zip(absolute, absolute[1:])):
        raise AnalysisError("invalid_timestamps", "Video timestamps are empty or non-monotonic")
    first = absolute[0]
    normalized = tuple(t - first for t in absolute)
    # Some containers include an absolute stream start in their duration.
    duration = max(normalized[-1], min(duration, normalized[-1] + 1.0))
    return VideoInfo(path, duration, width, height, normalized, first)


def nearest_index(pts: tuple[float, ...], target: float) -> int:
    position = bisect.bisect_left(pts, target)
    if position == 0:
        return 0
    if position == len(pts):
        return position - 1
    return min((position - 1, position), key=lambda i: abs(pts[i] - target))


def window_targets(duration: float, span: float = 12, hop: float = 8, count: int = 8) -> list[list[float]]:
    windows = []
    start = 0.0
    while start < duration:
        end = min(start + span, duration)
        windows.append([start + (end - start) * i / (count - 1) for i in range(count)])
        if end >= duration:
            break
        start += hop
    return windows


def extract_frames(info: VideoInfo, targets: list[float], folder: Path,
                   budget: Budget, height: int = 448) -> list[FrameRef]:
    """One local decode selects source frame indices; JPEGs retain their source PTS mapping."""
    folder.mkdir(parents=True, exist_ok=True)
    if not targets:
        return []
    requested = [nearest_index(info.pts, target) for target in targets]
    indices = sorted(set(requested))
    # FFmpeg's expression parser has a recursion limit: a flat 100+ term sum
    # becomes a deep left-associated tree. Balance it for long-video sampling.
    def balanced(items: list[int]) -> str:
        if len(items) == 1:
            return f"eq(n,{items[0]})"
        middle = len(items) // 2
        return f"({balanced(items[:middle])}+{balanced(items[middle:])})"
    expression = balanced(indices)
    scale = f",scale=-2:{height}" if height else ""
    destination = folder / "%06d.jpg"
    _run([os.environ.get("FORTE_FFMPEG", "ffmpeg"), "-nostdin", "-v", "error", "-threads", "2",
          "-i", str(info.path), "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
          f"select='{expression}'{scale}", "-fps_mode", "vfr", "-q:v", "2", "-y", str(destination)], budget, 45)
    paths = [folder / f"{i + 1:06d}.jpg" for i in range(len(indices))]
    if any(not p.is_file() or p.stat().st_size == 0 for p in paths):
        raise AnalysisError("frame_extraction_failed", "A selected source frame could not be exported")
    lookup = {index: FrameRef(index, info.pts[index], path) for index, path in zip(indices, paths)}
    return [lookup[index] for index in requested]


class FrameCache:
    """Reuse decoded source-index/size pairs across temporal verification requests."""
    def __init__(self, info: VideoInfo, directory: Path, budget: Budget):
        self.info, self.directory, self.budget = info, directory, budget
        self.frames: dict[tuple[int, int], FrameRef] = {}
        self.batch = 0
        self.crops: dict[tuple, FrameRef] = {}

    def seed(self, frames: list[FrameRef], height: int = 448) -> None:
        self.frames.update({(frame.index, height): frame for frame in frames})

    def get(self, targets: list[float], height: int = 640) -> list[FrameRef]:
        indices = [nearest_index(self.info.pts, target) for target in targets]
        missing = sorted({i for i in indices if (i, height) not in self.frames})
        if missing:
            self.batch += 1
            frames = extract_frames(self.info, [self.info.pts[i] for i in missing],
                                    self.directory / f"batch-{self.batch}-{height}", self.budget, height)
            self.seed(frames, height)
        return [self.frames[(i, height)] for i in indices]

    def crop(self, frames: list[FrameRef], bbox: list[float], label: str = "interaction region") -> list[FrameRef]:
        """Crop original pixels without changing source identity or inventing detail."""
        import math
        import cv2
        if (len(bbox) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)
                or not 0 <= bbox[0] < bbox[2] <= 1 or not 0 <= bbox[1] < bbox[3] <= 1):
            raise AnalysisError("invalid_crop", "A detailed view requires a valid normalized source region")
        originals = self.get([frame.timestamp for frame in frames], 0)
        result = []
        for frame in originals:
            self.budget.remaining(2)
            key = (frame.index, *bbox)
            if key not in self.crops:
                image = cv2.imread(str(frame.path))
                if image is None:
                    raise AnalysisError("frame_extraction_failed", "A source image could not be read for its detailed view")
                height, width = image.shape[:2]
                left, top = int(bbox[0] * width), int(bbox[1] * height)
                right, bottom = math.ceil(bbox[2] * width), math.ceil(bbox[3] * height)
                detail = image[top:bottom, left:right]
                path = self.directory / "crops" / f"crop-{len(self.crops):06d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                if detail.size == 0 or not cv2.imwrite(str(path), detail, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise AnalysisError("frame_extraction_failed", "A detailed source view could not be saved")
                self.crops[key] = FrameRef(frame.index, frame.timestamp, path, label, tuple(bbox))
            result.append(self.crops[key])
        return result
