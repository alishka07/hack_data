"""One local visual-observation request per video, without dataset labels.

The caller owns the persistent runtime lifecycle and exclusive model access.
This adapter never starts, stops, or attaches to a model server itself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from worker.runtime import _validate_json_schema
from worker.semantic_features import SEMANTIC_PROMPT, SEMANTIC_VERSION, encode_observables, observable_schema
from worker.video import AnalysisError, Budget, FrameRef, extract_frames, nearest_index, probe_video


ENCODER_VERSION = "qwen3vl_uniform8_height448_cap960_v1"
FRAME_COUNT = 8
FRAME_HEIGHT = 448
MAX_SIDE = 960
MAX_TOKENS = 768


def semantic_recipe() -> dict:
    """Stable cache identity for the complete label-independent observation recipe."""
    schema = observable_schema()
    return {"version": ENCODER_VERSION, "semantic_version": SEMANTIC_VERSION,
            "prompt_sha256": hashlib.sha256(SEMANTIC_PROMPT.encode("utf-8")).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
            "frame_count": FRAME_COUNT, "frame_height": FRAME_HEIGHT, "max_side": MAX_SIDE,
            "sampling": "uniform_timeline_nearest_source_pts_including_endpoints_v1",
            "max_tokens": MAX_TOKENS, "temperature": 0}


class SemanticEncoder:
    def __init__(self, root: Path, data_dir: Path, runtime):
        self.root, self.data_dir, self.runtime = Path(root), Path(data_dir), runtime
        self.recipe = semantic_recipe()

    def clip_features(self, video: Path, budget: Budget | None = None) -> tuple[dict, dict]:
        """Return fixed physical observations and their actual source-frame metadata."""
        budget = budget if budget is not None else Budget(time.time() + 120)
        info = probe_video(Path(video), budget)
        pts = np.asarray(info.pts, dtype=np.float64)
        if (pts.ndim != 1 or not len(pts) or not np.isfinite(pts).all()
                or pts[0] < 0 or np.any(np.diff(pts) <= 0)):
            raise AnalysisError("invalid_timestamps", "Semantic sampling needs finite increasing source timestamps")
        targets = np.linspace(float(pts[0]), float(pts[-1]), min(FRAME_COUNT, len(pts)))
        indices = sorted({nearest_index(info.pts, float(target)) for target in targets})
        directory = self.data_dir / "semantic-tmp"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sample-", dir=directory) as temporary:
            folder = Path(temporary)
            refs = extract_frames(info, [info.pts[index] for index in indices], folder / "source", budget,
                                  height=FRAME_HEIGHT)
            supplied, dimensions = [], []
            for ref in refs:
                image = cv2.imread(str(ref.path), cv2.IMREAD_COLOR)
                if image is None:
                    raise AnalysisError("frame_extraction_failed", "A semantic source frame could not be read")
                height, width = image.shape[:2]
                if max(height, width) > MAX_SIDE:
                    scale = MAX_SIDE / max(height, width)
                    width, height = max(1, round(width * scale)), max(1, round(height * scale))
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    destination = folder / f"capped-{ref.index}.jpg"
                    if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise AnalysisError("frame_extraction_failed", "A semantic source frame could not be resized")
                    ref = FrameRef(ref.index, ref.timestamp, destination)
                supplied.append(ref)
                dimensions.append({"width": width, "height": height})
            schema = observable_schema()
            answer = self.runtime.ask(SEMANTIC_PROMPT, supplied, schema, budget, MAX_TOKENS)
            _validate_json_schema(answer, schema)
            encode_observables(answer)
            metadata = {"timestamps": [float(ref.timestamp) for ref in supplied],
                        "source_indices": [ref.index for ref in supplied], "duration_sec": float(info.duration),
                        "frame_dimensions": dimensions, "recipe": self.recipe}
        return answer, metadata
