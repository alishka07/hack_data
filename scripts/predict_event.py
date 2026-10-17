"""Predict a dataset label with the frozen v2 action/event model.

The uncalibrated score and highest-scoring input window are estimates, not a
category, reviewed event interval, bounding box or legal conclusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.event_detector import EventDetector
from worker.video import AnalysisError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/case-detector-v2/model.json")
    parser.add_argument("--encoder", type=Path, default=ROOT / "models/action-encoder")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--experimental", action="store_true",
                        help="Explicitly allow research inference with a model that failed its approval gate")
    args = parser.parse_args(argv)
    try:
        detector = EventDetector(args.model, args.encoder, threads=args.threads, experimental=args.experimental)
        result = detector.predict(args.video)
    except (OSError, ValueError, ImportError, RuntimeError, AnalysisError) as exc:
        print(json.dumps({"error": str(exc)}, allow_nan=False), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
