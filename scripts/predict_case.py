"""Make an offline binary dataset prediction from video pixels and fitted weights.

The score is uncalibrated. This command does not predict a violation category,
event time, bounding box, or legal conclusion. Run in the Torch training env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.trained_detector import predict_video
from worker.video import AnalysisError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/case-detector/model.json")
    parser.add_argument("--encoder", type=Path, default=ROOT / "models/dinov2")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--experimental", action="store_true",
                        help="Explicitly allow research inference with a model that failed its approval gate")
    args = parser.parse_args(argv)
    try:
        result = predict_video(args.video, args.model, args.encoder,
                               threads=args.threads, experimental=args.experimental)
    except (OSError, ValueError, ImportError, RuntimeError, AnalysisError) as exc:
        print(json.dumps({"error": str(exc)}, allow_nan=False), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
