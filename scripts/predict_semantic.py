"""Predict a dataset label with the v3 semantic model and a local Qwen runtime.

By default this owns a new local server and fails if its port is occupied.
--attach explicitly trusts an idle existing server using the pinned recipe.
No category, event time, bounding box or legal conclusion is produced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.semantic_runtime import LocalRuntime
from worker.semantic_detector import SemanticDetector
from worker.video import AnalysisError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/case-detector-v3/model.json")
    parser.add_argument("--experimental", action="store_true",
                        help="Explicitly allow research inference with a model that failed its approval gate")
    parser.add_argument("--attach", action="store_true",
                        help="Explicitly attach to a trusted idle local model server without owning or stopping it")
    args = parser.parse_args(argv)
    runtime, started = None, False
    try:
        if not args.video.is_file():
            raise FileNotFoundError("The supplied video file does not exist")
        previous_attach = os.environ.get("FORTE_ATTACH_MODEL")
        try:
            if args.attach:
                os.environ["FORTE_ATTACH_MODEL"] = "1"
            runtime = LocalRuntime(ROOT, ROOT / "data")
        finally:
            if previous_attach is None:
                os.environ.pop("FORTE_ATTACH_MODEL", None)
            else:
                os.environ["FORTE_ATTACH_MODEL"] = previous_attach
        detector = SemanticDetector(ROOT, args.model, runtime, experimental=args.experimental)
        try:
            runtime.start()
            started = True
            result = detector.predict(args.video)
        finally:
            if not runtime.external and (started or runtime.process is not None or runtime.log is not None):
                runtime.stop()
    except (OSError, ValueError, ImportError, RuntimeError, AnalysisError) as exc:
        print(json.dumps({"error": str(exc)}, allow_nan=False), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"error": "Prediction interrupted"}), file=sys.stderr)
        return 130
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
