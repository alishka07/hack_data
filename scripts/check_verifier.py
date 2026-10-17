"""Check installed GGUF assets and actual four-frame structured inference locally.

Uses synthetic images only. CPU mode permits a compatibility check without
competing with the resident production GPU model. It is not an accuracy test.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from worker.runtime import LocalRuntime
from worker.video import AnalysisError, Budget, FrameRef


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18184")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    os.environ.update(FORTE_MODEL_PATH=str(args.model.resolve()), FORTE_MMPROJ_PATH=str(args.projector.resolve()),
                      FORTE_LLM_URL=args.url, FORTE_VLM_BACKEND="llama.cpp", FORTE_ATTACH_MODEL="0")
    if args.cpu:
        os.environ.update(FORTE_GPU_LAYERS="0", FORTE_VLM_CPU_PROJECTOR="1")
    directory = ROOT / "data" / "verifier-smoke" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(4):
        pixels = np.zeros((128, 128, 3), np.uint8)
        pixels[20:100, 20:100] = (0, 0, 220) if i % 2 == 0 else (0, 220, 0)
        path = directory / f"synthetic-{i}.jpg"
        if not cv2.imwrite(str(path), pixels):
            raise RuntimeError("Cannot write the synthetic verifier inputs")
        frames.append(FrameRef(i, float(i), path))
    properties = {"rule_id": {"type": "string", "enum": ["synthetic_visual_test"]},
                  "state": {"type": "string", "enum": ["visible", "not_visible"]},
                  "frame_indices": {"type": "array", "maxItems": 4, "items": {"type": "integer", "minimum": 0, "maximum": 3}},
                  "confidence": {"type": "number", "minimum": 0, "maximum": 1}}
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    runtime = LocalRuntime(ROOT, directory)
    report = {"purpose": "Actual local model/vision-projector and schema compatibility, not accuracy",
              "synthetic_frames": 4, "cpu_requested": args.cpu, "status": "failed"}
    started = time.perf_counter()
    try:
        runtime.start()
        report["startup_sec"] = round(time.perf_counter() - started, 3)
        began = time.perf_counter()
        answer = runtime.ask("Inspect only these four synthetic images. Use rule_id synthetic_visual_test. "
                             "Are colored squares visible? Cite their supplied frame indices. "
                             "Use not_visible when the requested observation cannot be seen.",
                             frames, schema, Budget(time.time() + 90), 128)
        report.update(status="verified", inference_sec=round(time.perf_counter() - began, 3),
                      runtime=runtime.metadata(), structured_response=answer,
                      basic_observation_passed=answer["state"] == "visible" and set(answer["frame_indices"]) == {0, 1, 2, 3})
    except AnalysisError as exc:
        report.update(error_code=exc.code, runtime=runtime.metadata())
    finally:
        runtime.stop()
        report["total_sec"] = round(time.perf_counter() - started, 3)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "startup_sec", "inference_sec", "total_sec", "error_code") if key in report}))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
