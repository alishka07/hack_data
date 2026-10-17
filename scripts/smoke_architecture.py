"""Run the real local pipeline on one source; publish only aggregate diagnostics.

Labels, filenames and annotation metadata never reach the model. An attached
server is never stopped by this script. This checks operation, not accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

from scripts.benchmark_local import validate_result
from worker.pipeline import PIPELINE_VERSION, Pipeline
from worker.runtime import LocalRuntime
from worker.video import AnalysisError, Budget, probe_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attach-model", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.)
    args = parser.parse_args()
    if not 0 < args.timeout <= 900:
        parser.error("--timeout must be between zero and 900 seconds; deployment remains 300 seconds")
    if args.attach_model:
        os.environ["FORTE_ATTACH_MODEL"] = "1"
    source = args.source.resolve()
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    info = probe_video(source, Budget(time.time() + 60))
    directory = ROOT / "data" / "architecture-smoke" / uuid.uuid4().hex
    runtime = LocalRuntime(ROOT, directory)
    report = {"purpose": "Real pipeline operation, not an accuracy evaluation",
              "pipeline_version": PIPELINE_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": digest, "duration_sec": info.duration,
              "timeout_sec": args.timeout, "runtime": runtime.metadata()}
    started = time.perf_counter()
    try:
        runtime.start()
        report["runtime"] = runtime.metadata()
        def progress(stage, amount, _message):
            print(json.dumps({"stage": stage, "progress": round(amount, 3)}), flush=True)
        result, details, timings = Pipeline(ROOT, directory, runtime).run(
            {"id": "smoke", "video_path": str(source), "deadline_at": time.time() + args.timeout}, progress)
        emitted = validate_result({"job_id": "smoke", "status": "done", **result}, info)
        declared = details.get("evidence", [])
        for frame in declared:
            image = cv2.imread(str(directory / "evidence" / "smoke" / frame["filename"]))
            if image is None:
                raise AnalysisError("evidence_failed", "A declared JPEG cannot be decoded")
        report.update(status="done", verdict=result["verdict"], finding_count=len(result["violations"]),
                      review_count=len(details["review_queue"]), candidate_count=timings["candidate_count"],
                      event_proposals=timings["event_proposal_count"], detection_count=timings["detection_count"],
                      declared_jpegs_validated=len(declared), emitted_jpegs_validated=emitted,
                      coverage=details["coverage"], capabilities=details["perception"].get("capabilities", {}),
                      scenario_coverage=details.get("scenario_coverage", {}),
                      supported_visual_scenarios=sum(row.get("visual_scenario_supported", False) for row in details["review_queue"]),
                      calibration=details["calibration"], timings=timings)
    except AnalysisError as exc:
        report.update(status="failed", error_code=exc.code)
    except Exception as exc:
        report.update(status="failed", error_code="unexpected_" + type(exc).__name__)
        raise
    finally:
        runtime.stop()
        report["wall_sec"] = round(time.perf_counter() - started, 3)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "duration_sec", "wall_sec")}), flush=True)
    return 0 if report["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
