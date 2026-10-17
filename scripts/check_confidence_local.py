"""Local confidence smoke test; only counts/timings leave this process.

No labels, video frames, scene text, or private observations are printed.
The manifest supplies paths only; inference receives generic-named video bytes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from scripts.benchmark_local import validate_result
from worker.video import Budget, probe_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longest", action="store_true")
    parser.add_argument("--selection", type=Path, default=ROOT / "data/labels/validation-selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/local-confidence-v2-4-smoke.json")
    parser.add_argument("--pipeline-version", default="2.4")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    candidates = [selection["longest_sample"]] + selection["validation_samples"]
    inspected = [(Path(row["video_path"]), probe_video(Path(row["video_path"]), Budget(time.time() + 30)))
                 for row in candidates]
    path, info = (max if args.longest else min)(inspected, key=lambda pair: pair[1].duration)
    headers = {"X-API-Key": (ROOT / "data/api-key.txt").read_text(encoding="utf-8").strip()}
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30, trust_env=False, follow_redirects=False) as client:
        assert client.get("/api/v1/health").json()["worker_ready"]
        began = time.monotonic()
        with path.open("rb") as video:
            response = client.post("/api/v1/analyze", headers=headers,
                                   files={"file": ("input.mov", video, "video/quicktime")})
        assert response.status_code == 202, f"Admission returned HTTP{response.status_code}"
        job_id = response.json()["job_id"]
        while True:
            result = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
            if result["status"] != "processing":
                break
            if time.monotonic() - began > 320:
                raise RuntimeError("Local confidence smoke test exceeded deadline allowance")
            time.sleep(1)
        metadata = client.get(f"/api/ui/jobs/{job_id}", headers=headers).json()
    record = {"purpose": "Local confidence operation, not accuracy/calibration validation",
              "job_id": job_id,
              "duration_sec": round(info.duration, 3), "status": result["status"],
              "elapsed_sec": round(time.monotonic() - began, 3)}
    if result["status"] == "done":
        record["jury_evidence_validated"] = validate_result(result, info)
        details = metadata["details"]
        rows = details["assessments"]
        assert len(rows) == 32 and details["versions"]["pipeline"] == args.pipeline_version
        assert all(row["calibrated"] is False for row in rows)
        for row in rows + details["assessment_episodes"]:
            confidence = row["model_confidence"]
            if confidence is not None:
                expected = 100 * (confidence if row["scoring_kind"] == "event_occurrence" else 1 - confidence)
                assert abs(row["score"] - expected) < .00001
            for reference in row["evidence_refs"]:
                if "source_index" in reference:
                    assert abs(info.pts[reference["source_index"]] - reference["timestamp_sec"]) < .000001
        record.update(pipeline_version=details["versions"]["pipeline"], scoring_version=details["versions"]["scoring"],
                      catalog_items=len(rows), episode_assessments=len(details["assessment_episodes"]),
                      confidence_observations=metadata["timings"]["confidence_observation_count"],
                      changed_scores=sum(row["score"] != row["initial_score"] for row in rows),
                      fractional_scores=sum(row["score"] not in (None, 0, 100) for row in rows),
                      event_increases=sum(row["scoring_kind"] == "event_occurrence" and row["score"] > 0 for row in rows),
                      procedure_decreases=sum(row["scoring_kind"] == "required_procedure" and row["score"] is not None and row["score"] < 100 for row in rows),
                      detection_count=len(details.get("detections", [])),
                      case_decision_status=details.get("case_decision", {}).get("status"),
                      finding_count=len(result["violations"]), windows=details["coverage"]["windows_completed"])
        record["pipeline_timings"] = metadata["timings"]
    else:
        record["error_code"] = metadata.get("error", {}).get("code", "unknown")
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)
    return 0 if result["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
