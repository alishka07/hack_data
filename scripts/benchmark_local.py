"""Exercise the real loopback service without printing video-derived content.

Results contain technical counts/timings only. Full observations stay in data/.
This checks operation and the jury contract, not detection accuracy.
"""
from __future__ import annotations
import argparse
import base64
import ipaddress
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import httpx
import numpy as np
from worker.video import Budget, nearest_index, probe_video


def validate_result(result: dict, info) -> int:
    assert set(result) == {"job_id", "status", "verdict", "violations"}
    assert result["status"] == "done"
    assert result["verdict"] in ("violation", "no_violation")
    assert bool(result["violations"]) == (result["verdict"] == "violation")
    evidence_count = 0
    for finding in result["violations"]:
        assert set(finding) == {"type", "timecode_sec", "description", "evidence_frames"}
        assert isinstance(finding["type"], str) and finding["type"]
        assert isinstance(finding["description"], str) and finding["description"]
        assert 0 <= finding["timecode_sec"] <= info.duration
        assert 1 <= len(finding["evidence_frames"]) <= 3
        for frame in finding["evidence_frames"]:
            assert set(frame) == {"timestamp_sec", "image_base64"}
            timestamp = frame["timestamp_sec"]
            assert 0 <= timestamp <= info.duration
            assert abs(info.pts[nearest_index(info.pts, timestamp)] - timestamp) < 1e-6
            jpeg = base64.b64decode(frame["image_base64"], validate=True)
            assert jpeg.startswith(b"\xff\xd8")
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and image.shape[:2] == (info.height, info.width)
            evidence_count += 1
    return evidence_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--selection", type=Path, help="Local evaluation manifest; only video bytes reach inference")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--key-file", type=Path, default=ROOT / "data/api-key.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/local-integration.json")
    args = parser.parse_args()
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        rows = [selection["longest_sample"]] + selection["validation_samples"]
        selected = set()
        for row in rows:
            if row["sha256"] not in selected:
                args.videos.append(Path(row["video_path"]))
                selected.add(row["sha256"])
    if not args.videos:
        parser.error("Provide videos or a local --selection manifest")
    parsed = urlparse(args.url)
    assert parsed.scheme == "http" and parsed.hostname and ipaddress.ip_address(parsed.hostname).is_loopback
    assert not parsed.username and not parsed.password and parsed.path in ("", "/")
    key = args.key_file.read_text(encoding="utf-8").strip()
    headers = {"X-API-Key": key}
    records = []
    with httpx.Client(base_url=args.url, timeout=30, trust_env=False, follow_redirects=False) as client:
        ready_by = time.monotonic() + 180
        while not client.get("/api/v1/health").json()["worker_ready"]:
            if time.monotonic() >= ready_by:
                raise RuntimeError("Worker readiness timeout")
            time.sleep(1)
        assert client.get("/api/ui/rules").status_code == 401
        invalid = client.post("/api/v1/analyze", headers=headers,
                              files={"file": ("input.mov", b"invalid synthetic media", "video/quicktime")})
        assert invalid.status_code == 400
        for path in args.videos:
            info = probe_video(path.resolve(), Budget(time.time() + 45))
            with path.open("rb") as video:
                began = time.monotonic()
                response = client.post("/api/v1/analyze", headers=headers,
                                       files={"file": ("input.mov", video, "video/quicktime")})
            assert response.status_code == 202, f"Admission returned HTTP {response.status_code}"
            assert set(response.json()) == {"job_id"}
            job_id = response.json()["job_id"]
            busy = client.post("/api/v1/analyze", headers=headers,
                               files={"file": ("input.mov", b"synthetic", "video/quicktime")})
            assert busy.status_code == 429
            while True:
                result = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
                if result["status"] != "processing":
                    break
                assert set(result) == {"job_id", "status"}
                if time.monotonic() - began > 320:
                    raise RuntimeError("Accepted job exceeded its deadline and recovery allowance")
                time.sleep(1)
            elapsed = time.monotonic() - began
            details = client.get(f"/api/ui/jobs/{job_id}", headers=headers).json()
            record = {"duration_sec": round(info.duration, 3), "status": result["status"],
                      "upload_to_result_sec": round(elapsed, 3), "under_300_sec": elapsed < 300}
            if result["status"] == "done":
                record["evidence_frames_validated"] = validate_result(result, info)
                record["finding_count"] = len(result["violations"])
                coverage = details["details"]["coverage"]
                assert coverage["windows_completed"] == coverage["windows_planned"]
                record["windows_completed"] = coverage["windows_completed"]
                record["pipeline_timings"] = details["timings"]
                record["pipeline_version"] = details["details"].get("versions", {}).get("pipeline", "unknown")
                record["review_queue_count"] = len(details["details"].get("review_queue", []))
            else:
                assert set(result) == {"job_id", "status"} and result["status"] == "failed"
                record["error_code"] = details.get("error", {}).get("code", "unknown")
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"purpose": "In-progress local technical benchmark", "samples": records}, indent=2) + "\n", encoding="utf-8")
    summary = {"purpose": "Real local service operation; not an accuracy evaluation", "samples": records,
               "checks": {"authentication": True, "corrupt_input_rejected": True, "busy_rejected": True,
                          "exact_jury_schema": True,
                          "emitted_evidence_frames_validated": sum(r.get("evidence_frames_validated", 0) for r in records),
                          "evidence_check_scope": "Every emitted frame is checked; zero frames does not test real positive-event evidence"},
               "all_completed": all(r["status"] == "done" for r in records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["all_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
