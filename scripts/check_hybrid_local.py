"""Measure the real local hybrid service; print/save aggregate technical data only.

Selection labels are evaluation-only. Only media bytes and the generic filename
input.mov reach inference. These previously inspected clips cannot establish
generalization or calibrated detection quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import httpx
import numpy as np

from scripts.benchmark_local import validate_result
from worker.video import AnalysisError, Budget, nearest_index, probe_video


class CheckError(ValueError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise CheckError(code)


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def loopback_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    require(parsed.scheme == "http" and loopback and not parsed.username and not parsed.password
            and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment, "loopback_url_required")
    return value.rstrip("/")


def select_samples(selection: dict, mode: str, inspect=probe_video) -> list[tuple[dict, object]]:
    rows = [selection["longest_sample"], *selection["validation_samples"]]
    unique = {}
    for row in rows:
        require(isinstance(row, dict) and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))) is not None,
                "invalid_selection")
        require(type(row.get("binary")) is bool and isinstance(row.get("video_path"), str), "invalid_selection")
        digest = row["sha256"]
        if digest in unique:
            require(unique[digest][0]["binary"] == row["binary"], "conflicting_duplicate_labels")
            continue
        path = Path(row["video_path"]).resolve()
        require(path.is_file() and not str(path).startswith(("\\\\", "//")), "local_source_required")
        info = inspect(path, Budget(time.time() + 60))
        unique[digest] = (row, info)
    ordered = sorted(unique.values(), key=lambda pair: (pair[1].duration, pair[0]["sha256"]))
    require(bool(ordered), "empty_selection")
    if mode == "shortest":
        return ordered[:1]
    if mode == "longest":
        return ordered[-1:]
    require(mode == "balanced", "invalid_sample_mode")
    require({row["binary"] for row, _ in ordered} == {False, True}, "balanced_classes_missing")
    return [next(pair for pair in ordered if pair[0]["binary"] == label) for label in (False, True)]


class GpuSampler:
    """Whole-device 1-second samples; neither process isolation nor a true peak."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.samples = 0
        self.maxima: dict[int, dict] = {}
        self.failures = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                        timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                require(result.returncode == 0, "gpu_sampling_unavailable")
                values = []
                for line in result.stdout.splitlines():
                    index, used, total = (int(part.strip()) for part in line.split(","))
                    values.append((index, used, total))
                require(bool(values), "gpu_sampling_unavailable")
                for index, used, total in values:
                    device = self.maxima.setdefault(index, {"gpu_index": index, "sampled_max_mib": used,
                                                           "total_memory_mib": total})
                    device["sampled_max_mib"] = max(device["sampled_max_mib"], used)
                self.samples += 1
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.failures += 1
            self.stop_event.wait(1.)

    def start(self):
        self.thread.start()

    def finish(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=3)
        return {"available": bool(self.samples), "samples": self.samples, "sampling_failures": self.failures,
                "sampled_max_mib": max((row["sampled_max_mib"] for row in self.maxima.values()), default=None),
                "devices": [self.maxima[index] for index in sorted(self.maxima)],
                "scope": "Maximum observed whole-GPU memory per device, sampled approximately once per second. "
                         "Includes other processes; this is not a true peak or model-only memory measurement."}


def source_time(timestamp, info) -> bool:
    return finite(timestamp) and 0 <= timestamp <= info.duration and abs(
        info.pts[nearest_index(info.pts, timestamp)] - timestamp) < 1e-6


def validate_details(client, headers: dict, job_id: str, response: dict, info,
                     expected_version: str, max_images: int) -> dict:
    details = response["details"]
    version = details.get("versions", {}).get("pipeline")
    require(version == expected_version, "pipeline_version_mismatch")
    coverage = details["coverage"]
    require(type(coverage.get("windows_completed")) is int and coverage["windows_completed"] > 0
            and coverage["windows_completed"] == coverage.get("windows_planned"), "incomplete_window_coverage")
    perception = coverage.get("perception", {})
    require(perception.get("complete") is True and perception.get("ending_included") is True,
            "incomplete_perception_coverage")
    require(abs(perception.get("first_timestamp_sec", -1)) < 1e-6
            and source_time(perception.get("last_timestamp_sec"), info)
            and abs(perception["last_timestamp_sec"] - info.pts[-1]) < 1e-6, "perception_endpoint_mismatch")
    declared = {entry["filename"] for entry in details.get("evidence", []) if isinstance(entry, dict)
                and isinstance(entry.get("filename"), str)}
    images = {}
    detections = details.get("detections", [])
    for detection in detections:
        index, timestamp = detection.get("source_index"), detection.get("timestamp_sec")
        require(type(index) is int and 0 <= index < len(info.pts) and source_time(timestamp, info)
                and abs(info.pts[index] - timestamp) < 1e-6, "invalid_detection_source")
        box = detection.get("bbox")
        require(isinstance(box, list) and len(box) == 4 and all(finite(value) for value in box)
                and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1, "invalid_detection_box")
        evidence = detection.get("evidence_frame", {})
        name = evidence.get("filename")
        require(isinstance(name, str) and Path(name).name == name and name in declared
                and source_time(evidence.get("timestamp_sec"), info)
                and abs(evidence["timestamp_sec"] - timestamp) < 1e-6, "undeclared_detection_evidence")
        images.setdefault(name, timestamp)
    detection_image_count = len(images)
    for finding in details.get("findings", []):
        for evidence in finding.get("evidence_frames", []):
            name = evidence.get("filename")
            require(isinstance(name, str) and Path(name).name == name and name in declared
                    and source_time(evidence.get("timestamp_sec"), info), "undeclared_finding_evidence")
            images.setdefault(name, evidence["timestamp_sec"])
    checked, auth_checked = 0, False
    for name in sorted(images):
        if max_images and checked >= max_images:
            break
        url = f"/api/ui/jobs/{job_id}/evidence/{name}"
        if not auth_checked:
            require(client.get(url).status_code == 401, "evidence_authentication_failed")
            auth_checked = True
        image_response = client.get(url, headers=headers)
        require(image_response.status_code == 200 and image_response.content.startswith(b"\xff\xd8"),
                "evidence_download_failed")
        image = cv2.imdecode(np.frombuffer(image_response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(image is not None and image.shape[0] > 0 and image.shape[1] > 0,
                "evidence_jpeg_invalid")
        checked += 1
    decision = details.get("case_decision", {})
    review_required = decision.get("status") == "review_required" or bool(details.get("review_queue"))
    timing_fields = ("total_sec", "decode_and_cache_sec", "perception_sec", "vlm_sec", "evidence_output_sec",
                     "probe_and_sampling_sec", "screening_sec", "verification_and_evidence_sec")
    timings = {name: response.get("timings", {}).get(name) for name in timing_fields
               if finite(response.get("timings", {}).get(name))}
    targets = {"decode_and_cache_sec": 25, "perception_sec": 45, "vlm_sec": 150, "evidence_output_sec": 20}
    return {"pipeline_version": version, "pipeline_timings": timings,
            "stage_targets": {name: {"target_sec": limit, "actual_sec": timings.get(name),
                                      "within_target": timings[name] <= limit if name in timings else None}
                              for name, limit in targets.items()},
            "windows_completed": coverage["windows_completed"], "windows_planned": coverage["windows_planned"],
            "perception_frames_analyzed": perception.get("frames_analyzed"),
            "perception_complete": True, "ending_included": True,
            "detection_count": len(detections), "detections_source_and_box_validated": len(detections),
            "detection_jpegs_declared": detection_image_count, "declared_jpegs_total": len(images),
            "declared_jpegs_validated": checked,
            "evidence_authentication_checked": auth_checked, "review_required": review_required,
            "review_queue_count": len(details.get("review_queue", [])),
            "case_decision_status": decision.get("status") if decision.get("status") in {
                "review_required", "confirmed_violation", "supported_violation", "no_supported_violation", "violation", "no_violation"} else "other"}


def summary(records: list[dict]) -> dict:
    done = [row for row in records if row.get("status") == "done"]
    failed = sum(row.get("status") == "failed" for row in records)
    checked = sum(row.get("checks_passed") is True for row in records)
    review_known = [row for row in done if type(row.get("review_required")) is bool]
    labels = [row for row in done if "verdict" in row and row.get("checks_passed") is True]
    tp = sum(row["reference_binary"] and row["verdict"] == "violation" for row in labels)
    fp = sum(not row["reference_binary"] and row["verdict"] == "violation" for row in labels)
    tn = sum(not row["reference_binary"] and row["verdict"] == "no_violation" for row in labels)
    fn = sum(row["reference_binary"] and row["verdict"] == "no_violation" for row in labels)
    def ratio(a, b):
        return a / b if b else None
    elapsed = sum(row.get("upload_to_result_sec", 0) for row in records)
    media_sec = sum(row["duration_sec"] for row in done)
    negatives = [row for row in labels if row["reference_binary"] is False]
    negative_hours = sum(row["duration_sec"] for row in negatives) / 3600
    return {"samples_attempted": len(records), "completed": len(done), "failed": failed,
            "technical_checks_passed_samples": checked, "failure_rate": ratio(failed, len(records)),
            "review_rate_completed_known": ratio(sum(row["review_required"] for row in review_known), len(review_known)),
            "review_rate_denominator": len(review_known),
            "media_seconds_per_processing_second": ratio(media_sec, elapsed),
            "completed_jobs_per_processing_hour": ratio(len(done) * 3600, elapsed),
            "all_under_300_sec": bool(records) and all(row.get("under_300_sec") is True for row in records),
            "exploratory_binary": {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
                "denominator": len(labels), "negative_video_hours": negative_hours,
                "emitted_findings_per_negative_hour": ratio(sum(row.get("finding_count", 0) for row in negatives), negative_hours),
                "caveat": "Weak labels from previously inspected footage; exploratory only. "
                          "Finding rate on negative-labeled footage is not human-verified false-alert rate."}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--selection", type=Path, default=ROOT / "data/labels/validation-selection.json")
    parser.add_argument("--sample", choices=("shortest", "longest", "balanced"), default="shortest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, default=ROOT / "data/api-key.txt")
    parser.add_argument("--pipeline-version", default="5.5")
    parser.add_argument("--max-detection-images", type=int, default=3, help="Validate N unique declared JPEGs per sample; 0 checks all")
    args = parser.parse_args(argv)
    records, checks = [], {}
    report = {"purpose": "Real local hybrid operation and exploratory weak-label comparison", "sample_mode": args.sample,
              "pipeline_version_requested": args.pipeline_version,
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "samples": records, "checks": checks,
              "caveat": "Previously inspected footage; no held-out generalization, annotation quality or score calibration claim."}

    def save():
        report["summary"] = summary(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    try:
        require(args.max_detection_images >= 0, "invalid_image_limit")
        url = loopback_url(args.url)
        selected = select_samples(json.loads(args.selection.read_text(encoding="utf-8")), args.sample)
        key = args.key_file.read_text(encoding="utf-8").strip()
        require(bool(key), "missing_api_key")
        headers = {"X-API-Key": key}
        with httpx.Client(base_url=url, timeout=30, trust_env=False, follow_redirects=False) as client:
            readiness_deadline = time.monotonic() + 180
            while True:
                try:
                    response = client.get("/api/v1/health", timeout=5)
                except httpx.HTTPError:
                    response = None
                if response is not None and response.status_code == 200 and response.json().get("worker_ready"):
                    require(response.json().get("busy") is False, "worker_already_busy")
                    checks["health_ready"] = True
                    break
                require(time.monotonic() < readiness_deadline, "worker_readiness_timeout")
                time.sleep(1)
            require(client.get("/api/ui/rules").status_code == 401, "authentication_check_failed")
            checks["unauthenticated_401"] = True
            invalid = client.post("/api/v1/analyze", headers=headers,
                                  files={"file": ("input.mov", b"invalid synthetic media", "video/quicktime")})
            require(invalid.status_code == 400, "corrupt_input_check_failed")
            checks["corrupt_input_400"] = True
            for row, info in selected:
                digest = hashlib.sha256()
                with info.path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                require(digest.hexdigest() == row["sha256"], "source_hash_mismatch")
                record = {"sample_number": len(records) + 1, "duration_sec": round(info.duration, 6),
                          "reference_binary": row["binary"], "checks_passed": False}
                sampler = GpuSampler()
                sampler.start()
                began = time.monotonic()
                try:
                    with info.path.open("rb") as source:
                        response = client.post("/api/v1/analyze", headers=headers,
                                               files={"file": ("input.mov", source, "video/quicktime")})
                    require(response.status_code == 202 and set(response.json()) == {"job_id"}, "admission_contract_failed")
                    job_id = response.json()["job_id"]
                    busy = client.post("/api/v1/analyze", headers=headers,
                                       files={"file": ("input.mov", b"synthetic busy probe", "video/quicktime")})
                    require(busy.status_code == 429, "busy_check_failed")
                    checks["busy_429"] = True
                    deadline = began + 320
                    while True:
                        remaining = deadline - time.monotonic()
                        require(remaining > 0, "poll_deadline_exceeded")
                        polled = client.get(f"/api/v1/jobs/{job_id}", headers=headers, timeout=min(30, remaining))
                        require(polled.status_code == 200, "poll_http_failed")
                        result = polled.json()
                        if result.get("status") != "processing":
                            break
                        require(set(result) == {"job_id", "status"}, "processing_contract_failed")
                        time.sleep(min(1, max(0, deadline - time.monotonic())))
                    elapsed = time.monotonic() - began
                    record.update(status=result["status"], upload_to_result_sec=round(elapsed, 3), under_300_sec=elapsed <= 300)
                    details_response = client.get(f"/api/ui/jobs/{job_id}", headers=headers)
                    require(details_response.status_code == 200, "details_http_failed")
                    details = details_response.json()
                    if result["status"] == "failed":
                        require(set(result) == {"job_id", "status"}, "failed_contract_failed")
                        error = (details.get("error") or {}).get("code", "unknown")
                        record["error_code"] = error if isinstance(error, str) and re.fullmatch(r"[a-z0-9_]{1,80}", error) else "unknown"
                    else:
                        record["evidence_frames_validated"] = validate_result(result, info)
                        require(all(source_time(finding["timecode_sec"], info) for finding in result["violations"]),
                                "finding_timestamp_not_source_pts")
                        record.update(verdict=result["verdict"], finding_count=len(result["violations"]))
                        record.update(validate_details(client, headers, job_id, details, info, args.pipeline_version,
                                                       args.max_detection_images))
                    record["checks_passed"] = True
                except (CheckError, AssertionError, KeyError, TypeError, ValueError, httpx.HTTPError):
                    record.setdefault("status", "benchmark_error")
                    record["benchmark_check_failed"] = True
                    record.setdefault("upload_to_result_sec", round(time.monotonic() - began, 3))
                finally:
                    record["gpu_memory"] = sampler.finish()
                    records.append(record)
                    save()
                    print(json.dumps({"sample_number": record["sample_number"], "status": record["status"],
                                      "checks_passed": record["checks_passed"],
                                      "upload_to_result_sec": record.get("upload_to_result_sec")}), flush=True)
                if record["status"] == "benchmark_error":
                    break
        report["complete"] = len(records) == len(selected)
        save()
        return 0 if report["complete"] and all(row["status"] == "done" and row["checks_passed"] for row in records) else 1
    except (CheckError, AnalysisError, OSError, ValueError, KeyError, TypeError, httpx.HTTPError):
        report["complete"] = False
        report["setup_or_service_check_failed"] = True
        save()
        print(json.dumps({"status": "benchmark_rejected", "reason": "Local setup or service contract check failed"}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
