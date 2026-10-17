"""Authenticated local HTTP interface and exact jury response contract."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.db import BusyError, JobStore
from app.settings import Settings, load_settings
from worker.catalog import load_catalog


class BodyLimitMiddleware:
    """Reject oversized declared bodies and bound multipart streams without buffering them."""
    def __init__(self, app, max_body_bytes: int, api_key: str):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.api_key = api_key.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        path = scope.get("path", "")
        protected = path.startswith("/api/ui/") or path.startswith("/api/v1/jobs/") or path == "/api/v1/analyze"
        if protected and not hmac.compare_digest(headers.get(b"x-api-key", b""), self.api_key):
            return await JSONResponse({"detail": "A valid X-API-Key header is required"}, status_code=401)(scope, receive, send)
        if scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        if length > self.max_body_bytes:
            return await JSONResponse({"detail": "Upload exceeds the configured size limit"}, status_code=413)(scope, receive, send)
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_body_bytes:
                    raise HTTPException(413, "Upload exceeds the configured size limit")
            return message

        await self.app(scope, bounded_receive, send)


def probe_video(path: Path, ffprobe_path: str = "ffprobe") -> dict[str, Any]:
    """Inspect media locally; decoding/inference validation continues in the worker."""
    executable = shutil.which(ffprobe_path)
    if executable is None:
        raise HTTPException(503, "Local video inspection is unavailable")
    try:
        completed = subprocess.run(
            [executable, "-v", "error", "-show_entries", "format=format_name,duration:stream=codec_type,width,height",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=15, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        raise HTTPException(400, "The video could not be inspected") from None
    if completed.returncode:
        raise HTTPException(400, "The upload is not a readable MOV video")
    try:
        metadata = json.loads(completed.stdout)
        format_info = metadata["format"]
        duration = float(format_info["duration"])
        video = next(stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video")
        valid = ("mov" in format_info["format_name"].split(",") and math.isfinite(duration) and duration > 0
                 and int(video.get("width", 0)) > 0 and int(video.get("height", 0)) > 0)
    except (KeyError, ValueError, TypeError, StopIteration, json.JSONDecodeError):
        valid = False
    if not valid:
        raise HTTPException(400, "The upload must contain a readable MOV video stream")
    return metadata


def _json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _valid_job_id(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _declared_evidence(details: Any) -> set[str]:
    """Only explicit evidence records can authorize a file download."""
    names: set[str] = set()
    if not isinstance(details, dict):
        return names
    entries = list(details.get("evidence", [])) if isinstance(details.get("evidence"), list) else []
    for finding in details.get("findings", []):
        if isinstance(finding, dict):
            frames = finding.get("evidence_frames", finding.get("evidence", []))
            if isinstance(frames, list):
                entries.extend(frames)
    for item in entries:
        if isinstance(item, dict):
            value = item.get("filename")
            if isinstance(value, str) and Path(value).name == value:
                names.add(value)
    return names


def create_app(settings: Settings | None = None, store: JobStore | None = None) -> FastAPI:
    settings = settings or load_settings()
    if not settings.api_key:
        raise ValueError("Configure a nonempty API key")
    store = store or JobStore(settings.db_path, worker_stale_sec=settings.worker_stale_sec)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    uploads = settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    upload_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store.init()
        store.recover_stale_jobs(settings.worker_stale_sec)
        yield

    application = FastAPI(title="ForteBank local video review", version="0.1.0", lifespan=lifespan,
                          docs_url=None, redoc_url=None)
    application.state.settings = settings
    application.state.store = store
    application.add_middleware(BodyLimitMiddleware, max_body_bytes=settings.max_upload_bytes + 64 * 1024,
                              api_key=settings.api_key)

    async def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "A valid X-API-Key header is required")

    auth = [Depends(require_key)]

    def worker_ready() -> bool:
        state = store.worker_state()
        return bool(state and state["ready"] and time.time() - state["heartbeat_at"] <= settings.worker_stale_sec)

    def recover() -> None:
        store.recover_stale_jobs(settings.worker_stale_sec)

    def lookup(job_id: str) -> dict:
        if not _valid_job_id(job_id):
            raise HTTPException(404, "Job not found")
        recover()
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @application.get("/api/v1/health")
    def health():
        recover()
        ready = worker_ready()
        busy = bool(store.active_job())
        return {"status": "busy" if ready and busy else "ready" if ready else "starting",
                "worker_ready": ready, "busy": busy, "version": "0.1.0"}

    @application.post("/api/v1/analyze", status_code=202, dependencies=auth)
    async def analyze(file: UploadFile = File(...)):
        await run_in_threadpool(recover)
        if not await run_in_threadpool(worker_ready):
            raise HTTPException(503, "The local inference worker is not ready", headers={"Retry-After": "5"})
        if upload_lock.locked():
            raise HTTPException(429, "The local worker is busy", headers={"Retry-After": "5"})
        original_name = (file.filename or "").replace("\\", "/").split("/")[-1]
        if Path(original_name).suffix.lower() != ".mov":
            raise HTTPException(415, "Upload a MOV file")
        async with upload_lock:
            if await run_in_threadpool(store.active_job):
                raise HTTPException(429, "The local worker is busy", headers={"Retry-After": "5"})
            # UUID generation is internal until all admission checks succeed.
            job_id = str(uuid4())
            path = uploads / f"{job_id}.mov"
            accepted = False
            try:
                size = 0
                with path.open("xb") as output:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > settings.max_upload_bytes:
                            raise HTTPException(413, "Upload exceeds the configured size limit")
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if size == 0:
                    raise HTTPException(400, "The upload is empty")
                await run_in_threadpool(probe_video, path, settings.ffprobe_path)
                if not await run_in_threadpool(worker_ready):
                    raise HTTPException(503, "The local inference worker is not ready", headers={"Retry-After": "5"})
                def commit_admission():
                    nonlocal accepted
                    store.create_job(job_id, path, original_name, settings.job_timeout_sec)
                    # Set in the committing thread so cancellation cannot delete an
                    # accepted source between the transaction and the await return.
                    accepted = True
                try:
                    await run_in_threadpool(commit_admission)
                except BusyError:
                    raise HTTPException(429, "The local worker is busy", headers={"Retry-After": "5"}) from None
                return {"job_id": job_id}
            finally:
                await file.close()
                if not accepted:
                    path.unlink(missing_ok=True)

    @application.get("/api/v1/jobs/{job_id}", dependencies=auth)
    def job_status(job_id: str):
        job = lookup(job_id)
        response = {"job_id": job_id, "status": job["status"]}
        if job["status"] == "done":
            result = _json(job["result_json"], {})
            response.update({"verdict": result["verdict"], "violations": result["violations"]})
        return response

    @application.get("/api/ui/active-job", dependencies=auth)
    def active_job():
        recover()
        job = store.active_job()
        payload = ({"job_id": job["id"], "status": job["status"],
                    "stage": job["internal_stage"], "progress": job["progress"]}
                   if job else {"job_id": None})
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @application.get("/api/ui/jobs/{job_id}", dependencies=auth)
    def ui_job(job_id: str):
        job = lookup(job_id)
        details = _json(job["details_json"], {})
        decision = details.get("case_decision") if isinstance(details, dict) else None
        cases = decision.get("cases") if isinstance(decision, dict) else None
        if (job["status"] == "done" and isinstance(cases, list)
                and any(isinstance(case, dict) and "probability" not in case for case in cases)):
            # Older saved reviews already retain verifier confidence on disk.
            # Add its display estimate without re-running inference or changing
            # the saved verdict, findings, case status, or review requirements.
            attempts = []
            path = settings.data_dir / "jobs" / job_id / "verification.json"
            try:
                if path.is_file() and path.stat().st_size <= 8 * 1024 * 1024:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                        attempts = value
            except (OSError, ValueError):
                pass
            from worker.decision import enrich_case_probabilities
            details = {**details, "case_decision": {**decision, "cases": enrich_case_probabilities(
                cases, attempts, details.get("findings", []), details.get("assessments", []))}}
        return {"job_id": job_id, "status": job["status"], "stage": job["internal_stage"],
                "progress": job["progress"], "message": job["message"], "original_name": job["original_name"],
                "created_at": job["created_at"], "deadline_at": job["deadline_at"],
                "result": _json(job["result_json"], None), "details": details,
                "timings": _json(job["timings_json"], {}),
                "error": {"code": job["error_code"], "message": job["error_message"]} if job["error_code"] else None}

    @application.get("/api/ui/jobs/{job_id}/video", dependencies=auth)
    def source_video(job_id: str):
        job = lookup(job_id)
        path = Path(job["video_path"]).resolve()
        if path.parent != uploads.resolve() or not path.is_file():
            raise HTTPException(404, "Video not found")
        return FileResponse(path, media_type="video/quicktime", headers={"Cache-Control": "no-store"})

    @application.get("/api/ui/jobs/{job_id}/evidence/{basename}", dependencies=auth)
    def evidence(job_id: str, basename: str):
        job = lookup(job_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:jpg|jpeg)", basename, flags=re.IGNORECASE):
            raise HTTPException(404, "Evidence not found")
        if basename not in _declared_evidence(_json(job["details_json"], {})):
            raise HTTPException(404, "Evidence not found")
        evidence_root = (settings.data_dir / "evidence").resolve()
        directory = (evidence_root / job_id).resolve()
        path = (directory / basename).resolve()
        if directory.parent != evidence_root or path.parent != directory or not path.is_file():
            raise HTTPException(404, "Evidence not found")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @application.get("/api/ui/rules", dependencies=auth)
    def rules():
        try:
            return json.loads((settings.root / "rules" / "registry.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(503, "The local rule registry is unavailable") from None

    @application.get("/api/ui/catalog", dependencies=auth)
    def catalog():
        try:
            return load_catalog(settings.root)
        except (OSError, ValueError):
            raise HTTPException(503, "The local review catalog is unavailable") from None

    web = settings.root / "web"
    if web.is_dir():
        application.mount("/static", StaticFiles(directory=web), name="static")

    @application.get("/policies/workplace-eating", include_in_schema=False)
    def workplace_eating_policy():
        path = settings.root / "rules" / "sources" / "workplace-eating.md"
        if not path.is_file():
            raise HTTPException(404, "Workplace policy not found")
        return FileResponse(path, media_type="text/plain", headers={"Cache-Control": "no-store"})

    @application.get("/style.css", include_in_schema=False)
    def stylesheet():
        path = web / "style.css"
        if not path.is_file():
            raise HTTPException(404, "Asset not found")
        return FileResponse(path, media_type="text/css")

    @application.get("/app.js", include_in_schema=False)
    def frontend_script():
        path = web / "app.js"
        if not path.is_file():
            raise HTTPException(404, "Asset not found")
        return FileResponse(path, media_type="text/javascript")

    @application.get("/", include_in_schema=False)
    def index():
        index_path = web / "index.html"
        if not index_path.is_file():
            return JSONResponse({"service": "ForteBank local video review", "health": "/api/v1/health"})
        return FileResponse(index_path, media_type="text/html", headers={"Cache-Control": "no-store"})

    @application.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return application


app = create_app()
