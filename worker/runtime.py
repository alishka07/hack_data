"""Own a local model process. Never download assets or contact remote inference."""
from __future__ import annotations
import base64
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from worker.video import AnalysisError, Budget, FrameRef


RUNTIME_PROTOCOL_VERSION = "bounded-vlm-json-1"
_HASH_CACHE: dict[tuple, str] = {}


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _asset_hashes(paths: list[Path], stop_event: threading.Event | None = None) -> list[dict]:
    """Hash actual assets once per unchanged file; never trust a filename manifest."""
    result = []
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        identity = (str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        digest = _HASH_CACHE.get(identity)
        if digest is None:
            hasher = hashlib.sha256()
            with resolved.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    if stop_event is not None and stop_event.is_set():
                        raise AnalysisError("worker_stopped", "The worker stopped before model validation completed")
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            _HASH_CACHE[identity] = digest
        # Position defines artifact role. Names/paths must not create different
        # IDs when exactly the same approved model is moved to another machine.
        result.append({"bytes": stat.st_size, "sha256": digest})
    return result


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A local service may not redirect inference or health checks elsewhere."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _validate_json_schema(value: object, schema: dict) -> None:
    """Validate the bounded schema subset used by local observation/verification."""
    def invalid() -> None:
        # Never include the returned object, prompt, or a visual description.
        raise AnalysisError("model_schema_invalid", "The local model response did not satisfy the requested schema")

    kind = schema.get("type")
    valid_type = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: not isinstance(value, bool) and (
            isinstance(value, int) or isinstance(value, float) and math.isfinite(value)),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }
    if kind not in valid_type or not valid_type[kind]():
        invalid()
    if "enum" in schema and value not in schema["enum"]:
        invalid()
    if kind == "object":
        properties = schema.get("properties", {})
        if any(name not in value for name in schema.get("required", [])):
            invalid()
        extra = schema.get("additionalProperties", True)
        if extra is False and any(name not in properties for name in value):
            invalid()
        for name, item in value.items():
            if name in properties:
                _validate_json_schema(item, properties[name])
            elif isinstance(extra, dict):
                _validate_json_schema(item, extra)
    elif kind == "array":
        if len(value) > schema.get("maxItems", math.inf) or len(value) < schema.get("minItems", 0):
            invalid()
        if "items" in schema:
            for item in value:
                _validate_json_schema(item, schema["items"])
    elif kind == "string":
        if len(value) > schema.get("maxLength", math.inf) or len(value) < schema.get("minLength", 0):
            invalid()
    elif kind in {"integer", "number"}:
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            invalid()


class LocalRuntime:
    def __init__(self, root: Path, data_dir: Path, stop_event: threading.Event | None = None):
        self.root, self.data_dir = root, data_dir
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.url = os.environ.get("FORTE_LLM_URL", "http://127.0.0.1:18080").rstrip("/")
        parsed = urllib.parse.urlsplit(self.url)
        try:
            local = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            local = parsed.hostname == "localhost"
        if (parsed.scheme != "http" or not local or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("FORTE_LLM_URL must be a plain loopback HTTP address")
        self.host = parsed.hostname
        self.port = parsed.port if parsed.port is not None else 80
        if not 1 <= self.port <= 65535:
            raise ValueError("FORTE_LLM_URL must contain a valid TCP port")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        self.process: subprocess.Popen | None = None
        self.log = None
        self.external = os.environ.get("FORTE_ATTACH_MODEL", "0") == "1"
        self.backend = os.environ.get("FORTE_VLM_BACKEND", "llama.cpp")
        if self.backend not in {"llama.cpp", "transformers"}:
            raise ValueError("FORTE_VLM_BACKEND must be llama.cpp or transformers")
        self.context_tokens = _bounded_int("FORTE_VLM_CONTEXT", 8192, 2048, 32768)
        self.max_frames = _bounded_int("FORTE_VLM_MAX_FRAMES", 8, 1, 8)
        self.max_pixels = _bounded_int("FORTE_VLM_MAX_PIXELS", 512 * 28 * 28, 28 * 28, 2048 * 28 * 28)
        self.min_pixels = _bounded_int("FORTE_VLM_MIN_PIXELS", min(64 * 28 * 28, self.max_pixels), 28 * 28, self.max_pixels)
        self.max_output_tokens = _bounded_int("FORTE_VLM_MAX_OUTPUT_TOKENS", 2048, 128, 4096)
        if self.max_output_tokens >= self.context_tokens:
            raise ValueError("FORTE_VLM_MAX_OUTPUT_TOKENS must be smaller than FORTE_VLM_CONTEXT")
        self.server_metadata: dict = {}
        self.asset_identity: list[dict] = []
        self.request_metrics = {"calls": 0, "attempts": 0, "retries": 0,
                                "failures": 0, "elapsed_sec": 0.0, "last_request": None}

    def metadata(self) -> dict:
        """Report configuration separately from a server's measured readiness/provenance."""
        assets = self.root / "tmp" / "planning" / "local_runtime"
        default_model = assets / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
        model = Path(os.environ.get("FORTE_TRANSFORMERS_MODEL_DIR", "models/qwen2.5-vl-7b")) if self.backend == "transformers" else Path(os.environ.get("FORTE_MODEL_PATH", default_model))
        configuration = {"backend": self.backend, "context_tokens": self.context_tokens,
                         "max_frames": self.max_frames, "min_pixels": self.min_pixels,
                         "max_pixels": self.max_pixels, "max_output_tokens": self.max_output_tokens,
                         "gpu_layers": os.getenv("FORTE_GPU_LAYERS", "99"),
                         "projector_cpu": os.getenv("FORTE_VLM_CPU_PROJECTOR", "0") == "1",
                         "protocol": RUNTIME_PROTOCOL_VERSION}
        server_id = self.server_metadata.get("model_id")
        if self.backend == "llama.cpp" and not self.external and self.asset_identity:
            model_id = "sha256:" + _fingerprint({"configuration": configuration, "assets": self.asset_identity})
        elif self.backend == "transformers" and isinstance(server_id, str) and server_id.startswith("sha256:"):
            model_id = "sha256:" + _fingerprint({"configuration": configuration, "server_model_id": server_id})
        else:
            model_id = "unattested:" + _fingerprint({"configuration": configuration, "configured_model": str(model)})
        return {"backend": self.backend, "model": model.name, "model_id": model_id,
                "identity_source": ("unattested configuration" if model_id.startswith("unattested:") else
                                    "server-reported hashed assets" if self.backend == "transformers" else "hashed managed assets"),
                "asset_hashes": self.asset_identity.copy(),
                "attached": self.external, "context_tokens": self.context_tokens,
                "max_frames": self.max_frames, "max_pixels_per_frame": self.max_pixels,
                "min_pixels_per_frame": self.min_pixels,
                "max_output_tokens": self.max_output_tokens,
                "gpu_layers": configuration["gpu_layers"], "projector_cpu": configuration["projector_cpu"],
                "structured_output": "JSON schema constrained decoding plus host validation",
                "strict_schema_decoding": self.server_metadata.get("strict_schema_decoding") if self.external else True,
                "local_only": True, "server": self.server_metadata.copy(),
                "requests": {**self.request_metrics,
                             "last_request": dict(self.request_metrics["last_request"])
                             if self.request_metrics["last_request"] is not None else None}}

    def _read_server_metadata(self) -> None:
        """Our optional server reports identity; llama.cpp need not expose this route."""
        if self.backend != "transformers":
            return
        try:
            with self.opener.open(self.url + "/v1/runtime", timeout=2) as response:
                value = json.load(response)
            if isinstance(value, dict):
                self.server_metadata = value
        except (OSError, ValueError):
            # Health alone is sufficient for explicitly attached compatible servers.
            self.server_metadata = {"metadata_available": False}

    def healthy(self) -> bool:
        try:
            with self.opener.open(self.url + "/health", timeout=2) as response:
                payload = json.load(response)
                return isinstance(payload, dict) and payload.get("status") == "ok"
        except (OSError, ValueError):
            return False

    def start(self) -> None:
        self._check_cancelled()
        if self.external:
            if not self.healthy():
                raise AnalysisError("model_unavailable", "The configured local model server is not ready")
            self._read_server_metadata()
            return
        if self.healthy():
            raise AnalysisError("model_port_busy", "The local model port is occupied; use FORTE_ATTACH_MODEL=1 only for a trusted local server")
        assets = self.root / "tmp" / "planning" / "local_runtime"
        executable = Path(os.environ.get("FORTE_LLAMA_SERVER", assets / "bin" / "llama-server.exe"))
        model = Path(os.environ.get("FORTE_MODEL_PATH", assets / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"))
        projector = Path(os.environ.get("FORTE_MMPROJ_PATH", assets / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"))
        if self.backend == "llama.cpp" and not all(p.is_file() for p in (executable, model, projector)):
            raise AnalysisError("model_assets_missing", "Local model weights, vision projector, or llama-server are missing; see README")
        if self.backend == "transformers":
            model_dir = Path(os.environ.get("FORTE_TRANSFORMERS_MODEL_DIR", self.root / "models" / "qwen2.5-vl-7b"))
            from worker.transformers_runtime import validate_model_directory
            validate_model_directory(model_dir)
        else:
            self.asset_identity = _asset_hashes([model, projector, executable], self.stop_event)
        directory = self.data_dir / "runtime"
        directory.mkdir(parents=True, exist_ok=True)
        self.log = (directory / "model.log").open("ab")
        command = [str(executable), "-m", str(model), "--mmproj", str(projector), "--ctx-size", str(self.context_tokens),
                   "--parallel", "1", "--n-gpu-layers", os.environ.get("FORTE_GPU_LAYERS", "99"),
                   "--threads", "8", "--host", self.host, "--port", str(self.port), "--flash-attn", "on"]
        if os.getenv("FORTE_VLM_CPU_PROJECTOR", "0") == "1":
            command.append("--no-mmproj-offload")
        if self.backend == "transformers":
            python = os.environ.get("FORTE_VLM_PYTHON", sys.executable)
            command = [python, str(self.root / "scripts" / "serve_vlm.py"), "--model-dir", str(model_dir),
                       "--host", self.host, "--port", str(self.port)]
        self._check_cancelled()
        self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        startup_timeout = _bounded_int("FORTE_VLM_STARTUP_TIMEOUT", 300 if self.backend == "transformers" else 120, 30, 900)
        until = time.monotonic() + startup_timeout
        while time.monotonic() < until:
            self._check_cancelled()
            if self.process.poll() is not None:
                raise AnalysisError("model_start_failed", "The local model runtime stopped during startup; inspect its local runtime log")
            if self.healthy():
                self._read_server_metadata()
                return
            self.stop_event.wait(0.5)
        raise AnalysisError("model_start_timeout", "The local model did not become ready within its startup timeout")

    def _check_cancelled(self) -> None:
        if self.stop_event.is_set():
            raise AnalysisError("worker_stopped", "The worker stopped before analysis completed")

    def cancel(self) -> None:
        """Nonblocking cancellation, safe from a signal handler or heartbeat thread."""
        self.stop_event.set()
        process = self.process
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
            except OSError:
                pass

    def stop(self) -> None:
        process = self.process
        try:
            if process and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    if process.poll() is None:
                        raise
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            self.process = None
            if self.log:
                self.log.close()
                self.log = None

    def ask(self, prompt: str, frames: list[FrameRef], schema: dict, budget: Budget,
            max_tokens: int = 512) -> dict:
        self._check_cancelled()
        if (len(frames) > self.max_frames or not 1 <= max_tokens <= self.max_output_tokens
                or len(prompt) > 65536):
            raise AnalysisError("model_input_budget", "The local model request exceeds its frame, prompt, or output budget")
        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, frame in enumerate(frames):
            view = f" View: {frame.view}."
            if frame.crop_bbox is not None:
                view += f" Crop of original normalized xyxy={list(frame.crop_bbox)}; outside this crop is unknown."
            content.extend([{"type": "text", "text": f"Frame {i}, source index {frame.index}, elapsed source time {frame.timestamp:.3f} seconds.{view}"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(self._inference_image(frame.path)).decode("ascii")}}])
        payload = {"model": "local-vlm", "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": max_tokens, "stream": False,
                   "response_format": {"type": "json_schema", "json_schema": {"name": "video_analysis", "strict": True, "schema": schema}}}
        request = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.monotonic()
        metrics = {"attempts": 0, "frame_count": len(frames), "max_tokens": max_tokens,
                   "timeout_sec": 0.0, "elapsed_sec": 0.0, "error_code": None,
                   "retry_error_code": None}
        self.request_metrics["calls"] += 1
        try:
            for attempt in range(2):
                self._check_cancelled()
                remaining = budget.remaining(5)
                timeout = min(45, remaining)
                deadline_limited = remaining < 45
                metrics["attempts"] += 1
                metrics["timeout_sec"] = round(timeout, 3)
                self.request_metrics["attempts"] += 1
                retryable = False
                try:
                    with self.opener.open(request, timeout=timeout) as response:
                        answer = json.load(response)
                    self._check_cancelled()
                    choice = answer["choices"][0]
                    if choice.get("finish_reason") == "length":
                        raise AnalysisError("model_output_truncated", "The model could not finish a structured response")
                    result = json.loads(choice["message"]["content"])
                    _validate_json_schema(result, schema)
                    return result
                except urllib.error.HTTPError as exc:
                    retryable = exc.code in {502, 503}
                    error = AnalysisError("model_http_error", f"The local model server returned HTTP {exc.code}")
                    exc.close()
                except (TimeoutError, urllib.error.URLError) as exc:
                    timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
                    if timed_out:
                        code = "deadline_exceeded" if deadline_limited else "model_timeout"
                        message = ("The video could not be fully processed before its deadline"
                                   if deadline_limited else "The local model request exceeded its 45 second timeout")
                        error = AnalysisError(code, message)
                    else:
                        error = AnalysisError("model_transport_error", "The local model server connection failed")
                except AnalysisError as exc:
                    if exc.code != "model_schema_invalid":
                        raise
                    retryable = True
                    error = exc
                except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                    retryable = True
                    error = AnalysisError("model_output_invalid", "The local model returned invalid structured output")
                except OSError as exc:
                    error = AnalysisError("model_transport_error", "The local model server connection failed")
                self._check_cancelled()
                # A retry gets a full request allowance plus the finalization
                # reserve; it can never consume the job's last seconds.
                if attempt == 0 and retryable and budget.deadline - time.time() >= 50:
                    self.request_metrics["retries"] += 1
                    metrics["retry_error_code"] = error.code
                    continue
                raise error from None
        except AnalysisError as exc:
            metrics["error_code"] = exc.code
            self.request_metrics["failures"] += 1
            raise
        finally:
            # Counts/timings only: never store prompts, frames, replies or HTTP bodies.
            metrics["elapsed_sec"] = round(time.monotonic() - started, 3)
            self.request_metrics["elapsed_sec"] = round(self.request_metrics["elapsed_sec"] + metrics["elapsed_sec"], 3)
            self.request_metrics["last_request"] = metrics

    def _inference_image(self, path: Path) -> bytes:
        """Bound inference pixels while retaining the original evidence JPEG unchanged."""
        import cv2
        import numpy as np
        if path.stat().st_size > 16 * 1024 * 1024:
            raise AnalysisError("model_input_budget", "An inference image exceeds the byte budget")
        raw = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise AnalysisError("model_image_invalid", "An inference frame is not a decodable image")
        height, width = decoded.shape[:2]
        if height * width <= self.max_pixels:
            return raw
        scale = (self.max_pixels / (height * width)) ** 0.5
        resized = cv2.resize(decoded, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise AnalysisError("model_image_invalid", "An inference frame could not be resized")
        return encoded.tobytes()

    def map_scene(self, frames: list[FrameRef], budget: Budget) -> list[dict]:
        """One focused geometry task, separate from actions, scores and legality."""
        from worker.scene import scene_schema
        schema = {"type": "object", "properties": {"scene_regions": scene_schema(len(frames))},
                  "required": ["scene_regions"], "additionalProperties": False}
        prompt = (
            "Locate clearly visible physical regions in these ordered images of a workstation. "
            "Return scene_regions for workstation (the worker's operating area), customer (visitor side), "
            "counter (surface where objects are handled), storage (clearly visible storage), window (service opening), "
            "safe (visible safe), and cash_zone (visible cash-handling surface). "
            "Use zone, status supported/unknown, bbox=[left,top,right,bottom] normalized0..1, "
            "and supporting frame_indices from this request. Visible surfaces may be mapped even when "
            "no activity is happening. Regions may overlap. Do not identify people, infer roles from faces, "
            "judge actions or legality, or guess off-camera geometry. Unsupported zones return status unknown, "
            "bbox [], frame_indices []. Geometry is only a crop proposal, not proof of legal room boundaries."
        )
        return self.ask(prompt, frames, schema, budget, 1024)["scene_regions"]
