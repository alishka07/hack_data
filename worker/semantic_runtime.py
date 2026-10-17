"""Own a local llama.cpp process. Never download assets or contact remote inference."""
from __future__ import annotations
import base64
import ipaddress
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from worker.video import AnalysisError, Budget, FrameRef


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
            return
        if self.healthy():
            raise AnalysisError("model_port_busy", "The local model port is occupied; use FORTE_ATTACH_MODEL=1 only for a trusted local server")
        assets = self.root / "tmp" / "planning" / "local_runtime"
        executable = Path(os.environ.get("FORTE_LLAMA_SERVER", assets / "bin" / "llama-server.exe"))
        model = Path(os.environ.get("FORTE_MODEL_PATH", assets / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"))
        projector = Path(os.environ.get("FORTE_MMPROJ_PATH", assets / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"))
        if not all(p.is_file() for p in (executable, model, projector)):
            raise AnalysisError("model_assets_missing", "Local model weights, vision projector, or llama-server are missing; see README")
        directory = self.data_dir / "runtime"
        directory.mkdir(parents=True, exist_ok=True)
        self.log = (directory / "model.log").open("ab")
        command = [str(executable), "-m", str(model), "--mmproj", str(projector), "--ctx-size", "8192",
                   "--parallel", "1", "--n-gpu-layers", os.environ.get("FORTE_GPU_LAYERS", "99"),
                   "--threads", "8", "--host", self.host, "--port", str(self.port), "--flash-attn", "on"]
        self._check_cancelled()
        self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        until = time.monotonic() + 120
        while time.monotonic() < until:
            self._check_cancelled()
            if self.process.poll() is not None:
                raise AnalysisError("model_start_failed", "The local model runtime stopped during startup; inspect its local runtime log")
            if self.healthy():
                return
            self.stop_event.wait(0.5)
        raise AnalysisError("model_start_timeout", "The local model did not become ready within 120 seconds")

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
        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, frame in enumerate(frames):
            content.extend([{"type": "text", "text": f"Frame {i}, elapsed source time {frame.timestamp:.3f} seconds."},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(frame.path.read_bytes()).decode("ascii")}}])
        payload = {"model": "local-qwen3vl", "messages": [{"role": "user", "content": content}],
                   "temperature": 0, "max_tokens": max_tokens, "stream": False,
                   "response_format": {"type": "json_schema", "json_schema": {"name": "video_analysis", "strict": True, "schema": schema}}}
        request = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=min(45, budget.remaining(5))) as response:
                answer = json.load(response)
            self._check_cancelled()
            choice = answer["choices"][0]
            if choice.get("finish_reason") == "length":
                raise AnalysisError("model_output_truncated", "The model could not finish a structured response")
            result = json.loads(choice["message"]["content"])
            _validate_json_schema(result, schema)
            return result
        except AnalysisError:
            raise
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            self._check_cancelled()
            # Deliberately do not expose model replies, prompts, image data, or HTTP bodies.
            raise AnalysisError("model_request_failed", "The local model request failed or returned invalid structured output") from exc
