"""Run an isolated, temporary public jury demo using an already-running local model.

The public URL changes after a restart. Keep this supervisor and the existing
model owner running. Only one app should analyze videos at a time: the local
app and jury app share one model server, but have separate admission queues.
"""
from __future__ import annotations

import argparse
import ctypes
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_local import _check_dependencies, _stop


PUBLIC_URL = re.compile(r"https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)*\.trycloudflare\.com(?![a-zA-Z0-9.-])")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _local_model_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    try:
        local = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        local = parsed.hostname == "localhost"
    if (parsed.scheme != "http" or not local or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("--model-url must be a plain loopback HTTP address")
    port = parsed.port if parsed.port is not None else 80
    if not 1 <= port <= 65535:
        raise ValueError("The model URL must contain a valid port")
    return value.rstrip("/")


def _read_health(url: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(url, timeout=2) as response:
            payload = json.loads(response.read(16384))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _check_api_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"Local API port {port} is already bound; no existing service was changed") from exc


def _start(command: list[str], environment: dict[str, str], output) -> subprocess.Popen:
    options = {"cwd": str(ROOT), "env": environment, "stdin": subprocess.DEVNULL,
               "stdout": output, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    return subprocess.Popen(command, **options)


def _awake(enabled: bool) -> None:
    """Prevent automatic system sleep, without keeping the screen awake."""
    if os.name != "nt":
        return
    function = ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]
    function.restype = ctypes.c_uint
    if not function(0x80000001 if enabled else 0x80000000) and enabled:
        print("Could not prevent automatic sleep; keep the laptop awake manually.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "jury",
                        help="Separate jury storage; reused when restarting this demo")
    parser.add_argument("--port", type=int, default=8001, help="Loopback API port (default: 8001)")
    parser.add_argument("--model-url", default="http://127.0.0.1:18080",
                        help="Existing trusted local model; this script does not own or stop it")
    parser.add_argument("--cloudflared", type=Path, required=True, help="Path to an installed cloudflared executable")
    arguments = parser.parse_args()
    try:
        _check_dependencies()
        model_url = _local_model_url(arguments.model_url)
        data_dir = arguments.data_dir.resolve()
        cloudflared = arguments.cloudflared.resolve()
        if data_dir == (ROOT / "data").resolve():
            raise ValueError("Use a separate jury data directory, not the local app's data directory")
        if not cloudflared.is_file():
            raise ValueError("--cloudflared must point to an installed executable")
        _check_api_port(arguments.port)
        if _read_health(model_url + "/health").get("status") != "ok":
            raise ValueError("The attached model is not healthy; start the existing local app first")
    except (OSError, ValueError) as exc:
        print(f"Jury demo did not start: {exc}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    for name in ("FORTE_API_KEY", "FORTE_CALIBRATION_PATH"):
        environment.pop(name, None)
    environment.update(FORTE_ROOT=str(ROOT), FORTE_DATA_DIR=str(data_dir),
                       FORTE_DB_PATH=str(data_dir / "jobs.sqlite"), FORTE_HOST="127.0.0.1",
                       FORTE_PORT=str(arguments.port), FORTE_ATTACH_MODEL="1", FORTE_LLM_URL=model_url,
                       FORTE_VLM_BACKEND="llama.cpp", FORTE_VLM_MAX_FRAMES="8",
                       FORTE_MAX_UPLOAD_BYTES="95000000", PYTHONUNBUFFERED="1")
    local_url = f"http://127.0.0.1:{arguments.port}"
    children: list[subprocess.Popen] = []
    names: list[str] = []
    logs = []
    interrupted = False
    public_url_path = data_dir / "public-url.txt"
    tunnel_log = data_dir / "cloudflared.log"

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)

    def start(name: str, command: list[str], path: Path) -> None:
        output = path.open("wb")
        logs.append(output)
        children.append(_start(command, environment, output))
        names.append(name)

    def check_children() -> None:
        for name, child in zip(names, children):
            code = child.poll()
            if code is not None:
                raise RuntimeError(f"{name} exited with code {code}; inspect logs in {data_dir}")

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        public_url_path.unlink(missing_ok=True)
        _awake(True)
        print(f"Jury storage: {data_dir}", flush=True)
        print(f"API key file: {data_dir / 'api-key.txt'}", flush=True)
        print("Temporary demo: keep this process and the local model owner running.", flush=True)
        print("Only one app should analyze videos at a time; leave local analysis idle during the jury demo.", flush=True)
        start("API", [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                      "--port", str(arguments.port), "--no-access-log"], data_dir / "api.log")
        start("Worker", [sys.executable, "-m", "worker.main"], data_dir / "worker.log")
        until = time.monotonic() + 60
        while not interrupted:
            check_children()
            if _read_health(local_url + "/api/v1/health").get("worker_ready") is True:
                break
            if time.monotonic() >= until:
                raise RuntimeError(f"Jury worker did not become ready; inspect logs in {data_dir}")
            time.sleep(.25)
        if interrupted:
            return 0
        print(f"Local jury interface ready: {local_url}", flush=True)
        start("Tunnel", [str(cloudflared), "--no-autoupdate", "tunnel", "--protocol", "http2",
                         "--url", local_url], tunnel_log)
        public_url = None
        until = time.monotonic() + 120
        while not interrupted:
            check_children()
            if public_url is None:
                # This log was truncated for this run; never publish a stale URL.
                match = PUBLIC_URL.search(tunnel_log.read_text(encoding="utf-8", errors="replace"))
                if match:
                    public_url = match.group(0)
                    public_url_path.write_text(public_url + "\n", encoding="utf-8")
                    print(f"Public website: {public_url}", flush=True)
                    print(f"API base URL: {public_url}/api/v1", flush=True)
                    print("The URL is temporary and stops working when this supervisor stops. Press Ctrl+C to stop.", flush=True)
                elif time.monotonic() >= until:
                    raise RuntimeError(f"Tunnel did not provide a public URL; inspect {tunnel_log}")
            time.sleep(.25)
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"Jury demo stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            _stop(children)
        finally:
            for output in logs:
                output.close()
            _awake(False)


if __name__ == "__main__":
    raise SystemExit(main())
