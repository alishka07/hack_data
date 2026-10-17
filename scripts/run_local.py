"""Run the local API and its single model-owning worker in one terminal."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _use_project_python() -> None:
    """Prefer the project environment when invoked with a system interpreter."""
    executable = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if executable.is_file() and Path(sys.executable).resolve() != executable.resolve():
        os.execv(str(executable), [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]])


def _check_dependencies() -> None:
    required = {"fastapi": "fastapi", "uvicorn": "uvicorn", "pydantic": "pydantic",
                "multipart": "python-multipart", "cv2": "opencv-python-headless",
                "numpy": "numpy", "httpx": "httpx"}
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if sys.version_info < (3, 12) or missing:
        explanation = "Python 3.12+ is required." if sys.version_info < (3, 12) else "Missing packages: " + ", ".join(missing)
        print(explanation, file=sys.stderr)
        print("From the repository directory run:\n"
              "  py -3.12 -m venv .venv\n"
              "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.lock.txt\n"
              "  .\\.venv\\Scripts\\python.exe scripts/run_local.py", file=sys.stderr)
        raise SystemExit(2)


def _start(command: list[str], environment: dict[str, str]) -> subprocess.Popen:
    options: dict = {"cwd": str(ROOT), "env": environment}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(command, **options)


def _stop(children: list[subprocess.Popen]) -> None:
    """Signal only our child groups; force-stop surviving owned trees as a last resort."""
    for child in reversed(children):
        if child.poll() is not None:
            continue
        try:
            if os.name == "nt":
                child.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(child.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    deadline = time.monotonic() + 12
    for child in reversed(children):
        if child.poll() is not None:
            continue
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                # The PID is the Popen child we still own; never search/kill by process name.
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass


def main() -> int:
    _use_project_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("FORTE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FORTE_PORT", "8000")))
    parser.add_argument("--show-key", action="store_true", help="Explicitly display the local API key in this terminal")
    arguments = parser.parse_args()
    if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("This local launcher accepts loopback hosts only; public deployment is a separate step.")
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    _check_dependencies()
    os.environ["FORTE_ROOT"] = str(ROOT)
    os.environ["FORTE_HOST"] = arguments.host
    os.environ["FORTE_PORT"] = str(arguments.port)
    sys.path.insert(0, str(ROOT))
    from app.settings import load_settings
    settings = load_settings()
    display_host = "[::1]" if arguments.host == "::1" else arguments.host
    print(f"Local review: http://{display_host}:{arguments.port}", flush=True)
    if os.environ.get("FORTE_API_KEY", "").strip():
        print("API key: supplied by FORTE_API_KEY", flush=True)
    else:
        print(f"API key file: {settings.data_dir / 'api-key.txt'}", flush=True)
    if arguments.show_key:
        print(f"API key: {settings.api_key}", flush=True)
    print("The worker loads the model before accepting analysis. Press Ctrl+C to stop.", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    children: list[subprocess.Popen] = []
    interrupted = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, request_stop)
    try:
        children.append(_start([sys.executable, "-m", "uvicorn", "app.main:app", "--host", arguments.host,
                                "--port", str(arguments.port), "--no-access-log"], environment))
        children.append(_start([sys.executable, "-m", "worker.main"], environment))
        while not interrupted:
            for name, child in zip(("API", "Worker"), children):
                code = child.poll()
                if code is not None:
                    print(f"{name} exited with code {code}; stopping the local service.", file=sys.stderr)
                    return code if code != 0 else 1
            time.sleep(0.25)
        return 0
    finally:
        _stop(children)


if __name__ == "__main__":
    raise SystemExit(main())
