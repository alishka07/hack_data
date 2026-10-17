"""Prepare a fresh laptop using public assets; never start or alter a running service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def python_in(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def executable(value: str | None, default: str) -> str:
    path = Path(value).expanduser().resolve() if value else None
    found = str(path) if path and path.is_file() else shutil.which(value or default)
    if not found:
        raise ValueError(f"Missing {default}. Install it first or pass its explicit path; see docs/local-setup.md")
    return str(Path(found).resolve())


def build_plan(root: Path, *, skip_downloads: bool = False, skip_runtime: bool = False) -> list[list[str]]:
    app_python = str(python_in(root / ".venv"))
    objects_python = str(python_in(root / ".venv-perception"))
    commands = []
    if not python_in(root / ".venv").is_file():
        commands.append([sys.executable, "-m", "venv", str(root / ".venv")])
    commands += [[app_python, "-m", "pip", "install", "-r", "requirements.lock.txt"],
                 [app_python, "-m", "pip", "install", "--no-deps", "-r", "requirements.perception.txt"]]
    if not python_in(root / ".venv-perception").is_file():
        commands.append([sys.executable, "-m", "venv", str(root / ".venv-perception")])
    commands += [[objects_python, "-m", "pip", "install", "torch==2.9.0", "torchvision==0.24.0",
                  *([] if sys.platform == "darwin" else ["--index-url", "https://download.pytorch.org/whl/cpu"])],
                 [objects_python, "-m", "pip", "install", "-r", "requirements.objects.txt"]]
    if not skip_downloads:
        commands += [[app_python, "scripts/setup_perception.py"],
                     [app_python, "scripts/setup_architecture_models.py", "--pose", "--objects"],
                     [app_python, "scripts/setup_local_runtime.py", *(["--skip-runtime"] if skip_runtime else [])]]
    return commands


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-server", help="Installed compatible llama.cpp llama-server executable")
    parser.add_argument("--ffmpeg", help="Installed ffmpeg executable")
    parser.add_argument("--ffprobe", help="Installed ffprobe executable")
    parser.add_argument("--cpu", action="store_true", help="CPU compatibility mode; five-minute video deadline is not assured")
    parser.add_argument("--skip-model-downloads", action="store_true", help="Reuse already installed public models")
    parser.add_argument("--plan", action="store_true", help="Print preparation commands without changes or downloads")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or newer is required (3.12 is the tested version)")
    commands = build_plan(ROOT, skip_downloads=args.skip_model_downloads,
                          skip_runtime=bool(args.llama_server) or os.name != "nt")
    if args.plan:
        print(json.dumps({"commands": commands, "note": "No commands executed; external executable paths are checked during real setup"}, indent=2))
        return 0
    try:
        if os.name != "nt" and not args.llama_server:
            raise ValueError("On macOS/Linux pass --llama-server /absolute/path/to/llama-server")
        if args.cpu and not args.llama_server:
            raise ValueError("For CPU mode, pass --llama-server pointing to a compatible CPU build")
        server = executable(args.llama_server, "llama-server") if args.llama_server else None
        ffmpeg = executable(args.ffmpeg, "ffmpeg")
        ffprobe = executable(args.ffprobe, "ffprobe")
        for command in commands:
            print("Preparing: " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        model_dir = ROOT / "tmp/planning/local_runtime"
        server = server or executable(str(model_dir / "bin/llama-server.exe"), "llama-server")
        configuration = {
            "FORTE_LLAMA_SERVER": server,
            "FORTE_FFMPEG": ffmpeg,
            "FORTE_FFPROBE": ffprobe,
            "FORTE_MODEL_PATH": str(model_dir / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
            "FORTE_MMPROJ_PATH": str(model_dir / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"),
            "FORTE_OBJECT_PYTHON": str(python_in(ROOT / ".venv-perception")),
            "FORTE_VLM_BACKEND": "llama.cpp",
            "FORTE_ATTACH_MODEL": "0",
            "FORTE_GPU_LAYERS": "0" if args.cpu else "99",
            "FORTE_VLM_CPU_PROJECTOR": "1" if args.cpu else "0",
        }
        output = ROOT / "data/local-runtime.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        print("Setup complete. Start with: python scripts/run_portable.py")
        print("Runtime configuration saved locally in data/local-runtime.json. No service was started.")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Setup stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
