"""Install the pinned local Qwen3-VL 4B verifier and Windows NVIDIA runtime.

Uses only Python's standard library. Model assets total approximately 3.3 GB.
Other systems: use --skip-runtime and supply FORTE_LLAMA_SERVER yourself.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1cd86afb9a95c410a6038ab3b40d8b578c892266"
HF = f"https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/{REVISION}/"
RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/b11707/"
MODELS = (
    ("Qwen3VL-4B-Instruct-Q4_K_M.gguf", 2497281664, "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a", HF + "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
    ("mmproj-Qwen3VL-4B-Instruct-F16.gguf", 836180256, "256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331", HF + "mmproj-Qwen3VL-4B-Instruct-F16.gguf"),
)
RUNTIME = (
    ("runtime.zip", 154749545, "3cbff977614a5a90cbf7abffa16c22b5d28264171671aeaf2619059c206b2d78", RELEASE + "llama-b11707-bin-win-cuda-13.4-x64.zip"),
    ("cudart.zip", 423535356, "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668", RELEASE + "cudart-llama-bin-win-cuda-13.4-x64.zip"),
)


def valid(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest() == digest


def fetch(asset: tuple, destination: Path, check_only: bool = False) -> Path:
    name, size, digest, url = asset
    target = destination / name
    print(f"Checking {name}...", flush=True)
    if valid(target, size, digest):
        return target
    if check_only:
        raise RuntimeError(f"Missing or invalid asset: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    if offset < size:
        headers = {"User-Agent": "Forte-local-setup/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Download returned an unexpected byte range")
            with partial.open("ab" if resumed else "wb") as output:
                downloaded = offset if resumed else 0
                report_at = downloaded
                while chunk := response.read(4 * 1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > size:
                        raise RuntimeError(f"Download exceeds expected size: {name}")
                    if downloaded - report_at >= 128 * 1024 * 1024:
                        print(f"  {name}: {downloaded / size:.0%}", flush=True)
                        report_at = downloaded
    if not valid(partial, size, digest):
        # A completed invalid partial must not poison subsequent resume attempts.
        if partial.exists() and partial.stat().st_size >= size:
            partial.unlink()
        raise RuntimeError(f"Size/SHA256 verification failed: {name}; rerun to retry")
    os.replace(partial, target)
    return target


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            name = item.filename.replace("\\", "/")
            output = (base / name).resolve()
            if not output.is_relative_to(base) or ":" in name or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError(f"Unsafe archive entry: {item.filename}")
            if item.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.name + ".install-part")
                with package.open(item) as source, temporary.open("wb") as sink:
                    shutil.copyfileobj(source, sink, 4 * 1024 * 1024)
                os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/planning/local_runtime")
    parser.add_argument("--skip-runtime", action="store_true", help="Download/check models only; provide your own llama-server")
    parser.add_argument("--check-only", action="store_true", help="Validate existing model files without network or modifications")
    args = parser.parse_args()
    try:
        if os.name != "nt" and not args.skip_runtime:
            raise RuntimeError("Bundled runtime is Windows x64 NVIDIA only; use --skip-runtime and set FORTE_LLAMA_SERVER")
        for asset in MODELS:
            fetch(asset, args.output, args.check_only)
        if not args.skip_runtime:
            if args.check_only:
                executable = args.output / "bin/llama-server.exe"
                if not executable.is_file():
                    raise RuntimeError(f"Missing runtime: {executable}")
            else:
                for asset in RUNTIME:
                    extract(fetch(asset, args.output), args.output / "bin")
                if not (args.output / "bin/llama-server.exe").is_file():
                    raise RuntimeError("Runtime archive did not contain bin/llama-server.exe")
        print("Local verifier assets are ready. NVIDIA CUDA 13.4 requires a compatible driver.")
        return 0
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
