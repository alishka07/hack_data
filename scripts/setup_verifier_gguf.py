"""Install a pinned official Qwen2.5-VL-7B four-bit verifier for offline use.

Setup is the only network step. Nothing launches a server or uses the GPU.
Interrupted downloads resume from sibling .part files; an asset becomes visible
under its final name only after its exact size and SHA-256 have been verified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"
REVISION = "508edd0afaa66bb9e9f40587acc2184f02daf1f6"
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
BASE_REVISION_INSPECTED = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"
METADATA_URL = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
BLOCK_SIZE = 8 * 1024 * 1024
DISK_RESERVE = 512 * 1024 * 1024
RANGE_CHUNK = 64 * 1024 * 1024


@dataclass(frozen=True)
class Asset:
    filename: str
    size_bytes: int
    sha256: str
    url: str
    role: str


MODEL = Asset("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4683072032,
              "9258bf05b12686d097ff3b6b18d968ab393649780aa2b3cd67fec43d50554392",
              BASE_URL + "/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "model")
PROJECTORS = {
    "q8_0": Asset("mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf", 853119712,
                  "2ddb555391bae966e412deab9e07b58afa18bcc06930ba0f1c78a3695ab9e506",
                  BASE_URL + "/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf", "projector"),
    "f16": Asset("mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", 1354162912,
                 "c24a7f5fcfc68286f0a217023b6738e73bea4f11787a43e8238d4bb1b8604cde",
                 BASE_URL + "/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf", "projector"),
}
MODEL_CARD = Asset("MODEL_CARD.md", 163,
                   "db8eec445c6e268ccc7d118d9e65b1cddd5d1acddde0d1eca2e285085ad17c19",
                   BASE_URL + "/README.md", "model_card")
LICENSE = Asset("LICENSE-Apache-2.0.txt", 11358,
                "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
                "https://www.apache.org/licenses/LICENSE-2.0.txt", "license")


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, asset: Asset) -> bool:
    return path.is_file() and path.stat().st_size == asset.size_bytes and _checksum(path) == asset.sha256


def verify_metadata(metadata: dict, projector: str) -> None:
    """Cross-check immutable Hugging Face LFS identities before downloading."""
    if metadata.get("sha") != REVISION or metadata.get("cardData", {}).get("license") != "apache-2.0":
        raise ValueError("The pinned model revision or declared Apache 2.0 license did not match")
    files = {item["rfilename"]: item for item in metadata.get("siblings", [])}
    for asset in (MODEL, PROJECTORS[projector]):
        item = files.get(asset.filename, {})
        if (item.get("size") != asset.size_bytes or item.get("lfs", {}).get("sha256") != asset.sha256
                or item.get("lfs", {}).get("size") != asset.size_bytes):
            raise ValueError("An official model asset did not match its pinned LFS identity")


def _ensure_asset(asset: Asset, destination: Path, *, opener=None, progress=None) -> bool:
    """Return whether bytes were installed, retaining partials on interruption."""
    if _verified(destination, asset):
        return False
    if destination.name != asset.filename or Path(asset.filename).name != asset.filename:
        raise ValueError("Asset filenames must be plain pinned basenames")
    if not re.fullmatch(r"[0-9a-f]{64}", asset.sha256) or asset.size_bytes <= 0:
        raise ValueError("Assets require exact SHA-256 and positive size")
    opener = opener or urllib.request.urlopen
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > asset.size_bytes:
        raise ValueError("A partial asset exceeds the pinned size")
    if offset == asset.size_bytes:
        if _checksum(partial) != asset.sha256:
            partial.unlink()
            raise ValueError("A completed partial asset failed SHA-256 verification; it was discarded")
        partial.replace(destination)
        return True
    headers = {"User-Agent": "forte-local-verifier-setup/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(asset.url, headers=headers)
    with opener(request, timeout=90) as response:
        status = response.status
        if status == 206:
            expected = f"bytes {offset}-{asset.size_bytes - 1}/{asset.size_bytes}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("The download server returned an unexpected resume range")
        elif status == 200:
            # Some CDNs ignore Range. Restart this private temporary file,
            # retaining any existing final asset until the new one verifies.
            offset = 0
        else:
            raise ValueError("The download server returned an unsupported status")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != asset.size_bytes - offset:
            raise ValueError("The download server returned an unexpected asset length")
        total, logged = offset, offset
        with partial.open("ab" if offset else "wb") as stream:
            while True:
                block = response.read(BLOCK_SIZE)
                if not block:
                    break
                total += len(block)
                if total > asset.size_bytes:
                    raise ValueError("Downloaded asset exceeds its pinned size")
                stream.write(block)
                if progress and (total - logged >= 128 * 1024 * 1024 or total == asset.size_bytes):
                    progress({"stage": "download", "file": asset.filename, "bytes": total, "expected_bytes": asset.size_bytes})
                    logged = total
            stream.flush()
            os.fsync(stream.fileno())
    if total != asset.size_bytes:
        raise OSError("Download ended early; verified-size partial retained for a resume")
    if _checksum(partial) != asset.sha256:
        partial.unlink()
        raise ValueError("Downloaded asset failed SHA-256 verification; partial discarded")
    partial.replace(destination)
    return True


def _ensure_parallel_asset(asset: Asset, destination: Path, connections: int, *, opener=None, progress=None,
                           chunk_size: int = RANGE_CHUNK) -> bool:
    """Resume bounded, checked byte ranges and verify the assembled whole file.

    The existing sequential .part is a preserved prefix. Range files have stable
    offsets so a retry changes neither the plan nor the bytes already received.
    Responses must be 206 with an exact Content-Range; no silently ignored Range
    request can contaminate the concatenation.
    """
    if _verified(destination, asset):
        return False
    if connections not in (2, 4, 8, 16) or chunk_size < 1:
        raise ValueError("Parallel downloads require 2, 4, 8 or 16 checked connections")
    opener = opener or urllib.request.urlopen
    prefix = destination.with_name(destination.name + ".part")
    prefix_bytes = prefix.stat().st_size if prefix.exists() else 0
    if prefix_bytes >= asset.size_bytes:
        return _ensure_asset(asset, destination, opener=opener, progress=progress)
    plan_path = destination.with_name(destination.name + ".ranges.json")
    plan = {"sha256": asset.sha256, "size_bytes": asset.size_bytes, "prefix_bytes": prefix_bytes,
            "chunk_size": chunk_size}
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("The retained range plan differs from the pinned asset or prefix")
    else:
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
    ranges = [(start, min(start + chunk_size, asset.size_bytes) - 1)
              for start in range(prefix_bytes, asset.size_bytes, chunk_size)]
    parts = {start: destination.with_name(f"{destination.name}.range-{start}-{end}.part") for start, end in ranges}
    lock = threading.Lock()
    completed = prefix_bytes + sum(min(path.stat().st_size, end - start + 1) if path.exists() else 0
                                   for start, end in ranges for path in [parts[start]])
    def fetch(pair):
        nonlocal completed
        start, end = pair
        part, expected = parts[start], end - start + 1
        for attempt in range(3):
            offset = part.stat().st_size if part.exists() else 0
            if offset == expected:
                return
            if offset > expected:
                raise ValueError("A retained byte-range file exceeds its pinned extent")
            request_start = start + offset
            # Different bounded ranges use distinct resolver cache keys. The
            # immutable revision remains the authority for the requested bytes.
            url = asset.url + f"?download=true&forte_range={request_start}-{end}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={request_start}-{end}",
                  "Accept-Encoding": "identity", "User-Agent": "forte-local-verifier-setup/1"})
            try:
                with opener(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {request_start}-{end}/{asset.size_bytes}":
                        raise ValueError("A parallel response did not match the exact requested byte range")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != expected - offset:
                        raise ValueError("A parallel response has an unexpected length")
                    received = offset
                    with part.open("ab" if offset else "wb") as stream:
                        while True:
                            block = response.read(min(BLOCK_SIZE, expected - received + 1))
                            if not block:
                                break
                            received += len(block)
                            if received > expected:
                                raise ValueError("A byte-range response exceeded its pinned extent")
                            stream.write(block)
                            with lock:
                                completed += len(block)
                        stream.flush()
                        os.fsync(stream.fileno())
                if received != expected:
                    raise OSError("A byte-range download ended early; partial retained")
                if progress:
                    with lock:
                        progress({"stage": "parallel_download", "file": asset.filename, "bytes": completed,
                                  "expected_bytes": asset.size_bytes, "connections": connections})
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=connections) as pool:
        list(pool.map(fetch, ranges))
    assembly = destination.with_name(destination.name + ".assembled.part")
    digest, total = hashlib.sha256(), 0
    with assembly.open("wb") as target:
        sources = ([prefix] if prefix_bytes else []) + [parts[start] for start, _ in ranges]
        for path in sources:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(BLOCK_SIZE), b""):
                    target.write(block)
                    digest.update(block)
                    total += len(block)
        target.flush()
        os.fsync(target.fileno())
    if total != asset.size_bytes or digest.hexdigest() != asset.sha256:
        assembly.unlink()
        for path in [*parts.values(), plan_path, *([prefix] if prefix_bytes else [])]:
            path.unlink()
        raise ValueError("Assembled byte ranges failed the pinned full-file size or SHA-256 check; unverified parts discarded")
    assembly.replace(destination)
    for path in [*parts.values(), plan_path, *([prefix] if prefix_bytes else [])]:
        path.unlink()
    return True


def build_manifest(projector: str) -> dict:
    assets = (MODEL, PROJECTORS[projector], MODEL_CARD, LICENSE)
    return {"schema_version": "forte-local-verifier-gguf/1", "model_family": "Qwen2.5-VL-7B-Instruct",
            "repository": REPOSITORY, "revision": REVISION, "base_model": BASE_MODEL,
            "base_revision_inspected": BASE_REVISION_INSPECTED, "model_quantization": "Q4_K_M",
            "projector_quantization": projector.upper(), "license": "Apache-2.0",
            "license_declaration": MODEL_CARD.url, "metadata_url": METADATA_URL,
            "assets": [{"role": asset.role, "filename": asset.filename, "size_bytes": asset.size_bytes,
                        "sha256": asset.sha256, "url": asset.url} for asset in assets],
            "matching_projector": "Both language GGUF and projector are from the same immutable maintainer repository revision",
            "inference_network": "disabled", "installation_scope": "assets_only_no_server_start_or_gpu_use",
            "memory_fit": "Not established by download or weight sizes; benchmark actual runtime frame and context budgets"}


def setup_assets(output: Path, projector: str = "q8_0", *, verify_only: bool = False, progress=None, connections: int = 8) -> dict:
    if projector not in PROJECTORS:
        raise ValueError("Only official pinned Q8_0 and F16 projectors are supported")
    output = Path(output).resolve()
    assets = (MODEL, PROJECTORS[projector], MODEL_CARD, LICENSE)
    downloaded = []
    if verify_only:
        if not all(_verified(output / asset.filename, asset) for asset in assets):
            raise ValueError("A local verifier asset is missing or failed its pinned checksum")
        manifest = output / "manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != build_manifest(projector):
            raise ValueError("The local verifier manifest does not match the pinned assets")
    else:
        output.mkdir(parents=True, exist_ok=True)
        # Inspect and verify official metadata before a multi-gigabyte download.
        with urllib.request.urlopen(METADATA_URL, timeout=30) as response:
            metadata = json.load(response)
        verify_metadata(metadata, projector)
        required = 0
        for asset in assets:
            if not _verified(output / asset.filename, asset):
                partial = output / (asset.filename + ".part")
                prefix_bytes = partial.stat().st_size if partial.is_file() else 0
                retained = prefix_bytes
                plan_path = output / (asset.filename + ".ranges.json")
                if plan_path.exists():
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    if (plan.get("sha256") == asset.sha256 and plan.get("size_bytes") == asset.size_bytes
                            and plan.get("prefix_bytes") == prefix_bytes and plan.get("chunk_size") == RANGE_CHUNK):
                        for start in range(prefix_bytes, asset.size_bytes, RANGE_CHUNK):
                            end = min(start + RANGE_CHUNK, asset.size_bytes) - 1
                            part = output / f"{asset.filename}.range-{start}-{end}.part"
                            if part.is_file():
                                retained += min(part.stat().st_size, end - start + 1)
                required += max(0, asset.size_bytes - retained)
        free = shutil.disk_usage(output).free
        if progress:
            progress({"stage": "preflight", "required_download_bytes": required, "disk_free_bytes": free,
                      "model_bytes": MODEL.size_bytes, "projector_bytes": PROJECTORS[projector].size_bytes, "revision": REVISION})
        assembly_space = max(asset.size_bytes for asset in assets) if connections > 1 else 0
        if free < required + assembly_space + DISK_RESERVE:
            raise OSError("Insufficient free disk space for pinned verifier assets and 512 MiB reserve")
        for asset in assets:
            for attempt in range(3):
                try:
                    ensure = _ensure_parallel_asset if connections > 1 and asset.size_bytes > RANGE_CHUNK else _ensure_asset
                    options = {"connections": connections} if ensure is _ensure_parallel_asset else {}
                    if ensure(asset, output / asset.filename, progress=progress, **options):
                        downloaded.append(asset.filename)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    if progress:
                        progress({"stage": "retry", "file": asset.filename, "attempt": attempt + 2, "partial_retained": True})
                    time.sleep(2)
        temporary = output / "manifest.json.part"
        temporary.write_text(json.dumps(build_manifest(projector), indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "manifest.json")
    return {"status": "verified", "revision": REVISION, "downloaded_files": downloaded,
            "manifest": str(output / "manifest.json"), "model_path": str(output / MODEL.filename),
            "projector_path": str(output / PROJECTORS[projector].filename),
            "environment": {"FORTE_MODEL_PATH": str(output / MODEL.filename),
                            "FORTE_MMPROJ_PATH": str(output / PROJECTORS[projector].filename)}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "vlm" / "qwen2.5-vl-7b")
    parser.add_argument("--projector", choices=tuple(PROJECTORS), default="q8_0")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--connections", type=int, choices=(1, 2, 4, 8, 16), default=8)
    args = parser.parse_args(argv)
    def progress(value):
        print(json.dumps(value), flush=True)
    try:
        result = setup_assets(args.output, args.projector, verify_only=args.verify_only, progress=progress, connections=args.connections)
    except (OSError, ValueError) as exc:
        parser.exit(1, json.dumps({"error": str(exc)}) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
