"""Explicitly provision the pinned official R3D-18 assets for offline encoding.

This setup command is the only network step. Existing verified assets are
reused; inference never downloads anything. Install requirements.training.txt
in the separate training environment before using worker.action_encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_URL = "https://download.pytorch.org/models/r3d_18-b3b3357e.pth"
WEIGHTS_FILENAME = "r3d_18-b3b3357e.pth"
WEIGHTS_BYTES = 133546016
WEIGHTS_SHA256 = "b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d"
LICENSE_URL = "https://raw.githubusercontent.com/pytorch/vision/v0.28.0/LICENSE"
LICENSE_FILENAME = "LICENSE-torchvision.txt"
LICENSE_BYTES = 1517
LICENSE_SHA256 = "6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d"


def build_manifest() -> dict:
    """Return the canonical provenance, including the installed source revision."""
    return {
        "encoder": "r3d_18",
        "pretraining": "KINETICS400_V1",
        "feature_dimension": 512,
        "source_code_version": "torchvision==0.28.0+cpu",
        "source_url": "https://github.com/pytorch/vision/tree/v0.28.0/torchvision/models/video",
        "weights_url": WEIGHTS_URL,
        "weights_filename": WEIGHTS_FILENAME,
        "weights_bytes": WEIGHTS_BYTES,
        "weights_sha256": WEIGHTS_SHA256,
        "license_file": LICENSE_FILENAME,
        "license_url": LICENSE_URL,
        "preprocessing": "full_frame_letterbox112_kinetics400_mean_std_v1",
        "sampling": {"clip_frames": 16, "window_sec": 8, "stride_sec": 4,
                     "targets": "uniform_including_window_endpoints",
                     "mapping": "nearest_authoritative_source_PTS"},
        "inference_network": "disabled",
        "pretraining_shift": "Uses full-frame letterboxing instead of the pretrained center crop; about 2 fps instead of the original 15 fps evaluation rate.",
        "source_file": "torchvision/models/video/resnet.py",
        "source_file_sha256": "ed78cebf305525042621709335c6927ebbca30ae7a041837c2ddae2a18cbb279",
        "source_revision": "8fb87713a24951e639c494b0f2a8a81b5f8e33a6",
    }


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_asset(url: str, destination: Path, expected_sha256: str, expected_bytes: int) -> bool:
    """Download to a sibling temporary file, verifying before atomic replace."""
    if url not in {WEIGHTS_URL, LICENSE_URL}:
        raise ValueError("Only the pinned official action-encoder asset URLs are allowed")
    if (destination.is_file() and destination.stat().st_size == expected_bytes
            and _checksum(destination) == expected_sha256):
        return False
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".action-download-", suffix=".tmp",
                                         dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            digest, total = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_bytes:
                        raise ValueError("An action asset exceeded its pinned size")
                    digest.update(chunk)
                    output.write(chunk)
        if total != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError("An action asset failed its pinned size or SHA-256 check")
        temporary.replace(destination)
        return True
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def setup_assets(output: Path) -> list[str]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for url, name, checksum, size in (
        (WEIGHTS_URL, WEIGHTS_FILENAME, WEIGHTS_SHA256, WEIGHTS_BYTES),
        (LICENSE_URL, LICENSE_FILENAME, LICENSE_SHA256, LICENSE_BYTES),
    ):
        if _ensure_asset(url, output / name, checksum, size):
            downloaded.append(name)
    manifest_path = output / "manifest.json"
    content = json.dumps(build_manifest(), indent=2) + "\n"
    if not manifest_path.is_file() or manifest_path.read_text(encoding="utf-8") != content:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             prefix=".action-manifest-", suffix=".tmp",
                                             dir=output, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(manifest_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return downloaded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/action-encoder")
    args = parser.parse_args(argv)
    try:
        downloaded = setup_assets(args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, json.dumps({"error": str(exc)}) + "\n")
    print(json.dumps({"encoder": "r3d_18", "weights_sha256": WEIGHTS_SHA256,
                      "downloaded_files": downloaded, "inference_network": "disabled"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
