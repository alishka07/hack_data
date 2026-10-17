"""One-time public YOLOX download. Inference never calls this script/network."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
LICENSE = "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/0.1.1rc0/LICENSE"
EXPECTED_SHA256 = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"


def setup(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    model = destination / "yolox_nano.onnx"
    if not model.is_file():
        temporary = destination / "yolox_nano.onnx.part"
        with urllib.request.urlopen(SOURCE, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if temporary.stat().st_size != 3659407:
            raise RuntimeError("Official release asset size differs from the reviewed release")
        temporary.replace(model)
    with model.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if model.stat().st_size != 3659407 or digest != EXPECTED_SHA256:
        raise RuntimeError("YOLOX model does not match the reviewed official release bytes")
    with urllib.request.urlopen(LICENSE, timeout=30) as response:
        license_text = response.read().decode("utf-8")
    if "Apache License" not in license_text:
        raise RuntimeError("Official model repository license is not the expected Apache license")
    (destination / "YOLOX-LICENSE.txt").write_text(license_text, encoding="utf-8")
    manifest = {"format": "yolox-raw-coco-v1", "input_size": [416, 416], "sha256": digest,
                "source_url": SOURCE, "source_release": "0.1.1rc0", "license": "Apache-2.0",
                "license_url": LICENSE, "bytes": model.stat().st_size,
                "installed_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Local anonymous person proposals; COCO person class only; uncalibrated"}
    model.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models" / "perception")
    args = parser.parse_args()
    print(json.dumps(setup(args.destination), indent=2))
