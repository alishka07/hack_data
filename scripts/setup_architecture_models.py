"""Explicit public model installation; inference never downloads these assets.

Examples:
  python scripts/setup_architecture_models.py --pose
  python scripts/setup_architecture_models.py --objects
  python scripts/setup_architecture_models.py --pose --objects --check
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POSE_SOURCE = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
               "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip")
POSE_LICENSE = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/LICENSE"
POSE_ARCHIVE_SHA256 = "937003a70832d9cc34ea16927f504792f3133e92dda1b9c626236bbbe9e805cb"
POSE_MODEL_SHA256 = "a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f"
OWL_REPOSITORY = "google/owlv2-base-patch16-ensemble"
OWL_SOURCE = f"https://huggingface.co/{OWL_REPOSITORY}"
OWL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(url, path, max_bytes):
    """Bounded streaming download and atomic replacement within destination."""
    temporary = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > max_bytes:
                    raise RuntimeError("Public model asset exceeds the configured size limit")
                output.write(block)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def setup_pose(destination):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "rtmpose_t.zip"
    model = destination / "rtmpose_t.onnx"
    if not archive.is_file():
        download(POSE_SOURCE, archive, 50 * 1024 * 1024)
    if digest(archive) != POSE_ARCHIVE_SHA256:
        raise RuntimeError("RTMPose archive differs from the verified official export")
    with zipfile.ZipFile(archive) as zipped:
        onnx = [item for item in zipped.infolist() if item.filename.endswith(".onnx")]
        if len(onnx) != 1 or onnx[0].file_size > 50 * 1024 * 1024:
            raise RuntimeError("Expected exactly one bounded ONNX asset in the official RTMPose archive")
        # Copy exactly the reviewed member, never extract archive paths.
        with zipped.open(onnx[0]) as source, model.open("wb") as output:
            shutil.copyfileobj(source, output)
    if digest(model) != POSE_MODEL_SHA256:
        raise RuntimeError("RTMPose ONNX differs from the verified official export")
    license_file = destination / "RTMPose-LICENSE.txt"
    download(POSE_LICENSE, license_file, 100000)
    if "Apache License" not in license_file.read_text(encoding="utf-8"):
        raise RuntimeError("Unexpected RTMPose repository license")
    manifest = {"format": "rtmpose-simcc-coco17-v1", "source_url": POSE_SOURCE,
                "license": "Apache-2.0", "license_url": POSE_LICENSE, "sha256": digest(model),
                "archive_sha256": digest(archive), "input_size": [192, 256], "color_order": "BGR",
                "simcc_split_ratio": 2., "keypoints": "COCO-17", "bytes": model.stat().st_size,
                "installed_utc": datetime.now(timezone.utc).isoformat(),
                "integrity_scope": "SHA256 captured from official HTTPS download; checked on every model load",
                "preprocessing_reference": "https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/examples/onnxruntime/main.py"}
    model.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def setup_objects(destination):
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{OWL_REPOSITORY}/revision/{OWL_REVISION}", timeout=60) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    if revision != OWL_REVISION:
        raise RuntimeError("The public model repository did not resolve to an immutable revision")
    allowed = {"config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
               "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors", "README.md"}
    available = {entry["rfilename"] for entry in metadata["siblings"]}
    if not {"model.safetensors", "config.json", "preprocessor_config.json"} <= available:
        raise RuntimeError("OWLv2 repository is missing expected safe model assets")
    files = {}
    for filename in sorted(available & allowed):
        output = destination / filename
        download(f"{OWL_SOURCE}/resolve/{revision}/{filename}", output, 2 * 1024**3)
        files[filename] = digest(output)
    if "apache-2.0" not in (destination / "README.md").read_text(encoding="utf-8").lower():
        raise RuntimeError("Unexpected OWLv2 model card license")
    manifest = {"format": "owlv2-local-v1", "source_url": OWL_SOURCE, "revision": revision,
                "license": "Apache-2.0", "license_url": f"{OWL_SOURCE}/blob/{revision}/README.md", "files": files,
                "installed_utc": datetime.now(timezone.utc).isoformat(),
                "integrity_scope": "Immutable source revision and per-file SHA256 captured at installation",
                "purpose": "Sparse local open-vocabulary object hypotheses; not calibrated rule decisions"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def check(destination, pose, objects):
    import numpy as np
    from worker.perception_models import optional_backend
    result = {}
    image = np.zeros((256, 192, 3), dtype=np.uint8)
    if pose:
        started = time.monotonic()
        backend, capability = optional_backend("pose", "rtmpose", destination / "rtmpose_t.onnx", .3)
        output = backend.estimate(image, [{"id": "synthetic-smoke", "track_id": "anonymous", "bbox": [.1, .1, .9, .9]}])
        result["pose"] = {"backend": backend.name, "decoded_keypoints": len(output[0]["keypoints"]),
                          "load_and_inference_sec": time.monotonic() - started, "capability": capability,
                          "note": "Synthetic pixel smoke check, not an accuracy evaluation"}
        backend.close()
    if objects:
        started = time.monotonic()
        backend, capability = optional_backend("objects", "owlv2", destination / "owlv2", .15)
        loaded = time.monotonic()
        output = backend.detect(image)
        result["objects"] = {"backend": backend.name, "output_boxes": len(output),
                             "load_sec": loaded - started, "inference_sec": time.monotonic() - loaded,
                             "capability": capability,
                             "note": "Synthetic pixel smoke check, not an accuracy evaluation"}
        backend.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models/perception")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--objects", action="store_true", help="Download approximately 620 MB of OWLv2 weights")
    parser.add_argument("--check", action="store_true", help="Run local CPU smoke inference after installation")
    parser.add_argument("--check-only", action="store_true", help="Check installed assets offline without downloading")
    args = parser.parse_args()
    if not args.pose and not args.objects:
        parser.error("Choose --pose and/or --objects explicitly")
    result = {}
    if args.pose and not args.check_only:
        result["pose"] = setup_pose(args.destination)
    if args.objects and not args.check_only:
        result["objects"] = setup_objects(args.destination / "owlv2")
    if args.check or args.check_only:
        result["checks"] = check(args.destination, args.pose, args.objects)
    print(json.dumps(result, indent=2))
