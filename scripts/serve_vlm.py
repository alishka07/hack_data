"""Serve a pre-downloaded Qwen2.5-VL checkpoint in local CUDA NF4 mode."""
from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--check", action="store_true", help="Validate local assets and package availability without loading model/GPU")
    arguments = parser.parse_args()
    try:
        loopback = ipaddress.ip_address(arguments.host).is_loopback
    except ValueError:
        loopback = arguments.host == "localhost"
    if not loopback or not 1 <= arguments.port <= 65535:
        parser.error("The server requires a loopback host and valid TCP port")
    from worker.transformers_runtime import VLMConfig, TransformersEngine, create_app, validate_model_directory
    from worker.video import AnalysisError
    try:
        max_pixels = int(os.environ.get("FORTE_VLM_MAX_PIXELS", str(512 * 28 * 28)))
        config = VLMConfig(arguments.model_dir,
                           context_tokens=int(os.environ.get("FORTE_VLM_CONTEXT", "8192")),
                           max_frames=int(os.environ.get("FORTE_VLM_MAX_FRAMES", "8")),
                           min_pixels=int(os.environ.get("FORTE_VLM_MIN_PIXELS", str(min(64 * 28 * 28, max_pixels)))),
                           max_pixels=max_pixels,
                           max_output_tokens=int(os.environ.get("FORTE_VLM_MAX_OUTPUT_TOKENS", "2048")))
        validate_model_directory(arguments.model_dir)
        if arguments.check:
            import importlib.util
            missing = [name for name in ("torch", "transformers", "accelerate", "bitsandbytes", "lmformatenforcer", "PIL", "uvicorn")
                       if importlib.util.find_spec(name) is None]
            print("Assets: required local safe-tensor files present; tensor contents not loaded")
            print("Packages: " + ("missing " + ", ".join(missing) if missing else "available; CUDA/model loading not tested"))
            return 2 if missing else 0
        import uvicorn
        uvicorn.run(create_app(TransformersEngine(config)), host=arguments.host, port=arguments.port,
                    access_log=False, log_level="warning")
    except (AnalysisError, ValueError) as exc:
        print(f"VLM startup failed: {getattr(exc, 'code', 'configuration_invalid')}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
