"""Label-free fingerprints shared by semantic feature training and inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import CodeType

from worker.semantic_encoder import _validate_json_schema as encoder_validator, semantic_recipe
from worker.semantic_features import SEMANTIC_PROMPT, SEMANTIC_VERSION, observable_schema
from worker.semantic_runtime import _validate_json_schema as frozen_validator


def _digest(path: Path) -> dict:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return {"sha256": value.hexdigest(), "bytes": path.stat().st_size}


def _code_signature(code: CodeType) -> tuple:
    """Compare executed helper code without filenames or source line positions."""
    constants = tuple(_code_signature(value) if isinstance(value, CodeType) else value for value in code.co_consts)
    return (code.co_code, constants, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
            code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags, code.co_exceptiontable)


def encoder_recipe(root: str | Path) -> dict:
    """Hash actual local assets before startup; never store filesystem paths."""
    if _code_signature(encoder_validator.__code__) != _code_signature(frozen_validator.__code__):
        raise ValueError("Shared semantic schema validation differs from the frozen encoder recipe")
    root = Path(root)
    defaults = root / "tmp/planning/local_runtime"
    paths = {
        "model": Path(os.environ.get("FORTE_MODEL_PATH", defaults / "Qwen3VL-4B-Instruct-Q4_K_M.gguf")),
        "projector": Path(os.environ.get("FORTE_MMPROJ_PATH", defaults / "mmproj-Qwen3VL-4B-Instruct-F16.gguf")),
        "llama_server": Path(os.environ.get("FORTE_LLAMA_SERVER", defaults / "bin/llama-server.exe")),
    }
    package = Path(__file__).resolve().parent
    return {"format_version": 1, "semantic_version": SEMANTIC_VERSION,
            "observation": semantic_recipe(), "prompt": SEMANTIC_PROMPT, "schema": observable_schema(),
            "runtime": {"context_tokens": 8192, "max_output_tokens": 768, "temperature": 0,
                        "threads": 8, "parallel": 1, "flash_attention": "on",
                        "gpu_layers": os.environ.get("FORTE_GPU_LAYERS", "99")},
            "assets": {name: _digest(path) for name, path in paths.items()},
            "implementation": {name + "_sha256": _digest(package / (filename + ".py"))["sha256"]
                               for name, filename in (("semantic_encoder", "semantic_encoder"),
                                                      ("semantic_features", "semantic_features"),
                                                      ("runtime", "semantic_runtime"))}}


def recipe_fingerprint(recipe: dict) -> str:
    """Canonical JSON identity, independent of dictionary insertion order."""
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
