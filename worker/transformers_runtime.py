"""Optional offline Qwen2.5-VL NF4 engine and bounded OpenAI-compatible server.

Heavy dependencies are imported only by ``load``. Production uses a separate
environment; importing the worker or running contract tests never loads CUDA.
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import threading
from typing import Any
from starlette.requests import Request

from worker.runtime import RUNTIME_PROTOCOL_VERSION, _asset_hashes, _fingerprint, _validate_json_schema
from worker.video import AnalysisError


@dataclass(frozen=True)
class VLMConfig:
    model_dir: Path
    context_tokens: int = 8192
    max_frames: int = 8
    min_pixels: int = 64 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    max_output_tokens: int = 2048
    max_request_bytes: int = 32 * 1024 * 1024
    request_timeout: float = 40.0

    def __post_init__(self) -> None:
        if not 2048 <= self.context_tokens <= 32768:
            raise ValueError("context_tokens must be between 2048 and 32768")
        if not 1 <= self.max_frames <= 8:
            raise ValueError("max_frames must be between 1 and 8")
        if not 28 * 28 <= self.min_pixels <= self.max_pixels <= 2048 * 28 * 28:
            raise ValueError("The image pixel limits are invalid")
        if not 128 <= self.max_output_tokens <= min(4096, self.context_tokens - 1):
            raise ValueError("The output token limit is invalid")
        if not 1024 <= self.max_request_bytes <= 64 * 1024 * 1024:
            raise ValueError("The request byte limit is invalid")
        if not 1 <= self.request_timeout <= 120:
            raise ValueError("request_timeout must be between 1 and 120 seconds")


def validate_model_directory(directory: Path) -> dict:
    """Fail before loading if required local, safe-tensor assets are incomplete."""
    def missing() -> None:
        raise AnalysisError("model_assets_missing", "A complete local Qwen2.5-VL model directory is required; see docs/vlm-runtime.md")

    if not directory.is_dir():
        missing()
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        missing()
    if not isinstance(config, dict) or config.get("model_type") != "qwen2_5_vl":
        raise AnalysisError("model_config_invalid", "The optional Transformers backend requires Qwen2.5-VL weights")
    if config.get("quantization_config"):
        raise AnalysisError("model_config_invalid", "Use the unquantized safe-tensor checkpoint; this backend loads it in NF4 locally")
    if not (directory / "preprocessor_config.json").is_file() or not (directory / "tokenizer_config.json").is_file():
        missing()
    if not any((directory / name).is_file() for name in ("tokenizer.json", "vocab.json")):
        missing()
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            value = json.loads(index.read_text(encoding="utf-8"))
            if not isinstance(value.get("weight_map"), dict):
                missing()
            shard_names = set(value["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            missing()
        if not shard_names:
            missing()
        base = directory.resolve()
        for name in shard_names:
            if (not isinstance(name, str) or not name.endswith(".safetensors")
                    or not (base / name).resolve().is_relative_to(base)
                    or not (base / name).is_file()):
                missing()
    elif not (directory / "model.safetensors").is_file():
        missing()
    return config


def parse_request(payload: object, config: VLMConfig) -> tuple[list[dict], dict, int]:
    """Accept only the local worker's deterministic, schema-bearing image requests."""
    def invalid() -> None:
        raise AnalysisError("model_request_invalid", "Expected bounded messages, embedded JPEGs, and a JSON object schema")

    if not isinstance(payload, dict) or payload.get("stream") is not False or payload.get("temperature") != 0:
        invalid()
    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= config.max_output_tokens:
        invalid()
    try:
        response_format = payload["response_format"]
        descriptor = response_format["json_schema"]
        schema = descriptor["schema"]
        if (response_format["type"] != "json_schema" or descriptor.get("strict") is not True
                or not isinstance(schema, dict) or schema.get("type") != "object"
                or len(json.dumps(schema)) > 32768):
            invalid()
    except (KeyError, TypeError, ValueError, RecursionError):
        invalid()
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 3:
        invalid()
    frames, text_length = 0, 0
    clean_messages = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
            invalid()
        content = message.get("content")
        if not isinstance(content, list) or not 1 <= len(content) <= 24:
            invalid()
        clean_content = []
        for item in content:
            if not isinstance(item, dict):
                invalid()
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_length += len(item["text"])
                clean_content.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image_url":
                frames += 1
                image_url = item.get("image_url")
                if (not isinstance(image_url, dict) or not isinstance(image_url.get("url"), str)
                        or not image_url["url"].startswith("data:image/jpeg;base64,")
                        or len(image_url["url"]) > 8 * 1024 * 1024):
                    invalid()
                clean_content.append({"type": "embedded_jpeg", "data": image_url["url"].split(",", 1)[1]})
            else:
                # URLs, file paths, audio, video and tool calls are never fetched.
                invalid()
        clean_messages.append({"role": message["role"], "content": clean_content})
    if not 1 <= frames <= config.max_frames or text_length > 65536:
        invalid()
    return clean_messages, schema, max_tokens


def decode_messages(messages: list[dict], config: VLMConfig) -> tuple[list[dict], list[Any]]:
    from PIL import Image
    images: list[Any] = []
    clean = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                content.append(item)
                continue
            try:
                data = base64.b64decode(item["data"], validate=True)
                with Image.open(io.BytesIO(data)) as source:
                    if source.format != "JPEG" or source.width * source.height > 16 * 1024 * 1024:
                        raise ValueError("Image exceeds the source pixel budget")
                    image = source.convert("RGB")
                # A processor's max_pixels also bounds vision tokens. This first
                # resize prevents unnecessary original-resolution host tensors.
                if image.width * image.height > config.max_pixels:
                    scale = (config.max_pixels / (image.width * image.height)) ** 0.5
                    image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
                images.append(image)
                content.append({"type": "image"})
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise AnalysisError("model_image_invalid", "An embedded JPEG is invalid or exceeds the image budget") from exc
        clean.append({"role": message["role"], "content": content})
    return clean, images


class TransformersEngine:
    def __init__(self, config: VLMConfig):
        self.config = config
        self.ready = False
        self.model = self.processor = self.torch = None
        self.lock = threading.Lock()
        self.identity: dict = {}
        self.model_id: str | None = None

    def load(self) -> None:
        model_config = validate_model_directory(self.config.model_dir)
        # Defense in depth: both environment and every loader prohibit network.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        except ImportError as exc:
            raise AnalysisError("model_dependencies_missing", "Install requirements.vlm.txt in the isolated CUDA environment") from exc
        if not torch.cuda.is_available():
            raise AnalysisError("model_cuda_unavailable", "The optional NF4 server requires an available CUDA GPU")
        self.torch = torch
        self.schema_parser = JsonSchemaParser
        self.prefix_function = build_transformers_prefix_allowed_tokens_fn
        directory = str(self.config.model_dir.resolve())
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
                                          llm_int8_skip_modules=["visual", "lm_head"])
        try:
            self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                                           min_pixels=self.config.min_pixels, max_pixels=self.config.max_pixels)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                quantization_config=quantization, device_map={"": 0}, dtype=torch.float16,
                attn_implementation="sdpa")
            self.model.eval()
        except Exception as exc:
            self.close()
            raise AnalysisError("model_load_failed", "Offline NF4 model loading failed; inspect the local environment and GPU capacity") from exc
        text_config = model_config.get("text_config", model_config)
        model_context = text_config.get("max_position_embeddings", 32768)
        if self.config.context_tokens > model_context:
            self.close()
            raise AnalysisError("model_config_invalid", "The configured context exceeds the model's context limit")
        if not getattr(self.model, "is_loaded_in_4bit", False):
            self.close()
            raise AnalysisError("model_quantization_invalid", "The model did not load in the required four-bit mode")
        config_hash = hashlib.sha256((self.config.model_dir / "config.json").read_bytes()).hexdigest()
        self.identity = {"config_sha256": config_hash, "model_type": model_config["model_type"],
                         "architectures": model_config.get("architectures", []),
                         "packages": {name: importlib.metadata.version(name) for name in
                                      ("torch", "transformers", "bitsandbytes", "accelerate", "lm-format-enforcer")}}
        relevant = [path for path in self.config.model_dir.iterdir()
                    if path.suffix == ".safetensors" or path.name in {
                        "config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                        "chat_template.json", "vocab.json", "merges.txt", "generation_config.json",
                        "special_tokens_map.json", "added_tokens.json"}]
        self.identity["asset_hashes"] = _asset_hashes(sorted(relevant, key=lambda path: path.name))
        self.model_id = "sha256:" + _fingerprint({"identity": self.identity, "context": self.config.context_tokens,
            "min_pixels": self.config.min_pixels, "max_pixels": self.config.max_pixels,
            "max_frames": self.config.max_frames, "max_output_tokens": self.config.max_output_tokens,
            "quantization": "nf4-double-quant-fp16-exclude-visual-lm_head", "protocol": RUNTIME_PROTOCOL_VERSION})
        self.ready = True

    def metadata(self) -> dict:
        return {"backend": "transformers", "ready": self.ready, "model": self.config.model_dir.name,
                "model_id": self.model_id,
                "quantization": "bitsandbytes-nf4-double-quant; vision/lm_head excluded",
                "strict_schema_decoding": True, "output_validation": True,
                "context_tokens": self.config.context_tokens, "max_frames": self.config.max_frames,
                "min_pixels_per_frame": self.config.min_pixels, "max_pixels_per_frame": self.config.max_pixels,
                "max_output_tokens": self.config.max_output_tokens, "local_files_only": True,
                "request_timeout_sec": self.config.request_timeout,
                "identity": self.identity.copy()}

    def generate(self, payload: object) -> dict:
        if not self.ready:
            raise AnalysisError("model_unavailable", "The optional local model is not ready")
        messages, schema, max_tokens = parse_request(payload, self.config)
        if not self.lock.acquire(blocking=False):
            raise AnalysisError("model_busy", "The local model is already processing a request")
        try:
            conversation, images = decode_messages(messages, self.config)
            # Include the schema in the prompt as well as constraining decoding;
            # observations remain bounded by supplied frame IDs and rule enums.
            conversation.append({"role": "user", "content": [{"type": "text", "text":
                "Return only a JSON object matching this schema. Use unknown/not_visible whenever visual evidence is insufficient. "
                + json.dumps(schema, separators=(",", ":"))}]})
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
            input_tokens = inputs["input_ids"].shape[-1]
            if input_tokens + max_tokens > self.config.context_tokens:
                raise AnalysisError("model_input_budget", "Images, prompt, and output reservation exceed the configured context")
            inputs = inputs.to(self.model.device)
            prefix = self.prefix_function(self.processor.tokenizer, self.schema_parser(schema))
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                             prefix_allowed_tokens_fn=prefix, max_time=self.config.request_timeout)
            generated = output[0][input_tokens:]
            text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
            try:
                result = json.loads(text)
            except (ValueError, TypeError) as exc:
                raise AnalysisError("model_output_truncated", "The local model did not complete its constrained JSON response") from exc
            _validate_json_schema(result, schema)
            return {"model": self.config.model_dir.name, "choices": [{"finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(result)}}],
                    "usage": {"prompt_tokens": input_tokens, "completion_tokens": len(generated)}}
        except AnalysisError:
            raise
        except Exception as exc:
            raise AnalysisError("model_inference_failed", "The local model failed during inference") from exc
        finally:
            self.lock.release()

    def close(self) -> None:
        self.ready = False
        self.model = None
        self.processor = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def create_app(engine: TransformersEngine):
    """Construct the isolated server; dependency injection supports CPU contracts."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    @asynccontextmanager
    async def lifespan(app):
        engine.load()
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(title="Local bounded VLM", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/health")
    def health():
        return JSONResponse({"status": "ok" if engine.ready else "loading"}, status_code=200 if engine.ready else 503)

    @app.get("/v1/runtime")
    def metadata():
        return engine.metadata()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        from starlette.concurrency import run_in_threadpool
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > engine.config.max_request_bytes:
                return JSONResponse({"error": {"code": "model_input_budget"}}, status_code=413)
        try:
            payload = json.loads(body)
            return await run_in_threadpool(engine.generate, payload)
        except (ValueError, RecursionError):
            return JSONResponse({"error": {"code": "model_request_invalid"}}, status_code=422)
        except AnalysisError as exc:
            status = 429 if exc.code == "model_busy" else 503 if exc.code == "model_unavailable" else 422
            return JSONResponse({"error": {"code": exc.code}}, status_code=status)

    return app
