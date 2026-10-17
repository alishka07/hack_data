"""Optional VLM contracts without downloads, private media, Torch, or a GPU."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from worker.runtime import LocalRuntime
from worker.transformers_runtime import (VLMConfig, TransformersEngine, create_app,
                                         parse_request, validate_model_directory)
from worker.video import AnalysisError, Budget, FrameRef


SCHEMA = {"type": "object", "properties": {"status": {"type": "string", "enum": ["not_visible", "supported"]}},
          "required": ["status"], "additionalProperties": False}


def payload(count=4):
    return {"model": "local-vlm", "temperature": 0, "stream": False, "max_tokens": 128,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Inspect supplied frames"}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,c3ludGhldGlj"}}
                for _ in range(count)]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": SCHEMA}}}


def assets(tmp_path, shards=False):
    config = {"model_type": "qwen2_5_vl", "architectures": ["Qwen2_5_VLForConditionalGeneration"],
              "text_config": {"max_position_embeddings": 32768}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in ["preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"]:
        (tmp_path / name).write_text("{}")
    if shards:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}))
        (tmp_path / "one.safetensors").write_bytes(b"synthetic")
    else:
        (tmp_path / "model.safetensors").write_bytes(b"synthetic")
    return config


@pytest.mark.parametrize("changes", [{"max_frames": 9}, {"max_frames": 0}, {"context_tokens": 512},
                                     {"max_output_tokens": 8192}, {"min_pixels": 900000, "max_pixels": 400000}])
def test_invalid_inference_limits_fail_before_model_import(tmp_path, changes):
    with pytest.raises(ValueError):
        VLMConfig(tmp_path, **changes)


def test_complete_local_assets_and_all_shards_required(tmp_path):
    assets(tmp_path, shards=True)
    with pytest.raises(AnalysisError) as error:
        validate_model_directory(tmp_path)
    assert error.value.code == "model_assets_missing"
    (tmp_path / "two.safetensors").write_bytes(b"synthetic")
    assert validate_model_directory(tmp_path)["model_type"] == "qwen2_5_vl"


def test_shard_index_cannot_escape_local_directory(tmp_path):
    assets(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(AnalysisError):
        validate_model_directory(tmp_path)


def test_prequantized_or_wrong_model_rejected(tmp_path):
    config = assets(tmp_path)
    config["quantization_config"] = {"quant_method": "awq"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(AnalysisError) as error:
        validate_model_directory(tmp_path)
    assert error.value.code == "model_config_invalid"


@pytest.mark.parametrize("count", [1, 3, 4, 8])
def test_scene_and_candidate_frame_counts_are_supported(tmp_path, count):
    messages, schema, tokens = parse_request(payload(count), VLMConfig(tmp_path))
    assert len([item for item in messages[0]["content"] if item["type"] == "embedded_jpeg"]) == count
    assert schema == SCHEMA and tokens == 128


@pytest.mark.parametrize("count", [0, 9])
def test_unbounded_or_empty_visual_requests_are_rejected(tmp_path, count):
    with pytest.raises(AnalysisError):
        parse_request(payload(count), VLMConfig(tmp_path))


@pytest.mark.parametrize("url", ["https://remote.invalid/frame.jpg", "file:///private.jpg", "data:image/png;base64,YQ=="])
def test_images_are_embedded_jpeg_only(tmp_path, url):
    request = payload()
    request["messages"][0]["content"][1]["image_url"]["url"] = url
    with pytest.raises(AnalysisError):
        parse_request(request, VLMConfig(tmp_path))


def test_request_requires_strict_schema_and_deterministic_output(tmp_path):
    request = payload()
    del request["response_format"]
    with pytest.raises(AnalysisError):
        parse_request(request, VLMConfig(tmp_path))
    request = payload()
    request["temperature"] = 1
    with pytest.raises(AnalysisError):
        parse_request(request, VLMConfig(tmp_path))


def fake_loaded_engine(monkeypatch, tmp_path, input_tokens=100, answer='{"status":"not_visible"}'):
    engine = TransformersEngine(VLMConfig(tmp_path))
    engine.ready = True
    ids = SimpleNamespace(shape=(1, input_tokens))

    class Inputs(dict):
        def to(self, device):
            assert device == "cuda:0"
            return self

    engine.processor = Mock(return_value=Inputs(input_ids=ids))
    engine.processor.tokenizer.decode.return_value = answer
    engine.model = Mock(device="cuda:0")
    engine.model.generate.return_value = [[1] * input_tokens + [2, 3]]
    engine.torch = SimpleNamespace(inference_mode=nullcontext)
    engine.schema_parser = Mock(return_value="parser")
    engine.prefix_function = Mock(return_value="prefix-filter")
    monkeypatch.setattr("worker.transformers_runtime.decode_messages", lambda messages, config: (messages, ["synthetic-image"]))
    return engine


def test_generation_uses_schema_token_filter_and_host_validation(monkeypatch, tmp_path):
    engine = fake_loaded_engine(monkeypatch, tmp_path)
    result = engine.generate(payload())
    assert json.loads(result["choices"][0]["message"]["content"]) == {"status": "not_visible"}
    options = engine.model.generate.call_args.kwargs
    assert options["prefix_allowed_tokens_fn"] == "prefix-filter"
    assert options["do_sample"] is False
    engine.schema_parser.assert_called_once_with(SCHEMA)
    assert not engine.lock.locked()


@pytest.mark.parametrize("answer", ['{"status":"invented"}', "{}", '{"status":'])
def test_invalid_or_truncated_output_never_becomes_a_success(monkeypatch, tmp_path, answer):
    engine = fake_loaded_engine(monkeypatch, tmp_path, answer=answer)
    with pytest.raises(AnalysisError) as error:
        engine.generate(payload())
    assert error.value.code in {"model_schema_invalid", "model_output_truncated"}
    assert not engine.lock.locked()


def test_exact_processed_context_budget_is_checked_before_gpu_generation(monkeypatch, tmp_path):
    engine = fake_loaded_engine(monkeypatch, tmp_path, input_tokens=8190)
    with pytest.raises(AnalysisError) as error:
        engine.generate(payload())
    assert error.value.code == "model_input_budget"
    engine.model.generate.assert_not_called()


def test_generation_is_single_owner(monkeypatch, tmp_path):
    engine = fake_loaded_engine(monkeypatch, tmp_path)
    engine.lock.acquire()
    try:
        with pytest.raises(AnalysisError) as error:
            engine.generate(payload())
        assert error.value.code == "model_busy"
    finally:
        engine.lock.release()


def test_loader_uses_only_local_safe_weights_and_required_four_bit_mode(monkeypatch, tmp_path):
    assets(tmp_path)
    model = Mock(is_loaded_in_4bit=True)
    processor = Mock()
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = Mock()
    fake_transformers.AutoProcessor.from_pretrained.return_value = processor
    fake_transformers.Qwen2_5_VLForConditionalGeneration = Mock()
    fake_transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.return_value = model
    fake_transformers.BitsAndBytesConfig = Mock()
    fake_torch = ModuleType("torch")
    fake_torch.cuda = Mock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.float16 = "fp16"
    fake_enforcer = ModuleType("lmformatenforcer")
    fake_enforcer.JsonSchemaParser = Mock()
    fake_integration = ModuleType("lmformatenforcer.integrations.transformers")
    fake_integration.build_transformers_prefix_allowed_tokens_fn = Mock()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "lmformatenforcer", fake_enforcer)
    monkeypatch.setitem(sys.modules, "lmformatenforcer.integrations", ModuleType("lmformatenforcer.integrations"))
    monkeypatch.setitem(sys.modules, "lmformatenforcer.integrations.transformers", fake_integration)
    monkeypatch.setattr("worker.transformers_runtime.importlib.metadata.version", lambda name: "synthetic")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    engine = TransformersEngine(VLMConfig(tmp_path))
    engine.load()
    options = fake_transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.call_args.kwargs
    assert options["local_files_only"] is True and options["trust_remote_code"] is False
    assert options["use_safetensors"] is True
    assert options["device_map"] == {"": 0}
    assert fake_transformers.BitsAndBytesConfig.call_args.kwargs["load_in_4bit"] is True
    assert engine.ready and engine.metadata()["strict_schema_decoding"] is True
    assert len(engine.metadata()["identity"]["config_sha256"]) == 64
    assert engine.metadata()["model_id"].startswith("sha256:")
    assert engine.metadata()["identity"]["asset_hashes"]


def test_server_has_lifecycle_readiness_and_bounded_request_body(tmp_path):
    engine = Mock(config=VLMConfig(tmp_path, max_request_bytes=1024), ready=False)
    engine.load.side_effect = lambda: setattr(engine, "ready", True)
    engine.close.side_effect = lambda: setattr(engine, "ready", False)
    engine.metadata.return_value = {"backend": "synthetic", "strict_schema_decoding": True}
    engine.generate.return_value = {"choices": []}
    with TestClient(create_app(engine)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/v1/runtime").json()["strict_schema_decoding"] is True
        assert client.post("/v1/chat/completions", content=b"x" * 1025).status_code == 413
        assert client.post("/v1/chat/completions", content=b"broken").status_code == 422
        assert client.post("/v1/chat/completions", json={}).status_code == 200
    assert engine.ready is False


def test_client_resizes_inference_copy_preserving_original_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("FORTE_VLM_MAX_PIXELS", "50176")
    monkeypatch.setenv("FORTE_LLM_URL", "http://127.0.0.1:18080")
    runtime = LocalRuntime(tmp_path, tmp_path)
    source = tmp_path / "source.jpg"
    cv2.imwrite(str(source), np.zeros((500, 1000, 3), dtype=np.uint8))
    original = source.read_bytes()
    request_image = runtime._inference_image(source)
    resized = cv2.imdecode(np.frombuffer(request_image, np.uint8), cv2.IMREAD_COLOR)
    assert resized.shape[0] * resized.shape[1] <= runtime.max_pixels
    assert source.read_bytes() == original
    assert runtime.metadata()["model"].endswith("Q4_K_M.gguf")
    assert runtime.metadata()["max_frames"] == 8


def test_client_rejects_ninth_frame_before_reading_or_network(monkeypatch, tmp_path):
    runtime = LocalRuntime(tmp_path, tmp_path)
    runtime.opener = Mock()
    frames = [FrameRef(i, float(i), tmp_path / "does-not-exist.jpg") for i in range(9)]
    with pytest.raises(AnalysisError) as error:
        runtime.ask("synthetic", frames, SCHEMA, Budget(time.time() + 30))
    assert error.value.code == "model_input_budget"
    runtime.opener.open.assert_not_called()


def test_managed_transformers_command_uses_selected_environment(monkeypatch, tmp_path):
    assets(tmp_path)
    monkeypatch.setenv("FORTE_VLM_BACKEND", "transformers")
    monkeypatch.setenv("FORTE_TRANSFORMERS_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("FORTE_VLM_PYTHON", "isolated-python.exe")
    runtime = LocalRuntime(tmp_path, tmp_path)
    runtime.healthy = Mock(side_effect=[False, True])
    runtime._read_server_metadata = Mock()
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr("worker.runtime.subprocess.Popen", popen)
    runtime.start()
    command = popen.call_args.args[0]
    assert command[0] == "isolated-python.exe"
    assert Path(command[1]).name == "serve_vlm.py"
    assert command[command.index("--model-dir") + 1] == str(tmp_path)
    runtime.stop()


def test_runtime_model_id_is_bound_to_actual_assets_and_inference_limits(monkeypatch, tmp_path):
    monkeypatch.delenv("FORTE_ATTACH_MODEL", raising=False)
    monkeypatch.setenv("FORTE_VLM_BACKEND", "llama.cpp")
    for name, filename in [("FORTE_LLAMA_SERVER", "server.exe"), ("FORTE_MODEL_PATH", "model.gguf"),
                           ("FORTE_MMPROJ_PATH", "vision.gguf")]:
        path = tmp_path / filename
        path.write_bytes(b"synthetic-" + filename.encode())
        monkeypatch.setenv(name, str(path))

    def start():
        runtime = LocalRuntime(tmp_path, tmp_path)
        runtime.healthy = Mock(side_effect=[False, True])
        process = Mock()
        process.poll.return_value = None
        monkeypatch.setattr("worker.runtime.subprocess.Popen", Mock(return_value=process))
        runtime.start()
        metadata = runtime.metadata()
        runtime.stop()
        return metadata

    before = start()
    assert before["model_id"].startswith("sha256:") and len(before["asset_hashes"]) == 3
    assert start()["model_id"] == before["model_id"]
    (tmp_path / "vision.gguf").write_bytes(b"changed-projector")
    changed = start()
    assert changed["model_id"] != before["model_id"]
    monkeypatch.setenv("FORTE_VLM_MAX_PIXELS", "200704")
    assert start()["model_id"] != changed["model_id"]


def test_attached_unknown_server_is_explicitly_unattested(monkeypatch, tmp_path):
    monkeypatch.setenv("FORTE_ATTACH_MODEL", "1")
    monkeypatch.setenv("FORTE_VLM_BACKEND", "llama.cpp")
    runtime = LocalRuntime(tmp_path, tmp_path)
    runtime.healthy = Mock(return_value=True)
    runtime.start()
    assert runtime.metadata()["model_id"].startswith("unattested:")
    assert runtime.metadata()["strict_schema_decoding"] is None
