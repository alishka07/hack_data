"""Runtime ownership/cancellation and structured output checks without model/media."""
import io
import json
import threading
import time
import urllib.error
from unittest.mock import Mock

import pytest

from worker.runtime import LocalRuntime, _validate_json_schema
from worker.video import AnalysisError, Budget


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:18080", "http://0.0.0.0:18080", "http://example.com",
    "http://127.0.0.1:18080?x=1", "http://127.0.0.1:18080#fragment",
    "http://user:pass@127.0.0.1:18080", "http://127.0.0.1:0",
])
def test_reject_nonplain_or_nonloopback_runtime_url(monkeypatch, tmp_path, url):
    monkeypatch.setenv("FORTE_LLM_URL", url)
    with pytest.raises(ValueError):
        LocalRuntime(tmp_path, tmp_path / "data")


def _assets(monkeypatch, tmp_path):
    for env, name in [("FORTE_LLAMA_SERVER", "llama-server.exe"),
                      ("FORTE_MODEL_PATH", "language.gguf"), ("FORTE_MMPROJ_PATH", "vision.gguf")]:
        path = tmp_path / name
        path.write_bytes(b"synthetic")
        monkeypatch.setenv(env, str(path))
    monkeypatch.delenv("FORTE_ATTACH_MODEL", raising=False)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_managed_binding_matches_loopback_url(monkeypatch, tmp_path, host):
    _assets(monkeypatch, tmp_path)
    authority = f"[{host}]" if ":" in host else host
    monkeypatch.setenv("FORTE_LLM_URL", f"http://{authority}:18080")
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    runtime.healthy = Mock(side_effect=[False, True])
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr("worker.runtime.subprocess.Popen", popen)
    runtime.start()
    command = popen.call_args.args[0]
    assert command[command.index("--host") + 1] == host
    runtime.stop()
    process.terminate.assert_called_once()


def test_startup_cancellation_preserves_owned_process_for_cleanup(monkeypatch, tmp_path):
    _assets(monkeypatch, tmp_path)
    monkeypatch.setenv("FORTE_LLM_URL", "http://127.0.0.1:18080")
    stop = threading.Event()
    runtime = LocalRuntime(tmp_path, tmp_path / "data", stop)
    runtime.healthy = Mock(return_value=False)
    process = Mock()
    process.poll.return_value = None

    def spawn(*args, **kwargs):
        stop.set()
        return process

    monkeypatch.setattr("worker.runtime.subprocess.Popen", spawn)
    with pytest.raises(AnalysisError) as error:
        runtime.start()
    assert error.value.code == "worker_stopped"
    assert runtime.process is process
    runtime.stop()
    process.terminate.assert_called_once()
    assert runtime.process is None
    assert runtime.log is None


def test_owned_cpu_profile_never_offloads_vision_projector(monkeypatch, tmp_path):
    _assets(monkeypatch, tmp_path)
    monkeypatch.setenv("FORTE_GPU_LAYERS", "0")
    monkeypatch.setenv("FORTE_VLM_CPU_PROJECTOR", "1")
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    runtime.healthy = Mock(side_effect=[False, True])
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr("worker.runtime.subprocess.Popen", popen)
    runtime.start()
    command = popen.call_args.args[0]
    assert "--no-mmproj-offload" in command
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert runtime.metadata()["projector_cpu"] is True
    runtime.stop()


def test_cancel_does_not_terminate_attached_server(monkeypatch, tmp_path):
    monkeypatch.setenv("FORTE_ATTACH_MODEL", "1")
    monkeypatch.setenv("FORTE_LLM_URL", "http://127.0.0.1:18080")
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    runtime.healthy = Mock(return_value=True)
    runtime.start()
    runtime.cancel()
    runtime.stop()
    assert runtime.process is None


SCHEMA = {"type": "object", "required": ["decision", "indices", "note"], "additionalProperties": False,
          "properties": {
              "decision": {"type": "string", "enum": ["supported", "uncertain"]},
              "indices": {"type": "array", "maxItems": 2,
                          "items": {"type": "integer", "minimum": 0, "maximum": 3}},
              "note": {"type": "string", "maxLength": 4}}}


@pytest.mark.parametrize("value", [
    {}, {"decision": "supported", "indices": [], "note": "ok", "extra": 1},
    {"decision": "other", "indices": [], "note": "ok"},
    {"decision": "supported", "indices": [True], "note": "ok"},
    {"decision": "supported", "indices": [4], "note": "ok"},
    {"decision": "supported", "indices": [-1], "note": "ok"},
    {"decision": "supported", "indices": [0, 1, 2], "note": "ok"},
    {"decision": "supported", "indices": [], "note": "too long"},
])
def test_schema_violations_are_model_failures(value):
    with pytest.raises(AnalysisError) as error:
        _validate_json_schema(value, SCHEMA)
    assert error.value.code == "model_schema_invalid"


def test_ask_rejects_empty_object_instead_of_clean_verdict(monkeypatch, tmp_path):
    monkeypatch.setenv("FORTE_LLM_URL", "http://127.0.0.1:18080")
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    reply = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
    runtime.opener = Mock()
    runtime.opener.open.return_value = io.BytesIO(json.dumps(reply).encode())
    with pytest.raises(AnalysisError) as error:
        runtime.ask("synthetic request", [], SCHEMA, Budget(time.time() + 30))
    assert error.value.code == "model_schema_invalid"


def test_schema_valid_output_passes():
    _validate_json_schema({"decision": "uncertain", "indices": [0, 3], "note": "ok"}, SCHEMA)


def _model_reply(value):
    return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop",
                                              "message": {"content": json.dumps(value)}}]}).encode())


@pytest.mark.parametrize("seconds,expected", [(100, "model_timeout"), (20, "deadline_exceeded")])
@pytest.mark.parametrize("wrapped", [False, True])
def test_request_timeout_distinguishes_deadline(monkeypatch, tmp_path, seconds, expected, wrapped):
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    error = TimeoutError("private transport detail")
    runtime.opener = Mock()
    runtime.opener.open.side_effect = urllib.error.URLError(error) if wrapped else error
    with pytest.raises(AnalysisError) as caught:
        runtime.ask("private prompt", [], SCHEMA, Budget(time.time() + seconds))
    assert caught.value.code == expected
    assert "private" not in str(caught.value)
    assert runtime.opener.open.call_count == 1
    metrics = runtime.metadata()["requests"]
    assert metrics["failures"] == 1
    assert metrics["last_request"]["error_code"] == expected
    assert "private" not in json.dumps(metrics)


@pytest.mark.parametrize("first", ["bad_json", "bad_schema", "http_502", "http_503"])
def test_transient_or_invalid_output_has_one_bounded_retry(monkeypatch, tmp_path, first):
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    runtime.opener = Mock()
    invalid = (io.BytesIO(b"not JSON") if first == "bad_json" else
               _model_reply({}) if first == "bad_schema" else
               urllib.error.HTTPError(runtime.url, int(first[-3:]), "private body", {}, io.BytesIO(b"private body")))
    valid = {"decision": "uncertain", "indices": [], "note": "ok"}
    runtime.opener.open.side_effect = [invalid, _model_reply(valid)]
    assert runtime.ask("private prompt", [], SCHEMA, Budget(time.time() + 100)) == valid
    assert runtime.opener.open.call_count == 2
    metrics = runtime.metadata()["requests"]
    assert metrics["retries"] == 1 and metrics["failures"] == 0
    assert metrics["last_request"]["attempts"] == 2
    assert metrics["last_request"]["retry_error_code"] is not None
    assert "private" not in json.dumps(metrics)


@pytest.mark.parametrize("seconds", [20, 49])
def test_no_retry_near_deadline(tmp_path, seconds):
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    runtime.opener = Mock()
    runtime.opener.open.side_effect = [io.BytesIO(b"invalid"), _model_reply({})]
    with pytest.raises(AnalysisError) as caught:
        runtime.ask("private prompt", [], SCHEMA, Budget(time.time() + seconds))
    assert caught.value.code == "model_output_invalid"
    assert runtime.opener.open.call_count == 1


def test_invalid_retry_is_bounded_even_with_large_budget(tmp_path):
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    runtime.opener = Mock()
    runtime.opener.open.side_effect = [io.BytesIO(b"bad"), io.BytesIO(b"bad"), _model_reply({})]
    with pytest.raises(AnalysisError) as caught:
        runtime.ask("prompt", [], SCHEMA, Budget(time.time() + 300))
    assert caught.value.code == "model_output_invalid"
    assert runtime.opener.open.call_count == 2


@pytest.mark.parametrize("failure,expected", [
    (lambda url: urllib.error.HTTPError(url, 400, "private detail", {}, io.BytesIO(b"private body")), "model_http_error"),
    (lambda url: urllib.error.URLError("private detail"), "model_transport_error"),
    (lambda url: ConnectionResetError("private detail"), "model_transport_error"),
])
def test_transport_and_http_errors_are_distinct_and_private(tmp_path, failure, expected):
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    runtime.opener = Mock()
    runtime.opener.open.side_effect = failure(runtime.url)
    with pytest.raises(AnalysisError) as caught:
        runtime.ask("private prompt", [], SCHEMA, Budget(time.time() + 100))
    assert caught.value.code == expected
    assert "private" not in str(caught.value)
    assert runtime.opener.open.call_count == 1


def test_truncated_output_is_not_retried(tmp_path):
    runtime = LocalRuntime(tmp_path, tmp_path / "data")
    runtime.opener = Mock()
    runtime.opener.open.return_value = io.BytesIO(json.dumps({"choices": [{"finish_reason": "length"}]}).encode())
    with pytest.raises(AnalysisError) as caught:
        runtime.ask("prompt", [], SCHEMA, Budget(time.time() + 100))
    assert caught.value.code == "model_output_truncated"
    assert runtime.opener.open.call_count == 1
