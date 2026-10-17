"""Offline boundaries, tested without sockets, model assets, or private footage."""
from email.message import Message
from html.parser import HTMLParser
import io
from pathlib import Path
import re
import time
import urllib.request
from urllib.response import addinfourl

import pytest

from worker.runtime import LocalRuntime
from worker.video import AnalysisError, Budget


ROOT = Path(__file__).resolve().parents[1]


class ResourceReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "img", "video", "source", "iframe", "audio", "embed"}:
            self.references.extend(value for name, value in attrs.items() if name in {"src", "srcset"})
        if tag == "link":
            self.references.append(attrs.get("href", ""))
        if tag == "form" and "action" in attrs:
            self.references.append(attrs["action"])


def test_web_assets_do_not_depend_on_external_hosts():
    parser = ResourceReferences()
    parser.feed((ROOT / "web" / "index.html").read_text(encoding="utf-8"))
    assert parser.references
    for reference in parser.references:
        assert reference.startswith("/") and not reference.startswith("//"), reference
        assert "\\" not in reference, reference
    css = (ROOT / "web" / "style.css").read_text(encoding="utf-8")
    assert not re.search(r"@import\b", css, re.I), "CSS imports can introduce remote dependencies"
    for reference in re.findall(r"url\(\s*['\"]?([^'\")\s]+)", css, re.I):
        assert not re.match(r"(?:[a-z]+:)?//", reference, re.I), reference
    javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    # Official rule links are user-clicked citations; network API calls and dynamically
    # loaded code must never contain a literal external destination.
    assert not re.search(r"(?:fetch|request|json|import|importScripts|WebSocket|EventSource)\s*\(\s*['\"`](?:https?:)?//", javascript)
    assert not re.search(r"(?:src|srcset)\s*=\s*['\"`](?:https?:)?//", javascript)
    assert not re.search(r"\b(?:localStorage|sessionStorage)\b", javascript), "API keys must not be persisted"


def fake_http_response(request, status=200, location=None):
    headers = Message()
    if location:
        headers["Location"] = location
    response = addinfourl(io.BytesIO(b'{"status":"ok"}'), headers, request.full_url, status)
    response.msg = "Synthetic response"
    return response


def test_local_inference_ignores_environment_proxies(monkeypatch, tmp_path):
    monkeypatch.setenv("FORTE_LLM_URL", "http://127.0.0.1:18080")
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"http": "http://proxy.invalid:3128"})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    destinations = []

    def open_without_network(handler, http_class, request, **kwargs):
        destinations.append((request.host, request.selector))
        return fake_http_response(request)

    monkeypatch.setattr(urllib.request.HTTPHandler, "do_open", open_without_network)
    runtime = LocalRuntime(tmp_path, tmp_path)
    assert runtime.healthy()
    assert destinations == [("127.0.0.1:18080", "/health")]


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_local_inference_never_follows_remote_redirects(monkeypatch, tmp_path, status):
    monkeypatch.setenv("FORTE_LLM_URL", "http://127.0.0.1:18080")
    destinations = []

    def redirect_without_network(handler, http_class, request, **kwargs):
        destinations.append(request.full_url)
        assert request.host == "127.0.0.1:18080", "A local model response attempted a remote request"
        return fake_http_response(request, status, "http://remote.invalid/collect")

    monkeypatch.setattr(urllib.request.HTTPHandler, "do_open", redirect_without_network)
    runtime = LocalRuntime(tmp_path, tmp_path)
    assert not runtime.healthy()
    with pytest.raises(AnalysisError) as error:
        runtime.ask("Synthetic test only", [], {"type": "object"}, Budget(time.time() + 30))
    assert error.value.code == "model_http_error"
    assert destinations == ["http://127.0.0.1:18080/health", "http://127.0.0.1:18080/v1/chat/completions"]
