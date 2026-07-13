"""Unit tests for slopfinity.llm.probe.

Hermetic: no real port scan, no real network. We patch
`urllib.request.urlopen` (probe uses it directly) and, for `discover`,
patch `_probe_one` so no sockets are opened.

asyncio_mode=auto (see pytest.ini) lets async test funcs run without a
marker.
"""
from __future__ import annotations

import json
import os
import sys
from unittest import mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity.llm import probe as PR  # noqa: E402


class _FakeResp:
    def __init__(self, raw: bytes):
        self._raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._raw


def _urlopen_router(mapping: dict, default_exc=ConnectionError("refused")):
    """Build a fake urlopen that returns canned bytes per-URL, else raises.

    mapping: {url_substring: bytes}
    """
    def fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for key, raw in mapping.items():
            if key in url:
                return _FakeResp(raw)
        raise default_exc
    return fake


# ---------------------------------------------------------------------------
# _probe_one — OpenAI-compat servers
# ---------------------------------------------------------------------------

def test_probe_one_openai_compat_models(monkeypatch):
    raw = json.dumps({"data": [{"id": "a"}, {"id": "b"}]}).encode()
    monkeypatch.setattr(
        PR.urllib.request, "urlopen",
        _urlopen_router({"/models": raw}))
    out = PR._probe_one(1234, "lmstudio", "http://localhost:1234/v1")
    assert out == {
        "port": 1234,
        "provider": "lmstudio",
        "base_url": "http://localhost:1234/v1",
        "model_count": 2,
        "flavor": "openai-compat",
    }


def test_probe_one_openai_compat_models_key(monkeypatch):
    raw = json.dumps({"models": [{"id": "x"}]}).encode()
    monkeypatch.setattr(
        PR.urllib.request, "urlopen",
        _urlopen_router({"/models": raw}))
    out = PR._probe_one(8000, "vllm", "http://localhost:8000/v1")
    assert out["provider"] == "vllm"
    assert out["model_count"] == 1


def test_probe_one_returns_none_when_dead(monkeypatch):
    monkeypatch.setattr(
        PR.urllib.request, "urlopen",
        _urlopen_router({}))  # everything refused
    assert PR._probe_one(1234, "lmstudio", "http://localhost:1234/v1") is None


def test_probe_one_handles_non_json_body(monkeypatch):
    monkeypatch.setattr(
        PR.urllib.request, "urlopen",
        _urlopen_router({"/models": b"not json at all"}))
    out = PR._probe_one(8080, "llamacpp", "http://localhost:8080/v1")
    # non-JSON -> data={} -> model_count 0 but still "alive"
    assert out is not None
    assert out["model_count"] == 0
    assert out["flavor"] == "openai-compat"


# ---------------------------------------------------------------------------
# _probe_one — Ollama: compat preferred, native fallback
# ---------------------------------------------------------------------------

def test_probe_one_ollama_prefers_compat(monkeypatch):
    compat = json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]}).encode()
    monkeypatch.setattr(
        PR.urllib.request, "urlopen",
        _urlopen_router({"/v1/models": compat}))
    out = PR._probe_one(11434, "ollama", "http://localhost:11434")
    assert out["provider"] == "ollama"
    assert out["flavor"] == "openai-compat"
    assert out["base_url"] == "http://localhost:11434/v1"
    assert out["model_count"] == 2


def test_probe_one_ollama_native_fallback(monkeypatch):
    native = json.dumps({"models": [{"name": "llama3"}]}).encode()

    def fake(req, timeout=None):
        url = req.full_url
        if "/v1/models" in url:
            raise ConnectionError("no compat")
        if "/api/tags" in url:
            return _FakeResp(native)
        raise ConnectionError("refused")

    monkeypatch.setattr(PR.urllib.request, "urlopen", fake)
    out = PR._probe_one(11434, "ollama", "http://localhost:11434")
    assert out["flavor"] == "native"
    assert out["base_url"] == "http://localhost:11434"
    assert out["model_count"] == 1


# ---------------------------------------------------------------------------
# discover — async fan-out over CANDIDATES (mock _probe_one, no sockets)
# ---------------------------------------------------------------------------

async def test_discover_filters_to_alive_dicts(monkeypatch):
    def fake_probe_one(port, provider, base_url, timeout=1.0):
        # Ollama is first in CANDIDATES; return it as the only alive host.
        if port == 11434:
            return {"port": 11434, "provider": "ollama",
                    "base_url": base_url, "model_count": 1,
                    "flavor": "openai-compat"}
        return None

    monkeypatch.setattr(PR, "_probe_one", fake_probe_one)
    found = await PR.discover(timeout=0.01)
    assert len(found) == 1
    assert found[0]["port"] == 11434


async def test_discover_empty_when_nothing_alive(monkeypatch):
    monkeypatch.setattr(PR, "_probe_one", lambda *a, **k: None)
    assert await PR.discover(timeout=0.01) == []


async def test_discover_swallows_probe_exceptions(monkeypatch):
    # _probe_one raising should not blow up discover (return_exceptions=True
    # in gather, then filtered to dicts only).
    def boom(port, provider, base_url, timeout=1.0):
        if port == 8000:
            return {"port": 8000, "provider": "vllm", "base_url": base_url,
                    "model_count": 0, "flavor": "openai-compat"}
        raise RuntimeError("scan error")

    monkeypatch.setattr(PR, "_probe_one", boom)
    found = await PR.discover(timeout=0.01)
    assert [f["port"] for f in found] == [8000]


def test_candidates_cover_known_ports():
    ports = {c[0] for c in PR.CANDIDATES}
    assert {1234, 11434, 8080, 8000} <= ports


# ---------------------------------------------------------------------------
# ping — latency parsing + error capture (mock the provider, no network)
# ---------------------------------------------------------------------------

def test_ping_ok_reports_latency_and_reply(monkeypatch):
    fake_provider = mock.Mock()
    fake_provider.chat.return_value = "pong reply text"

    # ping imports get_provider from .providers inside the function body.
    monkeypatch.setattr(
        "slopfinity.llm.providers.get_provider",
        lambda name: fake_provider)

    out = PR.ping("http://x/v1", "lmstudio", "model-a", api_key="k", timeout=4)
    assert out["ok"] is True
    assert isinstance(out["latency_ms"], int)
    assert out["latency_ms"] >= 0
    assert out["reply"] == "pong reply text"
    # ping uses a 1-token request
    _, kwargs = fake_provider.chat.call_args
    assert kwargs["max_tokens"] == 1
    assert kwargs["model_id"] == "model-a"
    assert kwargs["temperature"] == 0.0


def test_ping_truncates_long_reply(monkeypatch):
    fake_provider = mock.Mock()
    fake_provider.chat.return_value = "z" * 500
    monkeypatch.setattr(
        "slopfinity.llm.providers.get_provider",
        lambda name: fake_provider)
    out = PR.ping("http://x/v1", "lmstudio", "m")
    assert len(out["reply"]) == 80


def test_ping_error_captured(monkeypatch):
    fake_provider = mock.Mock()
    fake_provider.chat.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(
        "slopfinity.llm.providers.get_provider",
        lambda name: fake_provider)
    out = PR.ping("http://x/v1", "lmstudio", "m")
    assert out["ok"] is False
    assert "connection refused" in out["error"]
    assert isinstance(out["latency_ms"], int)


def test_ping_empty_reply_handled(monkeypatch):
    fake_provider = mock.Mock()
    fake_provider.chat.return_value = None
    monkeypatch.setattr(
        "slopfinity.llm.providers.get_provider",
        lambda name: fake_provider)
    out = PR.ping("http://x/v1", "lmstudio", "m")
    assert out["ok"] is True
    assert out["reply"] == ""
