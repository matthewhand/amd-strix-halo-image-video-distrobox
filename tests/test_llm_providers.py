"""Unit tests for slopfinity.llm.providers.

Hermetic: no real network. The providers use stdlib
`urllib.request.urlopen` (not httpx), so we patch the module-level
`_http_json` helper to capture requests and return canned shapes.

Conventions follow the rest of tests/: sys.path bootstrap, unittest.mock.
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

from slopfinity.llm import providers as P  # noqa: E402


# ---------------------------------------------------------------------------
# Registry / selection
# ---------------------------------------------------------------------------

def test_list_providers_contains_known_keys():
    names = P.list_providers()
    for expected in ("lmstudio", "ollama", "vllm", "llamacpp", "custom"):
        assert expected in names


def test_get_provider_known():
    assert P.get_provider("ollama").name == "ollama"
    assert P.get_provider("vllm").name == "vllm"
    assert P.get_provider("llamacpp").name == "llamacpp"
    assert P.get_provider("custom").name == "custom"


def test_get_provider_case_and_whitespace_insensitive():
    assert P.get_provider("  OLLAMA  ").name == "ollama"
    assert P.get_provider("LMStudio").name == "lmstudio"


def test_get_provider_unknown_falls_back_to_lmstudio():
    assert P.get_provider("totally-bogus").name == "lmstudio"
    assert P.get_provider("").name == "lmstudio"
    assert P.get_provider(None).name == "lmstudio"


# ---------------------------------------------------------------------------
# Model listing parsing — OpenAI-compat /models
# ---------------------------------------------------------------------------

def _patch_http(return_value=None, side_effect=None):
    """Patch providers._http_json. Returns the mock for assertions."""
    return mock.patch.object(
        P, "_http_json",
        side_effect=side_effect,
        return_value=return_value,
    )


def test_openai_list_models_data_shape():
    payload = {
        "object": "list",
        "data": [
            {"id": "qwen2.5", "object": "model"},
            {"id": "llama3", "object": "model"},
        ],
    }
    with _patch_http(return_value=payload):
        out = P.LMStudioProvider().list_models("http://localhost:1234/v1")
    assert [m["id"] for m in out] == ["qwen2.5", "llama3"]
    assert out[0]["raw"]["object"] == "model"


def test_openai_list_models_models_key_and_name_fallback():
    # Some servers use {"models": [...]} and {"name": ...} instead of "id".
    payload = {"models": [{"name": "alpha"}, {"id": "beta"}]}
    with _patch_http(return_value=payload):
        out = P.OpenAICompatProvider().list_models("http://x/v1")
    assert [m["id"] for m in out] == ["alpha", "beta"]


def test_openai_list_models_string_items():
    payload = {"data": ["m1", "m2"]}
    with _patch_http(return_value=payload):
        out = P.VLLMProvider().list_models("http://x/v1")
    assert [m["id"] for m in out] == ["m1", "m2"]


def test_openai_list_models_drops_empty_ids():
    payload = {"data": [{"id": ""}, {"foo": "bar"}, {"id": "keep"}]}
    with _patch_http(return_value=payload):
        out = P.LMStudioProvider().list_models("http://x/v1")
    assert [m["id"] for m in out] == ["keep"]


def test_openai_list_models_builds_models_url():
    payload = {"data": []}
    with _patch_http(return_value=payload) as m:
        P.LMStudioProvider().list_models("http://localhost:1234/v1/")
    method, url = m.call_args.args[0], m.call_args.args[1]
    assert method == "GET"
    assert url == "http://localhost:1234/v1/models"


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------

def test_auth_headers_with_key():
    h = P._auth_headers("secret", {"X-Test": "1"})
    assert h["Authorization"] == "Bearer secret"
    assert h["X-Test"] == "1"


def test_auth_headers_without_key():
    assert "Authorization" not in P._auth_headers(None)
    assert P._auth_headers(None, {"A": "b"}) == {"A": "b"}


def test_list_models_passes_api_key_header():
    with _patch_http(return_value={"data": []}) as m:
        P.LMStudioProvider().list_models("http://x/v1", api_key="k")
    headers = m.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer k"


# ---------------------------------------------------------------------------
# Chat — content parsing + reasoning_content fallback
# ---------------------------------------------------------------------------

def _chat_payload(content=None, reasoning=None):
    msg = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg}]}


def test_chat_returns_content():
    with _patch_http(return_value=_chat_payload(content="  hi there  ")):
        out = P.LMStudioProvider().chat(
            "http://x/v1", "m", [{"role": "user", "content": "q"}], 0.2)
    assert out == "hi there"


def test_chat_reasoning_fallback_when_content_empty():
    with _patch_http(return_value=_chat_payload(content="", reasoning="  the cot text ")):
        out = P.LMStudioProvider().chat(
            "http://x/v1", "m", [{"role": "user", "content": "q"}], 0.0)
    assert out == "the cot text"


def test_chat_reasoning_fallback_when_content_missing():
    with _patch_http(return_value=_chat_payload(reasoning="cot only")):
        out = P.LMStudioProvider().chat(
            "http://x/v1", "m", [{"role": "user", "content": "q"}], 0.0)
    assert out == "cot only"


def test_chat_malformed_response_raises():
    with _patch_http(return_value={"unexpected": "shape"}):
        with pytest.raises(RuntimeError, match="Malformed chat response"):
            P.LMStudioProvider().chat(
                "http://x/v1", "m", [{"role": "user", "content": "q"}], 0.0)


def test_chat_builds_payload_and_url():
    with _patch_http(return_value=_chat_payload(content="ok")) as m:
        P.LMStudioProvider().chat(
            "http://localhost:8080/v1/", "my-model",
            [{"role": "user", "content": "hello"}], 0.7, max_tokens=42)
    assert m.call_args.args[1] == "http://localhost:8080/v1/chat/completions"
    body = m.call_args.kwargs["body"]
    assert body["model"] == "my-model"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 42
    assert body["messages"][0]["content"] == "hello"


def test_chat_omits_max_tokens_when_none():
    with _patch_http(return_value=_chat_payload(content="ok")) as m:
        P.LMStudioProvider().chat(
            "http://x/v1", "m", [{"role": "user", "content": "q"}], 0.0)
    assert "max_tokens" not in m.call_args.kwargs["body"]


# ---------------------------------------------------------------------------
# Structured-output passthrough (response_format)
# ---------------------------------------------------------------------------

def test_chat_response_format_passthrough():
    rf = {"type": "json_schema",
          "json_schema": {"name": "chips", "schema": {"type": "array"}, "strict": True}}
    with _patch_http(return_value=_chat_payload(content="[]")) as m:
        P.LMStudioProvider().chat(
            "http://x/v1", "m", [{"role": "user", "content": "q"}], 0.0,
            response_format=rf)
    assert m.call_args.kwargs["body"]["response_format"] == rf


def test_chat_no_response_format_key_when_none():
    with _patch_http(return_value=_chat_payload(content="ok")) as m:
        P.LMStudioProvider().chat(
            "http://x/v1", "m", [{"role": "user", "content": "q"}], 0.0)
    assert "response_format" not in m.call_args.kwargs["body"]


def test_chat_extra_headers_merged_with_auth():
    with _patch_http(return_value=_chat_payload(content="ok")) as m:
        P.LMStudioProvider().chat(
            "http://x/v1", "m", [{"role": "user", "content": "q"}], 0.0,
            api_key="k", extra_headers={"X-Trace": "abc"})
    headers = m.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer k"
    assert headers["X-Trace"] == "abc"


# ---------------------------------------------------------------------------
# Ollama provider — compat-preferred vs native fallback
# ---------------------------------------------------------------------------

def test_ollama_compat_base_url_normalization():
    o = P.OllamaProvider()
    assert o._compat_base("http://localhost:11434") == "http://localhost:11434/v1"
    assert o._compat_base("http://localhost:11434/") == "http://localhost:11434/v1"
    assert o._compat_base("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_ollama_native_base_url_normalization():
    o = P.OllamaProvider()
    assert o._native_base("http://localhost:11434/v1") == "http://localhost:11434"
    assert o._native_base("http://localhost:11434") == "http://localhost:11434"


def test_ollama_list_models_prefers_openai_compat():
    o = P.OllamaProvider()

    def fake_http(method, url, **kw):
        # First call: the _openai_ok probe to /v1/models -> succeeds.
        # Second call: the real list to /v1/models -> data shape.
        assert url.endswith("/v1/models")
        return {"data": [{"id": "compat-model"}]}

    with mock.patch.object(P, "_http_json", side_effect=fake_http):
        out = o.list_models("http://localhost:11434")
    assert [m["id"] for m in out] == ["compat-model"]


def test_ollama_list_models_native_fallback_on_compat_failure():
    o = P.OllamaProvider()
    calls = []

    def fake_http(method, url, **kw):
        calls.append(url)
        if url.endswith("/v1/models"):
            raise RuntimeError("no compat endpoint")
        # native /api/tags shape
        return {"models": [{"name": "llama3:8b"}, {"name": "nomic-embed"}]}

    with mock.patch.object(P, "_http_json", side_effect=fake_http):
        out = o.list_models("http://localhost:11434")
    assert [m["id"] for m in out] == ["llama3:8b", "nomic-embed"]
    # native tags endpoint was hit
    assert any(u.endswith("/api/tags") for u in calls)


def test_ollama_native_tags_drops_unnamed():
    o = P.OllamaProvider()

    def fake_http(method, url, **kw):
        if url.endswith("/v1/models"):
            raise RuntimeError("down")
        return {"models": [{"name": "a"}, {"size": 1}, {"name": ""}]}

    with mock.patch.object(P, "_http_json", side_effect=fake_http):
        out = o.list_models("http://x")
    assert [m["id"] for m in out] == ["a"]


def test_ollama_chat_prefers_compat():
    o = P.OllamaProvider()

    def fake_http(method, url, **kw):
        if method == "GET" and url.endswith("/v1/models"):
            return {"data": []}  # _openai_ok probe
        assert url.endswith("/v1/chat/completions")
        return {"choices": [{"message": {"content": "compat reply"}}]}

    with mock.patch.object(P, "_http_json", side_effect=fake_http):
        out = o.chat("http://localhost:11434", "m",
                     [{"role": "user", "content": "q"}], 0.0)
    assert out == "compat reply"


def test_ollama_chat_native_fallback_and_json_format():
    o = P.OllamaProvider()
    captured = {}

    def fake_http(method, url, **kw):
        if method == "GET" and url.endswith("/v1/models"):
            raise RuntimeError("no compat")
        assert url.endswith("/api/chat")
        captured["body"] = kw.get("body")
        return {"message": {"content": "  native reply  "}}

    rf = {"type": "json_object"}
    with mock.patch.object(P, "_http_json", side_effect=fake_http):
        out = o.chat("http://localhost:11434", "m",
                     [{"role": "user", "content": "q"}], 0.3,
                     response_format=rf)
    assert out == "native reply"
    assert captured["body"]["format"] == "json"
    assert captured["body"]["options"]["temperature"] == 0.3
    assert captured["body"]["stream"] is False


def test_ollama_chat_native_no_format_when_not_json():
    o = P.OllamaProvider()
    captured = {}

    def fake_http(method, url, **kw):
        if method == "GET" and url.endswith("/v1/models"):
            raise RuntimeError("no compat")
        captured["body"] = kw.get("body")
        return {"message": {"content": "x"}}

    with mock.patch.object(P, "_http_json", side_effect=fake_http):
        o.chat("http://localhost:11434", "m",
               [{"role": "user", "content": "q"}], 0.0)
    assert "format" not in captured["body"]


def test_ollama_chat_native_empty_message_returns_empty_string():
    o = P.OllamaProvider()

    def fake_http(method, url, **kw):
        if method == "GET" and url.endswith("/v1/models"):
            raise RuntimeError("no compat")
        return {}  # no "message" key

    with mock.patch.object(P, "_http_json", side_effect=fake_http):
        out = o.chat("http://x", "m", [{"role": "user", "content": "q"}], 0.0)
    assert out == ""


# ---------------------------------------------------------------------------
# _http_json helper — verify request shaping without real network
# ---------------------------------------------------------------------------

def test_http_json_get_no_body(monkeypatch):
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=None):
        captured["method"] = req.get_method()
        captured["data"] = req.data
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(P._OPENER, "open", fake_urlopen)
    out = P._http_json("GET", "http://x/v1/models", timeout=3)
    assert out == {"ok": True}
    assert captured["method"] == "GET"
    assert captured["data"] is None
    assert captured["timeout"] == 3


def test_http_json_post_encodes_body(monkeypatch):
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""  # empty -> {} per helper contract

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        captured["ct"] = req.get_header("Content-type")
        return FakeResp()

    monkeypatch.setattr(P._OPENER, "open", fake_urlopen)
    out = P._http_json("POST", "http://x", body={"a": 1})
    assert out == {}
    assert json.loads(captured["data"].decode()) == {"a": 1}
    assert captured["ct"] == "application/json"
