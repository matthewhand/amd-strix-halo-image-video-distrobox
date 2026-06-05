"""Opportunistic ollama model selection — slopfinity rides whatever model is
already loaded (via /api/ps) instead of forcing its configured one."""
import json

import slopfinity.llm as llm


class _Resp:
    def __init__(self, payload):
        self._p = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._p


def test_loaded_model_parsed_from_api_ps(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _Resp({"models": [{"name": "gemma4:31b"}]}))
    assert llm._ollama_loaded_model("http://127.0.0.1:11434/v1") == "gemma4:31b"


def test_loaded_model_strips_v1_for_native_api_ps(monkeypatch):
    seen = {}

    def _open(url, *a, **k):
        seen["url"] = url
        return _Resp({"models": [{"model": "qwen3:8b"}]})

    monkeypatch.setattr("urllib.request.urlopen", _open)
    assert llm._ollama_loaded_model("http://h:11434/v1") == "qwen3:8b"
    assert seen["url"] == "http://h:11434/api/ps"   # /v1 stripped, native path


def test_loaded_model_none_when_nothing_loaded(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _Resp({"models": []}))
    assert llm._ollama_loaded_model("http://h:11434/v1") is None


def test_loaded_model_none_when_unreachable(monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert llm._ollama_loaded_model("http://h:11434/v1") is None


def test_opportunistic_default_on_and_config_gated(monkeypatch):
    monkeypatch.setattr(llm, "_opportunistic_enabled", llm._opportunistic_enabled)
    from slopfinity import config as cfg
    cfg.save_config({"scheduler": {}})
    assert llm._opportunistic_enabled() is True               # default ON
    cfg.save_config({"scheduler": {"llm_opportunistic": False}})
    assert llm._opportunistic_enabled() is False              # opt-out honored
