"""Unit tests for slopfinity.llm.pool.

Hermetic: no real network/endpoint probes. `get_env_pool_config` reads
os.environ at call time, so we drive it with monkeypatch.setenv. For
`probe_endpoint` / `get_pool_status` we patch `get_provider` so the
provider's `list_models` is a stub (pool calls it via asyncio.to_thread).

asyncio_mode=auto (pytest.ini) runs async tests without a marker.
"""
from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity.llm import pool as POOL  # noqa: E402


# ---------------------------------------------------------------------------
# get_env_pool_config — env-driven construction
# ---------------------------------------------------------------------------

_ENV_KEYS = [
    "SLOPFINITY_LLM_PRIMARY_URL",
    "SLOPFINITY_LLM_PRIMARY_MODEL",
    "SLOPFINITY_LLM_CPU_URL",
    "SLOPFINITY_LLM_CPU_MODEL",
    "SLOPFINITY_LLM_FAILOVER_URLS",
    "SLOPFINITY_LLM_FAILOVER_MODELS",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


def test_pool_config_defaults():
    cfg = POOL.get_env_pool_config()
    assert cfg["primary"]["url"] == "http://localhost:1234/v1"
    assert cfg["primary"]["model"] == ""
    assert cfg["cpu"]["url"] == "http://localhost:11434/v1"
    assert cfg["failovers"] == []


def test_pool_config_reads_primary_and_cpu(monkeypatch):
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_URL", "http://gpu:1234/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_MODEL", "qwen-big")
    monkeypatch.setenv("SLOPFINITY_LLM_CPU_URL", "http://cpu:11434/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_CPU_MODEL", "tiny")
    cfg = POOL.get_env_pool_config()
    assert cfg["primary"] == {"url": "http://gpu:1234/v1", "model": "qwen-big"}
    assert cfg["cpu"] == {"url": "http://cpu:11434/v1", "model": "tiny"}


def test_pool_config_failovers_zipped(monkeypatch):
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://a/v1, http://b/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_MODELS", "ma, mb")
    cfg = POOL.get_env_pool_config()
    assert cfg["failovers"] == [
        {"url": "http://a/v1", "model": "ma"},
        {"url": "http://b/v1", "model": "mb"},
    ]


def test_pool_config_pads_missing_models(monkeypatch):
    # 3 urls but only 1 model -> models padded with "".
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://a, http://b, http://c")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_MODELS", "only")
    cfg = POOL.get_env_pool_config()
    assert [f["url"] for f in cfg["failovers"]] == ["http://a", "http://b", "http://c"]
    assert [f["model"] for f in cfg["failovers"]] == ["only", "", ""]


def test_pool_config_strips_blank_failover_urls(monkeypatch):
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://a/v1, , ,http://b/v1,")
    cfg = POOL.get_env_pool_config()
    assert [f["url"] for f in cfg["failovers"]] == ["http://a/v1", "http://b/v1"]


def test_pool_config_no_models_env_means_empty_models(monkeypatch):
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://a/v1")
    cfg = POOL.get_env_pool_config()
    assert cfg["failovers"] == [{"url": "http://a/v1", "model": ""}]


# ---------------------------------------------------------------------------
# probe_endpoint — model auto-select + non-embedding preference
# ---------------------------------------------------------------------------

def _provider_returning(models):
    """A fake provider whose list_models returns the given [{"id":...}]."""
    p = mock.Mock()
    p.list_models.return_value = models
    return p


async def test_probe_endpoint_uses_default_model_when_set(monkeypatch):
    prov = _provider_returning([{"id": "a"}, {"id": "b"}])
    monkeypatch.setattr(POOL, "get_provider", lambda name="lmstudio": prov)
    out = await POOL.probe_endpoint("http://x/v1", "preferred")
    assert out["ok"] is True
    assert out["selected_model"] == "preferred"
    assert out["available_models"] == ["a", "b"]
    assert out["error"] is None


async def test_probe_endpoint_autoselects_first_non_embedding(monkeypatch):
    prov = _provider_returning([
        {"id": "nomic-embed-text"},
        {"id": "text-embedding-3"},
        {"id": "qwen2.5-7b"},
        {"id": "llama3"},
    ])
    monkeypatch.setattr(POOL, "get_provider", lambda name="lmstudio": prov)
    out = await POOL.probe_endpoint("http://x/v1", "")
    assert out["selected_model"] == "qwen2.5-7b"


async def test_probe_endpoint_falls_back_to_first_if_all_embedding(monkeypatch):
    prov = _provider_returning([{"id": "nomic-embed"}, {"id": "bge-embed"}])
    monkeypatch.setattr(POOL, "get_provider", lambda name="lmstudio": prov)
    out = await POOL.probe_endpoint("http://x/v1", "")
    assert out["selected_model"] == "nomic-embed"


async def test_probe_endpoint_empty_models(monkeypatch):
    prov = _provider_returning([])
    monkeypatch.setattr(POOL, "get_provider", lambda name="lmstudio": prov)
    out = await POOL.probe_endpoint("http://x/v1", "")
    assert out["ok"] is True
    assert out["available_models"] == []
    assert out["selected_model"] == ""


async def test_probe_endpoint_error_returns_not_ok(monkeypatch):
    prov = mock.Mock()
    prov.list_models.side_effect = ConnectionError("refused")
    monkeypatch.setattr(POOL, "get_provider", lambda name="lmstudio": prov)
    out = await POOL.probe_endpoint("http://dead/v1", "fallback-model")
    assert out["ok"] is False
    assert out["selected_model"] == "fallback-model"
    assert out["available_models"] == []
    assert "refused" in out["error"]


async def test_probe_endpoint_passes_provider_name(monkeypatch):
    seen = {}

    def fake_get_provider(name="lmstudio"):
        seen["name"] = name
        return _provider_returning([{"id": "m"}])

    monkeypatch.setattr(POOL, "get_provider", fake_get_provider)
    await POOL.probe_endpoint("http://x/v1", "", provider_name="ollama")
    assert seen["name"] == "ollama"


# ---------------------------------------------------------------------------
# get_pool_status — full pool probe (primary + cpu + failovers)
# ---------------------------------------------------------------------------

async def test_get_pool_status_shape(monkeypatch):
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_URL", "http://primary/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_CPU_URL", "http://cpu/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://f1/v1, http://f2/v1")

    async def fake_probe(url, default_model, provider_name="lmstudio", timeout=5):
        return {"url": url, "ok": True, "selected_model": "m",
                "available_models": ["m"], "error": None}

    monkeypatch.setattr(POOL, "probe_endpoint", fake_probe)
    status = await POOL.get_pool_status()
    assert status["primary"]["url"] == "http://primary/v1"
    assert status["cpu"]["url"] == "http://cpu/v1"
    assert [f["url"] for f in status["failovers"]] == ["http://f1/v1", "http://f2/v1"]


async def test_get_pool_status_no_failovers(monkeypatch):
    async def fake_probe(url, default_model, provider_name="lmstudio", timeout=5):
        return {"url": url, "ok": True, "selected_model": "m",
                "available_models": ["m"], "error": None}

    monkeypatch.setattr(POOL, "probe_endpoint", fake_probe)
    status = await POOL.get_pool_status()
    assert status["failovers"] == []
    assert "primary" in status and "cpu" in status


async def test_get_pool_status_failover_to_next_on_primary_error(monkeypatch):
    # Primary dead, cpu + one failover alive. Status surfaces ok flags so a
    # caller can fail over to the next live endpoint.
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://f1/v1")

    async def fake_probe(url, default_model, provider_name="lmstudio", timeout=5):
        ok = "primary" not in url and "1234" not in url
        return {"url": url, "ok": ok, "selected_model": "m" if ok else "",
                "available_models": ["m"] if ok else [], "error": None if ok else "down"}

    monkeypatch.setattr(POOL, "probe_endpoint", fake_probe)
    status = await POOL.get_pool_status()
    # default primary url contains :1234 -> marked down
    assert status["primary"]["ok"] is False
    assert status["cpu"]["ok"] is True
    assert status["failovers"][0]["ok"] is True

    # Simulate the failover selection a caller would do: first ok endpoint.
    ordered = [status["primary"], status["cpu"], *status["failovers"]]
    live = next(e for e in ordered if e["ok"])
    assert live["url"] == status["cpu"]["url"]


async def test_get_pool_status_cpu_uses_ollama_provider(monkeypatch):
    calls = []

    async def fake_probe(url, default_model, provider_name="lmstudio", timeout=5):
        calls.append((url, provider_name))
        return {"url": url, "ok": True, "selected_model": "m",
                "available_models": ["m"], "error": None}

    monkeypatch.setattr(POOL, "probe_endpoint", fake_probe)
    await POOL.get_pool_status()
    providers_by_default_url = dict(calls)
    # cpu endpoint probed with the ollama provider per pool.py wiring.
    assert providers_by_default_url["http://localhost:11434/v1"] == "ollama"
    # primary probed with the default (lmstudio) provider.
    assert providers_by_default_url["http://localhost:1234/v1"] == "lmstudio"


async def test_get_pool_status_dedup_of_duplicate_failover_urls(monkeypatch):
    # pool.py now de-duplicates the pool before probing so the same endpoint
    # is never probed twice. Two identical failover URLs collapse to one.
    # (This assertion was intentionally flipped from the original
    # no-dedup placeholder once dedup landed.)
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://dup/v1, http://dup/v1")

    async def fake_probe(url, default_model, provider_name="lmstudio", timeout=5):
        return {"url": url, "ok": True, "selected_model": "m",
                "available_models": ["m"], "error": None}

    monkeypatch.setattr(POOL, "probe_endpoint", fake_probe)
    status = await POOL.get_pool_status()
    assert len(status["failovers"]) == 1  # deduped on normalized url
