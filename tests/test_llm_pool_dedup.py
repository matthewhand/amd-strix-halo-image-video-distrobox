"""Unit tests for LLM pool endpoint de-duplication.

Separate from tests/test_llm_pool.py (#158) so the two suites do not
collide. Covers:

  * the _dedup_endpoints / _normalize_url helpers in slopfinity.llm.pool
  * get_pool_status dropping duplicate failover URLs before probing
  * the failover-on-error chain in slopfinity.llm.__init__ honouring the
    deduped, priority-ordered pool and advancing past a failing endpoint

Hermetic: no real network. get_env_pool_config reads os.environ at call
time so we drive it with monkeypatch.setenv; providers/probes are stubbed.

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
import slopfinity.llm as LLM  # noqa: E402


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


# ---------------------------------------------------------------------------
# _normalize_url
# ---------------------------------------------------------------------------

def test_normalize_url_strips_trailing_slash_and_lowercases():
    assert POOL._normalize_url("http://Host:1234/v1/") == "http://host:1234/v1"
    assert POOL._normalize_url("  http://host:1234/v1  ") == "http://host:1234/v1"
    assert POOL._normalize_url("") == ""
    assert POOL._normalize_url(None) == ""


# ---------------------------------------------------------------------------
# _dedup_endpoints — order/priority preserved, dedup on (url, model)
# ---------------------------------------------------------------------------

def test_dedup_drops_duplicate_urls_preserving_order():
    eps = [
        {"url": "http://primary/v1", "model": "p"},
        {"url": "http://cpu/v1", "model": "c"},
        {"url": "http://f1/v1", "model": "f"},
        {"url": "http://f1/v1", "model": "f"},  # exact dup -> dropped
    ]
    out = POOL._dedup_endpoints(eps)
    assert [e["url"] for e in out] == [
        "http://primary/v1", "http://cpu/v1", "http://f1/v1",
    ]


def test_dedup_is_case_and_trailing_slash_insensitive():
    eps = [
        {"url": "http://Host:1234/v1", "model": "m"},
        {"url": "http://host:1234/v1/", "model": "m"},  # same after normalize
    ]
    out = POOL._dedup_endpoints(eps)
    assert len(out) == 1
    # First occurrence (and its original casing) is the one kept.
    assert out[0]["url"] == "http://Host:1234/v1"


def test_dedup_keeps_same_url_with_different_model():
    eps = [
        {"url": "http://host/v1", "model": "a"},
        {"url": "http://host/v1", "model": "b"},  # distinct (url, model)
    ]
    out = POOL._dedup_endpoints(eps)
    assert [e["model"] for e in out] == ["a", "b"]


def test_dedup_skips_blank_urls():
    eps = [
        {"url": "", "model": "x"},
        {"url": "   ", "model": "y"},
        {"url": "http://real/v1", "model": "z"},
    ]
    out = POOL._dedup_endpoints(eps)
    assert [e["url"] for e in out] == ["http://real/v1"]


def test_dedup_priority_primary_then_cpu_then_failovers():
    # A failover that duplicates the primary URL is dropped; the primary
    # (higher priority, earlier) is the survivor.
    eps = [
        {"url": "http://shared/v1", "model": "m"},   # primary
        {"url": "http://cpu/v1", "model": "c"},      # cpu
        {"url": "http://shared/v1", "model": "m"},   # failover dup of primary
        {"url": "http://f2/v1", "model": "f2"},      # unique failover
    ]
    out = POOL._dedup_endpoints(eps)
    assert [e["url"] for e in out] == [
        "http://shared/v1", "http://cpu/v1", "http://f2/v1",
    ]


# ---------------------------------------------------------------------------
# get_pool_status — duplicate failover URLs collapsed before probing
# ---------------------------------------------------------------------------

async def test_get_pool_status_dedups_duplicate_failovers(monkeypatch):
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS",
                       "http://dup/v1, http://DUP/v1/, http://other/v1")

    probed = []

    async def fake_probe(url, default_model, provider_name="lmstudio", timeout=5):
        probed.append(url)
        return {"url": url, "ok": True, "selected_model": "m",
                "available_models": ["m"], "error": None}

    monkeypatch.setattr(POOL, "probe_endpoint", fake_probe)
    status = await POOL.get_pool_status()

    # http://dup/v1 and http://DUP/v1/ normalize equal -> one survives.
    assert [f["url"] for f in status["failovers"]] == [
        "http://dup/v1", "http://other/v1",
    ]
    # The duplicate failover was never probed.
    assert probed.count("http://DUP/v1/") == 0


async def test_get_pool_status_failover_matching_primary_is_dropped(monkeypatch):
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_URL", "http://shared/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://shared/v1/, http://f2/v1")

    async def fake_probe(url, default_model, provider_name="lmstudio", timeout=5):
        return {"url": url, "ok": True, "selected_model": "m",
                "available_models": ["m"], "error": None}

    monkeypatch.setattr(POOL, "probe_endpoint", fake_probe)
    status = await POOL.get_pool_status()

    # The failover duplicating the primary is dropped; primary slot kept.
    assert status["primary"]["url"] == "http://shared/v1"
    assert [f["url"] for f in status["failovers"]] == ["http://f2/v1"]


# ---------------------------------------------------------------------------
# lmstudio_call — failover-on-error walks the deduped pool in priority order
#
# This behaviour already existed in __init__.py before this change; these
# tests pin it down and confirm it operates over the deduped chain.
# ---------------------------------------------------------------------------

class _StubProvider:
    """Provider whose chat() succeeds only for whitelisted base_urls."""

    def __init__(self, ok_urls, calls):
        self._ok = {u.rstrip("/") for u in ok_urls}
        self._calls = calls

    def chat(self, base_url, model_id, messages, **kw):
        self._calls.append(base_url)
        if base_url in self._ok:
            return f"ok:{base_url}:{model_id}"
        raise ConnectionError(f"refused {base_url}")


def _patch_llm(monkeypatch, ok_urls, calls):
    """Wire __init__.py so it uses our stub provider and a quiet scheduler."""
    stub = _StubProvider(ok_urls, calls)
    monkeypatch.setattr(LLM, "get_provider", lambda name="lmstudio": stub)
    monkeypatch.setattr(LLM, "_load_llm_cfg", lambda: dict(LLM.DEFAULT_LLM_CONFIG))

    class _GPU:
        resident_gb = 0
        in_flight = []

    fake_sched = mock.Mock()
    fake_sched.GPU = _GPU()
    monkeypatch.setitem(sys.modules, "slopfinity.scheduler", fake_sched)


def test_lmstudio_call_advances_past_failing_endpoint(monkeypatch):
    # primary dead, first failover dead, second failover alive -> success.
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_URL", "http://primary/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_MODEL", "pm")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://f1/v1, http://f2/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_MODELS", "fm1, fm2")

    calls = []
    _patch_llm(monkeypatch, ok_urls=["http://f2/v1"], calls=calls)

    out = LLM.lmstudio_call("sys", "user")
    assert out == "ok:http://f2/v1:fm2"
    # Tried primary, f1, then f2 in priority order.
    assert calls == ["http://primary/v1", "http://f1/v1", "http://f2/v1"]


def test_lmstudio_call_does_not_try_duplicate_endpoint_twice(monkeypatch):
    # A failover duplicating the primary must not be re-tried.
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_URL", "http://primary/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_MODEL", "pm")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://primary/v1/, http://f2/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_MODELS", "pm, fm2")

    calls = []
    _patch_llm(monkeypatch, ok_urls=["http://f2/v1"], calls=calls)

    out = LLM.lmstudio_call("sys", "user")
    assert out == "ok:http://f2/v1:fm2"
    # primary tried once (not again as a failover), then f2.
    assert calls == ["http://primary/v1", "http://f2/v1"]


def test_lmstudio_call_surfaces_last_error_when_all_fail(monkeypatch):
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_URL", "http://primary/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_PRIMARY_MODEL", "pm")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_URLS", "http://f1/v1")
    monkeypatch.setenv("SLOPFINITY_LLM_FAILOVER_MODELS", "fm1")

    calls = []
    _patch_llm(monkeypatch, ok_urls=[], calls=calls)

    out = LLM.lmstudio_call("sys", "user")
    assert out.startswith("Error: All endpoints failed.")
    assert "refused" in out
    assert calls == ["http://primary/v1", "http://f1/v1"]
