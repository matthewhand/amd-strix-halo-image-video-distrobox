"""POST /settings must reject SSRF-prone llm.base_url before persist."""
from __future__ import annotations

import importlib
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_CSRF = {"Origin": "http://testserver"}


@pytest.fixture()
def conf_env(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("SLOPFINITY_STATE_DIR", str(state))
    import slopfinity.config as cfg
    importlib.reload(cfg)
    cfg.save_config({"llm": {"api_key": ""}, "allow_cloud_endpoints": False})
    # Point server.cfg at reloaded module (same object after reload of config only).
    import slopfinity.server as server
    # Do not reload server (breaks router wiring); just rebind cfg reference.
    server.cfg = cfg
    from fastapi.testclient import TestClient
    yield TestClient(server.app), cfg
    importlib.reload(cfg)


def test_settings_rejects_metadata_base_url(conf_env):
    client, cfg = conf_env
    r = client.post(
        "/settings",
        json={"llm": {"base_url": "http://169.254.169.254/latest/meta-data/"}},
        headers=_CSRF,
    )
    assert r.status_code == 400, r.text
    body = r.json()
    assert body.get("ok") is False
    err = (body.get("error") or "").lower()
    assert "base_url" in err or "blocked" in err or "metadata" in err
    saved = (cfg.load_config().get("llm") or {}).get("base_url") or ""
    assert "169.254" not in saved


def test_settings_accepts_loopback_base_url(conf_env):
    client, cfg = conf_env
    r = client.post(
        "/settings",
        json={"llm": {"base_url": "http://127.0.0.1:1234/v1", "provider": "lmstudio"}},
        headers=_CSRF,
    )
    assert r.status_code == 200, r.text
    assert r.json().get("ok") is True
    saved = cfg.load_config()
    assert (saved.get("llm") or {}).get("base_url", "").startswith("http://127.0.0.1")
