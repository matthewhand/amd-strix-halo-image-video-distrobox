"""HTTP surface for docker lifecycle control (GET /services, warm/park).

Drives the real FastAPI routes with mocked service_registry so CI needs no GPU.
Also asserts dashboard assets wire to those endpoints (UI widen, not dead labels).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

_TMP = tempfile.mkdtemp(prefix="slop_svc_api_")
os.environ.setdefault("SLOPFINITY_EXP_DIR", _TMP)
os.environ.setdefault("SLOPFINITY_DISABLE_CSRF", "1")

from fastapi.testclient import TestClient  # noqa: E402
from slopfinity.server import app  # noqa: E402
from slopfinity import service_registry as reg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_get_services_returns_status_shape(client):
    fake = [
        {
            "id": "qwen-tts",
            "label": "TTS",
            "enabled": True,
            "health_url": "http://127.0.0.1:8010/health",
            "base_url": "http://127.0.0.1:8010",
            "container_name": "strix-halo-qwen-tts",
            "lifecycle_mode": "compose",
            "up": False,
            "probe": {"ok": False},
        }
    ]
    with mock.patch.object(reg, "status_all", return_value=fake):
        r = client.get("/services")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert len(body["services"]) == 1
    assert body["services"][0]["id"] == "qwen-tts"
    assert "up" in body["services"][0]


def test_warm_calls_ensure_up(client):
    with mock.patch.object(reg, "get_service", return_value={"id": "qwen-tts", "enabled": True}):
        with mock.patch.object(
            reg, "ensure_up", return_value={"ok": True, "id": "qwen-tts", "started": True}
        ) as eu:
            r = client.post("/services/qwen-tts/warm")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["action"] == "warm"
    eu.assert_called_once_with("qwen-tts")


def test_park_calls_ensure_down(client):
    with mock.patch.object(reg, "get_service", return_value={"id": "qwen-image", "enabled": True}):
        with mock.patch.object(
            reg, "ensure_down", return_value={"ok": True, "id": "qwen-image", "already_down": True}
        ) as ed:
            r = client.post("/services/qwen-image/park")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["action"] == "park"
    ed.assert_called_once_with("qwen-image")


def test_warm_unknown_service_404(client):
    with mock.patch.object(reg, "get_service", return_value=None):
        r = client.post("/services/nope/warm")
    assert r.status_code == 404
    assert r.json()["ok"] is False


def test_park_ensure_failure_503(client):
    with mock.patch.object(reg, "get_service", return_value={"id": "comfyui"}):
        with mock.patch.object(
            reg, "ensure_down", return_value={"ok": False, "id": "comfyui", "error": "stop failed"}
        ):
            r = client.post("/services/comfyui/park")
    assert r.status_code == 503
    assert r.json()["ok"] is False


def test_ui_wires_services_status_and_actions():
    """Dashboard must poll /services and POST warm/park — not dead labels only."""
    js = (ROOT / "slopfinity" / "static" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "slopfinity" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "network-services-list" in html
    assert "refreshNetworkServices" in html
    assert "function refreshNetworkServices" in js
    assert "fetch('/services')" in js or 'fetch("/services")' in js
    # warm/park are action args; URL is `/services/${id}/${action}`
    assert "networkServiceAction" in js
    assert "'warm'" in js and "'park'" in js
    assert "/services/" in js
    assert "function networkServiceAction" in js
    assert "renderNetworkServicesList" in js
    # Settings load hooks status refresh
    assert "refreshNetworkServices()" in js


def test_lifecycle_entrypoints_criteria_1_to_3():
    """Direct exercise of shipped registry (criteria 1–3) without docker."""
    entries = reg.merge_network_services(None)
    with mock.patch.object(reg, "_entries", return_value=entries):
        # 1: warm worker stage maps + exclusive peers
        assert reg.service_for_stage("image", "qwen") == "qwen-image"
        peers = set(reg._exclusive_peers("qwen-image"))
        assert peers == {"heartmula", "comfyui"}
        # 2: one-shot clears uma-heavy
        with mock.patch.object(reg, "ensure_down_group", return_value={"ok": True, "results": []}) as clr:
            r = reg.ensure_for_stage("video", "wan2.2")
            assert r.get("oneshot") is True
            clr.assert_called_once_with("uma-heavy")
        # 3: remote base → derived health
        with mock.patch.dict(os.environ, {"IMAGE_API_URL": "http://10.0.0.9:8180"}, clear=False):
            assert reg.health_url_for("qwen-image") == "http://10.0.0.9:8180/docs"
        # compose vs container cmd shapes
        e = next(x for x in entries if x["id"] == "qwen-tts")
        assert "compose" in str(reg.resolve_start_cmd(e))
        e2 = dict(e, lifecycle_mode="container")
        assert reg.resolve_start_cmd(e2) == ["docker", "start", "strix-halo-qwen-tts"]
        assert reg.resolve_stop_cmd(e2) == ["docker", "stop", "strix-halo-qwen-tts"]
