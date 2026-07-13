"""Unit tests for slopfinity.service_registry (no docker / GPU)."""
from __future__ import annotations

import os
import sys
import urllib.error
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import service_registry as reg  # noqa: E402


def _http_ok(url, timeout=None):
    class R:
        status = 200
        def read(self, n=-1):
            return b'{"ok":true}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
    return R()


def _http_down(url, timeout=None):
    raise urllib.error.URLError("connection refused")


def _defaults():
    return reg.merge_network_services(None)


def test_merge_defaults_include_four_services():
    merged = _defaults()
    ids = {e["id"] for e in merged}
    assert {"qwen-image", "qwen-tts", "heartmula", "comfyui"} <= ids
    for e in merged:
        assert e.get("container_name")
        assert e.get("lifecycle_mode")
        assert e.get("health_path")
        assert e.get("start_cmd") or e.get("compose_service")
        assert e.get("stop_cmd") or e.get("container_name")


def test_merge_partial_override_and_null_ignored():
    stored = [
        {"id": "qwen-tts", "lifecycle_mode": "container", "health_url": None},
        {"id": "custom-svc", "label": "X", "enabled": True, "health_url": "http://x/h"},
    ]
    merged = reg.merge_network_services(stored)
    tts = next(e for e in merged if e["id"] == "qwen-tts")
    assert tts["lifecycle_mode"] == "container"
    # null does not clear default health_url
    assert tts.get("health_url")
    assert any(e["id"] == "custom-svc" for e in merged)


def test_normalize_stage_aliases():
    assert reg.normalize_stage("Base Image") == "image"
    assert reg.normalize_stage("TTS") == "tts"
    assert reg.normalize_stage("video") == "video"
    assert reg.normalize_stage("Post Process") == "upscale"


def test_service_for_stage_hygiene():
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        assert reg.service_for_stage("tts", "kokoro") == "qwen-tts"
        assert reg.service_for_stage("TTS", "kokoro") == "qwen-tts"
        assert reg.service_for_stage("audio", "heartmula") == "heartmula"
        assert reg.service_for_stage("image", "qwen") == "qwen-image"
        assert reg.service_for_stage("Base Image", "qwen") == "qwen-image"
        # wildcards removed — ernie/wan are one-shot
        assert reg.service_for_stage("image", "ernie") is None
        assert reg.service_for_stage("video", "wan2.2") is None
        assert reg.service_for_stage("video", "wan2.5") is None
        # LTX maps to Comfy
        assert reg.service_for_stage("video", "ltx-2.3") == "comfyui"
        assert reg.service_for_stage("image", "ltx-2.3") == "comfyui"
        assert reg.service_for_stage("upscale", "ltx-spatial") == "comfyui"
        assert reg.service_for_stage("concept", "llm") is None


def test_resolve_start_stop_container_mode():
    e = {
        "lifecycle_mode": "container",
        "container_name": "strix-halo-qwen-tts",
        "start_cmd": "docker compose --profile qwen-tts up -d qwen-tts-service",
        "stop_cmd": "docker stop other",
    }
    assert reg.resolve_start_cmd(e) == ["docker", "start", "strix-halo-qwen-tts"]
    assert reg.resolve_stop_cmd(e) == ["docker", "stop", "strix-halo-qwen-tts"]


def test_resolve_start_compose_uses_start_cmd():
    e = next(x for x in _defaults() if x["id"] == "qwen-tts")
    assert "compose" in str(reg.resolve_start_cmd(e))
    assert reg.resolve_stop_cmd(e) == ["docker", "stop", "strix-halo-qwen-tts"]


def test_exclusive_peers_uma_heavy():
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        peers = set(reg._exclusive_peers("qwen-image"))
        assert peers == {"heartmula", "comfyui"}
        assert reg._exclusive_peers("qwen-tts") == []


def test_probe_up():
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch("urllib.request.urlopen", side_effect=_http_ok):
            p = reg.probe("qwen-tts")
            assert p["ok"] is True
            assert p["status"] == 200


def test_probe_down():
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch("urllib.request.urlopen", side_effect=_http_down):
            p = reg.probe("qwen-tts")
            assert p["ok"] is False


def test_health_url_derives_from_remote_base_env(monkeypatch=None):
    # use mock env without pytest monkeypatch dependency
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch.dict(os.environ, {"TTS_WORKER_URL": "http://192.168.1.50:8010/tts"}, clear=False):
            url = reg.health_url_for("qwen-tts")
            assert url == "http://192.168.1.50:8010/health"
            assert reg.base_url_for("qwen-tts") == "http://192.168.1.50:8010"


def test_ensure_up_noop_when_healthy():
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch("urllib.request.urlopen", side_effect=_http_ok):
            with mock.patch.object(reg, "_run_cmd") as run:
                r = reg.ensure_up("qwen-tts", stop_exclusive=False)
                assert r["ok"] is True
                assert r.get("already_up") is True
                run.assert_not_called()


def test_ensure_up_starts_when_unhealthy():
    probes = iter([
        {"ok": False, "id": "qwen-tts", "error": "down"},
        {"ok": True, "id": "qwen-tts", "status": 200},
    ])

    def _probe(sid, timeout=2.0):
        try:
            return next(probes)
        except StopIteration:
            return {"ok": True, "id": sid, "status": 200}

    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch.object(reg, "probe", side_effect=_probe):
            with mock.patch.object(reg, "_run_cmd", return_value={"ok": True, "rc": 0}) as run:
                with mock.patch("slopfinity.service_registry.time.sleep", lambda s: None):
                    r = reg.ensure_up("qwen-tts", timeout_s=10, poll_s=0, stop_exclusive=False)
                assert r["ok"] is True
                assert r.get("started") is True
                run.assert_called()


def test_ensure_up_stops_exclusive_peers():
    probe_n = {"n": 0}
    downs = []

    def _probe(sid, timeout=2.0):
        if sid != "qwen-image":
            return {"ok": True, "id": sid, "status": 200}
        probe_n["n"] += 1
        # First post-peer probe is down (cold); subsequent healthy.
        if probe_n["n"] == 1:
            return {"ok": False, "id": sid}
        return {"ok": True, "id": sid, "status": 200}

    def _down(sid):
        downs.append(sid)
        return {"ok": True, "id": sid, "already_down": True}

    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch.object(reg, "probe", side_effect=_probe):
            with mock.patch.object(reg, "ensure_down", side_effect=_down):
                with mock.patch.object(reg, "_run_cmd", return_value={"ok": True, "rc": 0}) as run:
                    with mock.patch("slopfinity.service_registry.time.sleep", lambda s: None):
                        r = reg.ensure_up("qwen-image", timeout_s=5, poll_s=0, stop_exclusive=True)
                assert r["ok"] is True
                assert set(downs) == {"heartmula", "comfyui"}
                assert run.called


def test_ensure_up_already_up_still_stops_exclusive_peers():
    """UMA hygiene: warm target must still park exclusive peers (criterion 1)."""
    downs = []

    def _probe(sid, timeout=2.0):
        # Target already healthy; peers would also look up if probed.
        return {"ok": True, "id": sid, "status": 200}

    def _down(sid):
        downs.append(sid)
        return {"ok": True, "id": sid, "action": "stop"}

    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch.object(reg, "probe", side_effect=_probe):
            with mock.patch.object(reg, "ensure_down", side_effect=_down) as ed:
                with mock.patch.object(reg, "_run_cmd") as run:
                    r = reg.ensure_up("qwen-image", stop_exclusive=True)
    assert r["ok"] is True
    assert r.get("already_up") is True
    assert set(downs) == {"heartmula", "comfyui"}
    assert "peers" in r and len(r["peers"]) == 2
    # No start needed when already healthy
    run.assert_not_called()
    assert ed.call_count == 2

    # stop_exclusive=False must not park peers
    downs.clear()
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch.object(reg, "probe", side_effect=_probe):
            with mock.patch.object(reg, "ensure_down", side_effect=_down):
                r2 = reg.ensure_up("qwen-image", stop_exclusive=False)
    assert r2.get("already_up") is True
    assert downs == []


def test_ensure_down_idempotent_when_already_down():
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch("urllib.request.urlopen", side_effect=_http_down):
            with mock.patch.object(reg, "_run_cmd") as run:
                r = reg.ensure_down("qwen-image")
                assert r["ok"] is True
                assert r.get("already_down") is True
                run.assert_not_called()


def test_ensure_for_stage_maps_tts():
    with mock.patch.object(reg, "ensure_up", return_value={"ok": True, "id": "qwen-tts"}) as eu:
        with mock.patch.object(reg, "_entries", return_value=_defaults()):
            r = reg.ensure_for_stage("tts", "kokoro")
            assert r["ok"] is True
            eu.assert_called_once()
            assert eu.call_args[0][0] == "qwen-tts"


def test_ensure_for_stage_oneshot_clears_uma():
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch.object(reg, "ensure_down_group", return_value={"ok": True, "results": []}) as clr:
            r = reg.ensure_for_stage("image", "ernie")
            assert r.get("oneshot") is True
            assert r.get("skipped") is True
            clr.assert_called_once_with("uma-heavy")


def test_ensure_for_stage_base_image_label():
    with mock.patch.object(reg, "ensure_up", return_value={"ok": True, "id": "qwen-image"}) as eu:
        with mock.patch.object(reg, "_entries", return_value=_defaults()):
            reg.ensure_for_stage("Base Image", "qwen")
            eu.assert_called_once_with("qwen-image")


def test_run_cmd_inherits_docker_host():
    with mock.patch("subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.dict(
            os.environ,
            {"DOCKER_HOST": "ssh://gpu-host"},
            clear=False,
        ):
            # Ensure SLOPFINITY override is not forcing a different host
            os.environ.pop("SLOPFINITY_DOCKER_HOST", None)
            r = reg._run_cmd(["docker", "start", "strix-halo-qwen-tts"], timeout=5)
            assert r["ok"] is True
            assert sp.called
            env = sp.call_args.kwargs.get("env") or {}
            assert env.get("DOCKER_HOST") == "ssh://gpu-host"


def test_run_cmd_context_prefix():
    with mock.patch("subprocess.run") as sp:
        sp.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.dict(os.environ, {"SLOPFINITY_DOCKER_CONTEXT": "gpu"}, clear=False):
            r = reg._run_cmd(["docker", "stop", "strix-halo-qwen-tts"], timeout=5)
            assert r["ok"] is True
            argv = sp.call_args[0][0]
            assert argv[0] == "docker"
            assert argv[1:3] == ["--context", "gpu"]
            assert argv[3:] == ["stop", "strix-halo-qwen-tts"]


def test_status_all_shape():
    with mock.patch.object(reg, "_entries", return_value=_defaults()):
        with mock.patch("urllib.request.urlopen", side_effect=_http_ok):
            st = reg.status_all()
            assert len(st) >= 4
            assert all("up" in x and "id" in x and "lifecycle_mode" in x for x in st)


def test_disabled_ensure_up():
    entries = _defaults()
    for e in entries:
        if e["id"] == "qwen-tts":
            e["enabled"] = False
    with mock.patch.object(reg, "_entries", return_value=entries):
        r = reg.ensure_up("qwen-tts")
        assert r["ok"] is False
        assert "disabled" in r.get("error", "")


if __name__ == "__main__":
    import traceback
    failed = 0
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    for name, fn in tests:
        try:
            fn()
            print(f"{name} PASSED")
        except Exception:
            failed += 1
            print(f"{name} FAILED")
            traceback.print_exc()
    print(f"{len(tests) - failed} passed, {failed} failed in {len(tests)} tests")
    sys.exit(1 if failed else 0)
