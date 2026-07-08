"""Unit tests for slopfinity.service_registry (no docker / GPU)."""
from __future__ import annotations

import io
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


def test_merge_defaults_include_four_services():
    merged = reg.merge_network_services(None)
    ids = {e["id"] for e in merged}
    assert {"qwen-image", "qwen-tts", "heartmula", "comfyui"} <= ids
    for e in merged:
        assert e.get("start_cmd")
        assert e.get("stop_cmd")
        assert e.get("health_url")


def test_service_for_stage_tts_and_wildcard():
    with mock.patch.object(reg, "_entries", return_value=reg.merge_network_services(None)):
        assert reg.service_for_stage("tts", "kokoro") == "qwen-tts"
        assert reg.service_for_stage("audio", "heartmula") == "heartmula"
        assert reg.service_for_stage("image", "qwen") == "qwen-image"
        assert reg.service_for_stage("video", "ltx-2.3") == "comfyui"
        assert reg.service_for_stage("concept", "llm") is None


def test_probe_up():
    with mock.patch.object(reg, "_entries", return_value=reg.merge_network_services(None)):
        with mock.patch("urllib.request.urlopen", side_effect=_http_ok):
            p = reg.probe("qwen-tts")
            assert p["ok"] is True
            assert p["status"] == 200


def test_probe_down():
    with mock.patch.object(reg, "_entries", return_value=reg.merge_network_services(None)):
        with mock.patch("urllib.request.urlopen", side_effect=_http_down):
            p = reg.probe("qwen-tts")
            assert p["ok"] is False


def test_ensure_up_noop_when_healthy():
    with mock.patch.object(reg, "_entries", return_value=reg.merge_network_services(None)):
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

    with mock.patch.object(reg, "_entries", return_value=reg.merge_network_services(None)):
        with mock.patch.object(reg, "probe", side_effect=_probe):
            with mock.patch.object(reg, "_run_cmd", return_value={"ok": True, "rc": 0}) as run:
                with mock.patch("slopfinity.service_registry.time.sleep", lambda s: None):
                    r = reg.ensure_up("qwen-tts", timeout_s=10, poll_s=0, stop_exclusive=False)
                assert r["ok"] is True
                assert r.get("started") is True
                run.assert_called()


def test_ensure_down_idempotent_when_already_down():
    with mock.patch.object(reg, "_entries", return_value=reg.merge_network_services(None)):
        with mock.patch("urllib.request.urlopen", side_effect=_http_down):
            with mock.patch.object(reg, "_run_cmd") as run:
                r = reg.ensure_down("qwen-image")
                assert r["ok"] is True
                assert r.get("already_down") is True
                run.assert_not_called()


def test_ensure_for_stage_maps_tts():
    with mock.patch.object(reg, "ensure_up", return_value={"ok": True, "id": "qwen-tts"}) as eu:
        with mock.patch.object(reg, "_entries", return_value=reg.merge_network_services(None)):
            r = reg.ensure_for_stage("tts", "kokoro")
            assert r["ok"] is True
            eu.assert_called_once()
            assert eu.call_args[0][0] == "qwen-tts"


def test_status_all_shape():
    with mock.patch.object(reg, "_entries", return_value=reg.merge_network_services(None)):
        with mock.patch("urllib.request.urlopen", side_effect=_http_ok):
            st = reg.status_all()
            assert len(st) >= 4
            assert all("up" in x and "id" in x for x in st)


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
