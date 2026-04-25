"""Unit tests for slopfinity.auto_suspend dispatcher.

Stdlib only. Each test mocks the underlying syscall (os.kill, urlopen,
subprocess.run, pgrep) so the suite is fully hermetic — no real
processes / network / docker required.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from unittest import mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import auto_suspend as asus  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# suspend_all / resume_all dispatch + filter behavior
# ---------------------------------------------------------------------------

def test_suspend_all_skips_disabled_entries():
    entries = [
        {"id": "a", "enabled": False, "method": "sigstop", "process_name": "x"},
        {"id": "b", "enabled": False, "method": "rest_unload", "endpoint": "http://localhost/free"},
    ]
    res = _run(asus.suspend_all(entries))
    assert res == []


def test_suspend_all_unknown_method_returns_error_not_raise():
    entries = [{"id": "weird", "enabled": True, "method": "warpdrive"}]
    res = _run(asus.suspend_all(entries))
    assert len(res) == 1
    assert res[0]["ok"] is False
    assert "unknown method" in res[0]["error"]


def test_suspend_all_continues_after_per_entry_failure():
    entries = [
        {"id": "broken", "enabled": True, "method": "sigstop"},  # missing process_name
        {"id": "ok", "enabled": True, "method": "sigstop", "process_name": "nothing-matches-this-xyz"},
    ]
    with mock.patch.object(asus, "_pgrep", return_value=[]):
        res = _run(asus.suspend_all(entries))
    assert len(res) == 2
    assert res[0]["ok"] is False  # broken entry
    assert res[1]["ok"] is True   # ok entry, even though no PIDs matched


# ---------------------------------------------------------------------------
# sigstop method
# ---------------------------------------------------------------------------

def test_sigstop_no_matching_processes_is_ok():
    entry = {"id": "lmstudio", "enabled": True, "method": "sigstop",
             "process_name": "definitely-not-a-real-process-xyz"}
    with mock.patch.object(asus, "_pgrep", return_value=[]):
        res = _run(asus.suspend_all([entry]))
    assert res[0]["ok"] is True
    assert res[0]["detail"]["pids"] == []


def test_sigstop_sends_sigstop_to_matched_pids():
    entry = {"id": "lmstudio", "enabled": True, "method": "sigstop",
             "process_name": "fakellm"}
    with mock.patch.object(asus, "_pgrep", return_value=[101, 202]), \
         mock.patch.object(os, "kill") as kill:
        res = _run(asus.suspend_all([entry]))
    assert res[0]["ok"] is True
    assert res[0]["detail"]["pids"] == [101, 202]
    kill.assert_any_call(101, signal.SIGSTOP)
    kill.assert_any_call(202, signal.SIGSTOP)


def test_sigstop_resume_sends_sigcont():
    entry = {"id": "lmstudio", "enabled": True, "method": "sigstop",
             "process_name": "fakellm"}
    with mock.patch.object(asus, "_pgrep", return_value=[42]), \
         mock.patch.object(os, "kill") as kill:
        res = _run(asus.resume_all([entry]))
    assert res[0]["ok"] is True
    kill.assert_called_once_with(42, signal.SIGCONT)


def test_sigstop_swallows_permission_and_lookup_errors():
    """A dead/unreachable PID shouldn't crash the dispatcher."""
    entry = {"id": "lmstudio", "enabled": True, "method": "sigstop",
             "process_name": "fakellm"}
    def _kill(pid, sig):
        raise ProcessLookupError("gone")
    with mock.patch.object(asus, "_pgrep", return_value=[999]), \
         mock.patch.object(os, "kill", side_effect=_kill):
        res = _run(asus.suspend_all([entry]))
    assert res[0]["ok"] is True
    assert res[0]["detail"]["pids"] == []  # nothing successfully signalled
    assert res[0]["detail"]["errors"]


# ---------------------------------------------------------------------------
# rest_unload method
# ---------------------------------------------------------------------------

def test_rest_unload_posts_to_endpoint_with_default_body():
    """Mock urllib to verify POST body and URL."""
    captured = {}

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=5):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = req.data
        captured["content_type"] = req.headers.get("Content-type")
        return _Resp()

    entry = {"id": "comfyui", "enabled": True, "method": "rest_unload",
             "endpoint": "http://localhost:8188/free"}
    with mock.patch("urllib.request.urlopen", fake_urlopen):
        res = _run(asus.suspend_all([entry]))

    assert res[0]["ok"] is True
    assert res[0]["detail"]["status"] == 200
    assert captured["url"] == "http://localhost:8188/free"
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    body = json.loads(captured["body"].decode("utf-8"))
    assert body == {"unload_models": True, "free_memory": True}


def test_rest_unload_resume_is_noop():
    """Resume side does NOT call the endpoint — it's lazy by design."""
    entry = {"id": "comfyui", "enabled": True, "method": "rest_unload",
             "endpoint": "http://localhost:8188/free"}
    with mock.patch("urllib.request.urlopen") as up:
        res = _run(asus.resume_all([entry]))
    assert res[0]["ok"] is True
    assert "skipped" in res[0]["detail"]
    up.assert_not_called()


def test_rest_unload_missing_endpoint_errors_cleanly():
    entry = {"id": "comfyui", "enabled": True, "method": "rest_unload"}
    res = _run(asus.suspend_all([entry]))
    assert res[0]["ok"] is False
    assert "endpoint" in res[0]["error"]


# ---------------------------------------------------------------------------
# docker_stop method
# ---------------------------------------------------------------------------

def test_docker_stop_runs_docker_stop_then_start():
    """Suspend should run `docker stop`, resume should run `docker start`."""
    calls = []

    def fake_run(cmd, capture_output=True, text=True, timeout=30):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    entry = {"id": "qwen-tts", "enabled": True, "method": "docker_stop",
             "container": "strix-halo-qwen-tts"}
    with mock.patch.object(subprocess, "run", side_effect=fake_run):
        s_res = _run(asus.suspend_all([entry]))
        r_res = _run(asus.resume_all([entry]))

    assert s_res[0]["ok"] is True
    assert r_res[0]["ok"] is True
    assert calls[0] == ["docker", "stop", "strix-halo-qwen-tts"]
    assert calls[1] == ["docker", "start", "strix-halo-qwen-tts"]


def test_docker_stop_nonzero_rc_reports_error():
    def fake_run(cmd, capture_output=True, text=True, timeout=30):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No such container")

    entry = {"id": "qwen-tts", "enabled": True, "method": "docker_stop",
             "container": "ghost"}
    with mock.patch.object(subprocess, "run", side_effect=fake_run):
        res = _run(asus.suspend_all([entry]))
    assert res[0]["ok"] is False
    assert "No such container" in res[0]["error"]


def test_docker_stop_missing_container_errors_cleanly():
    entry = {"id": "qwen-tts", "enabled": True, "method": "docker_stop"}
    res = _run(asus.suspend_all([entry]))
    assert res[0]["ok"] is False
    assert "container" in res[0]["error"]


# ---------------------------------------------------------------------------
# sigterm method
# ---------------------------------------------------------------------------

def test_sigterm_sends_sigterm_on_suspend_only():
    entry = {"id": "kill-me", "enabled": True, "method": "sigterm",
             "process_name": "fake"}
    with mock.patch.object(asus, "_pgrep", return_value=[7]), \
         mock.patch.object(os, "kill") as kill:
        s_res = _run(asus.suspend_all([entry]))
        r_res = _run(asus.resume_all([entry]))
    kill.assert_called_once_with(7, signal.SIGTERM)
    assert s_res[0]["ok"] is True
    assert r_res[0]["ok"] is True
    assert "skipped" in r_res[0]["detail"]


# ---------------------------------------------------------------------------
# Legacy compat shims used by /llm/suspend and /llm/resume
# ---------------------------------------------------------------------------

def test_legacy_lmstudio_returns_pid_list_shape():
    with mock.patch.object(asus, "_pgrep", return_value=[1234]), \
         mock.patch.object(os, "kill"):
        res = _run(asus.legacy_suspend_lmstudio())
    assert res["suspended"] == [1234]
    assert "results" in res
