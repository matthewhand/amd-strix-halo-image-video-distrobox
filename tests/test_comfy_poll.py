"""_poll_comfy_history must never hang the orchestrator: it has a hard deadline,
a per-request socket timeout, and a consecutive-error cap. These cover success,
ComfyUI execution_error, deadline-timeout, and prolonged-unreachable."""
import json

import pytest

import run_fleet


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._p).encode()


def test_poll_returns_filenames_on_completion(monkeypatch):
    monkeypatch.setattr(run_fleet.time, "sleep", lambda *_: None)
    payload = {"pid": {"status": {"completed": True},
                       "outputs": {"14": {"images": [{"filename": "a.png"},
                                                     {"filename": "b.png"}]}}}}
    monkeypatch.setattr(run_fleet.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(payload))
    out = run_fleet._poll_comfy_history("pid", "14", poll_s=0, label="t")
    assert out == ["a.png", "b.png"]


def test_poll_raises_on_execution_error(monkeypatch):
    monkeypatch.setattr(run_fleet.time, "sleep", lambda *_: None)
    payload = {"pid": {"status": {"completed": True,
                                  "messages": [["execution_error", {"x": 1}]]},
                       "outputs": {}}}
    monkeypatch.setattr(run_fleet.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp(payload))
    with pytest.raises(RuntimeError, match="execution error"):
        run_fleet._poll_comfy_history("pid", "14", poll_s=0, label="t")


def test_poll_times_out_without_hanging(monkeypatch):
    # job never appears in history → must raise on the deadline, not spin forever.
    monkeypatch.setattr(run_fleet.time, "sleep", lambda *_: None)
    monkeypatch.setattr(run_fleet.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResp({}))
    clock = {"v": 1000.0}

    def fake_time():
        clock["v"] += 100.0
        return clock["v"]

    monkeypatch.setattr(run_fleet.time, "time", fake_time)
    with pytest.raises(RuntimeError, match="timed out"):
        run_fleet._poll_comfy_history("pid", "14", timeout_s=1, poll_s=0, label="t")


def test_poll_raises_when_unreachable(monkeypatch):
    # urlopen always errors → must raise after the consecutive-error cap, not
    # block on a dead connection forever.
    monkeypatch.setattr(run_fleet.time, "sleep", lambda *_: None)

    def boom(*a, **k):
        raise run_fleet.urllib.error.URLError("conn refused")

    monkeypatch.setattr(run_fleet.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="unreachable"):
        run_fleet._poll_comfy_history("pid", "14", timeout_s=100000, poll_s=0, label="t")
