"""Tests for slopfinity.scheduler — check_budget, acquire_gpu, free_between."""
from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest import mock

import pytest

# Ensure repo root is importable.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import scheduler as sched  # noqa: E402


def _mk_meminfo(mem_available_kb: int) -> str:
    return (
        "MemTotal:       134217728 kB\n"
        f"MemAvailable:   {mem_available_kb} kB\n"
        "Cached:              0 kB\n"
    )


def test_check_budget_qwen_plenty(monkeypatch):
    # 100 GB available — plenty for qwen (28 + overhead 6 + safety 10 = 44).
    m = mock.mock_open(read_data=_mk_meminfo(100 * 1024 * 1024))
    with mock.patch("builtins.open", m):
        ok, avail, need = sched.check_budget("image", "qwen")
    assert ok is True
    assert avail == pytest.approx(100.0, abs=0.1)
    assert need == pytest.approx(44.0, abs=0.1)


def test_check_budget_wan25_tight(monkeypatch):
    # 50 GB available — NOT enough for wan2.5 (96 + 6 + 10 = 112).
    m = mock.mock_open(read_data=_mk_meminfo(50 * 1024 * 1024))
    with mock.patch("builtins.open", m):
        ok, avail, need = sched.check_budget("video", "wan2.5")
    assert ok is False
    assert need == pytest.approx(112.0, abs=0.1)


def test_check_budget_unknown_stage(monkeypatch):
    # Unknown stage -> base 0 + overhead 6 + safety 10 = 16. Should pass @ 64 GB.
    m = mock.mock_open(read_data=_mk_meminfo(64 * 1024 * 1024))
    with mock.patch("builtins.open", m):
        ok, avail, need = sched.check_budget("mystery", "unknown")
    assert ok is True
    assert need == pytest.approx(16.0, abs=0.1)


def test_check_budget_respects_safety_override(monkeypatch):
    m = mock.mock_open(read_data=_mk_meminfo(50 * 1024 * 1024))
    with mock.patch("builtins.open", m):
        ok_default, _, need_default = sched.check_budget("image", "qwen", safety_gb=10)
        ok_strict, _, need_strict = sched.check_budget("image", "qwen", safety_gb=30)
    assert need_strict > need_default
    assert ok_default is True
    # 50 GB still > 28+6+30 = 64? No: 50 < 64 -> False.
    assert ok_strict is False


def test_acquire_gpu_concurrent_when_budget_fits(monkeypatch):
    """Phase 5 — two stages may run concurrently if their reservations fit.

    With 200 GB available and two qwen reservations (34 GB each = 68 GB)
    well under the 190 GB headroom, both stages should overlap.
    """
    # Plenty of host RAM; safety_gb=10 -> 190 GB headroom.
    #
    # NB: use a fresh-handle fake `open` rather than mock.mock_open. A single
    # mock_open handle exhausts its line iterator after the first read, so the
    # SECOND concurrent _mem_available_gb() call would iterate an empty file
    # and see 0 GB available — spuriously blocking the second reservation and
    # making the genuinely-concurrent path look serialized. _mem_available_gb
    # re-opens /proc/meminfo on every call, so the fake must too.
    meminfo = _mk_meminfo(200 * 1024 * 1024)

    def fake_open(path, *a, **k):
        return mock.mock_open(read_data=meminfo)(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    async def _noop(*a, **k):
        return {"ok": True, "before_gb": 100.0, "after_gb": 100.0, "freed_gb": 0.0}
    monkeypatch.setattr(sched, "free_between", _noop)
    # Stub auto-suspend: the default config has the LM-Studio entry ENABLED
    # (method=sigstop), so without this stub acquire_gpu would shell out to
    # pgrep/kill on every reservation — variable-latency subprocess work that
    # races against this test's barrier and is irrelevant to GPU budgeting.
    monkeypatch.setattr(sched, "_load_auto_suspend_entries", lambda: [])
    while not sched.SchedulerEvents.empty():
        sched.SchedulerEvents.get_nowait()

    async def main():
        sched.GPU = sched.GPUReservation()
        sched.gpu_lock = sched.GPU.cond
        sched.paused = asyncio.Event()
        sched.paused.set()

        order = []
        # Barrier instead of wall-clock timing: each worker records its
        # "start", waits until BOTH have started (proving they hold the GPU
        # concurrently — if acquire_gpu serialized, the second worker would
        # never reach the barrier and gather would deadlock/timeout), and
        # only then records its "end". This is deterministic and immune to
        # event-loop scheduling jitter between the awaits inside acquire_gpu.
        both_started = asyncio.Event()
        started_count = {"n": 0}

        async def worker(name: str):
            async with sched.acquire_gpu("image", "qwen"):
                order.append(("start", name))
                started_count["n"] += 1
                if started_count["n"] == 2:
                    both_started.set()
                # Wait for the other worker to also be inside the GPU
                # reservation. If reservations serialized, this blocks forever.
                await asyncio.wait_for(both_started.wait(), timeout=2.0)
                order.append(("end", name))

        await asyncio.gather(worker("a"), worker("b"))
        return order

    order = asyncio.run(main())
    assert len(order) == 4
    # Concurrent: both starts recorded before either end (guaranteed by the
    # barrier — both workers were simultaneously inside acquire_gpu).
    starts = [i for i, o in enumerate(order) if o[0] == "start"]
    ends = [i for i, o in enumerate(order) if o[0] == "end"]
    assert max(starts) < min(ends)


def test_acquire_gpu_serializes_when_budget_tight(monkeypatch):
    """Two oversized stages should still serialize when both can't fit."""
    # 70 GB available — fits ONE qwen (34) but not two (68 > 70-10 safety = 60).
    m = mock.mock_open(read_data=_mk_meminfo(70 * 1024 * 1024))
    monkeypatch.setattr("builtins.open", m)
    async def _noop(*a, **k):
        return {"ok": True, "before_gb": 70.0, "after_gb": 70.0, "freed_gb": 0.0}
    monkeypatch.setattr(sched, "free_between", _noop)
    while not sched.SchedulerEvents.empty():
        sched.SchedulerEvents.get_nowait()

    async def main():
        sched.GPU = sched.GPUReservation()
        sched.gpu_lock = sched.GPU.cond
        sched.paused = asyncio.Event()
        sched.paused.set()

        order = []

        async def worker(name: str):
            async with sched.acquire_gpu("image", "qwen"):
                order.append(("start", name, time.time()))
                await asyncio.sleep(0.05)
                order.append(("end", name, time.time()))

        await asyncio.gather(worker("a"), worker("b"))
        return order

    order = asyncio.run(main())
    assert len(order) == 4
    first_name = order[0][1]
    # Strict serialization — first stage must end before second starts.
    assert order[1] == ("end", first_name, order[1][2])
    assert order[2][1] != first_name


def test_planner_hit_skips_cold_load(monkeypatch):
    """Phase 5 — when use_planner=True and the model is resident from a
    previous stage, the second acquire_gpu reserves only OVERHEAD_GB."""
    m = mock.mock_open(read_data=_mk_meminfo(200 * 1024 * 1024))
    monkeypatch.setattr("builtins.open", m)
    async def _noop(*a, **k):
        return {"ok": True, "before_gb": 100.0, "after_gb": 100.0, "freed_gb": 0.0}
    monkeypatch.setattr(sched, "free_between", _noop)
    monkeypatch.setattr(sched, "_planner_enabled", lambda: True)
    while not sched.SchedulerEvents.empty():
        sched.SchedulerEvents.get_nowait()

    async def main():
        sched.GPU = sched.GPUReservation()
        sched.gpu_lock = sched.GPU.cond
        sched.paused = asyncio.Event()
        sched.paused.set()
        # First stage: cold load qwen.
        async with sched.acquire_gpu("image", "qwen"):
            pass
        # qwen should still be marked resident with use_planner=True.
        assert "qwen" in sched.GPU.resident_models
        # Second stage on same model: planner hit.
        async with sched.acquire_gpu("image", "qwen") as info:
            pass
        return True

    assert asyncio.run(main()) is True
    # Check that a planner_hit event was emitted.
    events = []
    while not sched.SchedulerEvents.empty():
        events.append(sched.SchedulerEvents.get_nowait())
    assert any(e["type"] == "planner_hit" for e in events), events


def test_free_between_posts_to_comfy(monkeypatch):
    """free_between should POST /free and return freed_gb."""
    calls = []

    class _FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=5):
        calls.append({
            "url": req.full_url,
            "method": req.get_method(),
            "body": req.data,
        })
        return _FakeResp()

    # Simulate more memory being available after the call.
    available_kb = {"n": 50 * 1024 * 1024}

    def fake_open(path, *a, **k):
        available_kb["n"] += 2 * 1024 * 1024  # +2 GB per read
        return mock.mock_open(read_data=_mk_meminfo(available_kb["n"]))(path, *a, **k)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with mock.patch("builtins.open", fake_open):
        result = asyncio.run(sched.free_between("http://localhost:8188"))

    assert calls, "urlopen should have been called"
    assert calls[0]["url"].endswith("/free")
    assert calls[0]["method"] == "POST"
    assert b"unload_models" in calls[0]["body"]
    assert result["ok"] is True
    assert result["freed_gb"] >= 0


def test_free_between_handles_network_error(monkeypatch):
    def boom(req, timeout=5):
        raise OSError("connection refused")

    m = mock.mock_open(read_data=_mk_meminfo(50 * 1024 * 1024))
    monkeypatch.setattr("urllib.request.urlopen", boom)
    with mock.patch("builtins.open", m):
        result = asyncio.run(sched.free_between())
    assert result["ok"] is False


def test_pause_resume_toggle():
    async def main():
        sched.paused = asyncio.Event()
        sched.paused.set()
        assert sched.is_paused() is False
        await sched.pause()
        assert sched.is_paused() is True
        await sched.resume()
        assert sched.is_paused() is False

    asyncio.run(main())


def test_stage_budget_gb_includes_overhead():
    assert sched.stage_budget_gb("image", "qwen") == 28 + sched.OVERHEAD_GB
    assert sched.stage_budget_gb("video", "wan2.5") == 96 + sched.OVERHEAD_GB
    assert sched.stage_budget_gb("bogus", "bogus") == sched.OVERHEAD_GB


def test_emergency_free_clears_resident_set_and_pkills(monkeypatch):
    """emergency_free should free_between, pkill launchers, AND wipe the
    scheduler's resident-set / in-flight tracking, emitting an event."""
    # Stub free_between so no real ComfyUI/network is touched.
    async def _fake_free_between(*a, **k):
        return {"ok": True, "before_gb": 10.0, "after_gb": 42.0, "freed_gb": 32.0}

    monkeypatch.setattr(sched, "free_between", _fake_free_between)

    # pkill returns 0 (process found) for every launcher pattern.
    def _fake_run(cmd, *a, **k):
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(sched.subprocess, "run", _fake_run)

    # Drain any stale events.
    while not sched.SchedulerEvents.empty():
        sched.SchedulerEvents.get_nowait()

    def main():
        sched.GPU = sched.GPUReservation()
        sched.gpu_lock = sched.GPU.cond
        # Seed bookkeeping that emergency_free must wipe.
        sched.GPU.resident_models = {"qwen": 1.0, "wan2.5": 2.0}
        sched.GPU.in_flight = {"job-1": {"stage": "image", "model": "qwen", "gb": 34}}
        sched.GPU.resident_gb = 34.0
        return asyncio.run(sched.emergency_free())

    result = main()

    # free_between result is propagated.
    assert result["ok"] is True
    assert result["freed_gb"] == 32.0
    # All three launchers reported killed.
    assert set(result["killed"]) == {
        "qwen_launcher.py",
        "ernie_launcher.py",
        "wan_launcher.py",
    }
    # Resident set + in-flight reported and actually cleared.
    assert set(result["evicted_models"]) == {"qwen", "wan2.5"}
    assert result["cleared_in_flight"] == 1
    assert sched.GPU.resident_models == {}
    assert sched.GPU.in_flight == {}
    assert sched.GPU.resident_gb == 0.0

    # An emergency_free event was emitted.
    events = []
    while not sched.SchedulerEvents.empty():
        events.append(sched.SchedulerEvents.get_nowait())
    ef = [e for e in events if e["type"] == "emergency_free"]
    assert ef, events
    assert ef[0]["freed_gb"] == 32.0
    assert set(ef[0]["evicted_models"]) == {"qwen", "wan2.5"}
