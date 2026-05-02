"""TDD tests for GPU Idle Guard (Pause for Idle GPU) feature."""
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

@pytest.fixture(autouse=True)
def setup_scheduler():
    """Reset scheduler state before each test."""
    sched.GPU = sched.GPUReservation()
    sched.gpu_lock = sched.GPU.cond
    sched.paused = asyncio.Event()
    sched.paused.set()
    # Clear event queue
    while not sched.SchedulerEvents.empty():
        try:
            sched.SchedulerEvents.get_nowait()
        except asyncio.QueueEmpty:
            break

def test_gpu_reservation_history_init():
    """Verify that GPUReservation starts with an empty history."""
    assert hasattr(sched.GPU, "gpu_history")
    assert len(sched.GPU.gpu_history) == 0

def test_record_gpu_usage():
    """Verify that record_gpu_usage adds samples and respects the limit."""
    sched.GPU.record_gpu_usage(10, max_samples=3)
    sched.GPU.record_gpu_usage(20, max_samples=3)
    sched.GPU.record_gpu_usage(30, max_samples=3)
    assert list(sched.GPU.gpu_history) == [10, 20, 30]
    
    sched.GPU.record_gpu_usage(40, max_samples=3)
    assert list(sched.GPU.gpu_history) == [20, 30, 40]

def test_get_gpu_avg():
    """Verify that get_gpu_avg calculates the mean correctly."""
    assert sched.GPU.get_gpu_avg() == 0.0 # Empty case
    
    sched.GPU.record_gpu_usage(10)
    sched.GPU.record_gpu_usage(30)
    assert sched.GPU.get_gpu_avg() == 20.0
    
    sched.GPU.record_gpu_usage(50)
    assert sched.GPU.get_gpu_avg() == 30.0

@pytest.mark.asyncio
async def test_acquire_gpu_waits_for_idle_gpu(monkeypatch):
    """Verify that acquire_gpu blocks when average GPU usage is above threshold."""
    # Mock plenty of memory so we only test GPU usage gating.
    monkeypatch.setattr(sched, "_mem_available_gb", lambda: 500.0)
    
    # Mock config: enabled, threshold 50%, 3 samples.
    config = {
        "pause_for_idle_gpu": True,
        "pause_idle_max_pct": 50,
        "pause_idle_samples": 3
    }
    monkeypatch.setattr(sched, "_load_scheduler_config", lambda: config)
    
    # Fill history with busy samples (80%)
    for _ in range(3):
        sched.GPU.record_gpu_usage(80)
        
    # Attempt to acquire GPU
    # Since we need to yield to the loop to see it block, we'll use a task.
    acquire_called = asyncio.Event()
    
    async def task():
        async with sched.acquire_gpu("video", "wan2.5"):
            acquire_called.set()
            
    t = asyncio.create_task(task())
    
    # Wait a bit — it should be blocked by GPU usage.
    await asyncio.sleep(0.2)
    assert not acquire_called.is_set()
    
    # Check for gpu_idle_wait event
    events = []
    while not sched.SchedulerEvents.empty():
        events.append(sched.SchedulerEvents.get_nowait())
    assert any(e["type"] == "gpu_idle_wait" for e in events)

    # Now make GPU idle (20%)
    for _ in range(3):
        sched.GPU.record_gpu_usage(20)
        
    # Wait for the next poll cycle (scheduler polls every 1.0s or on notify)
    # We should notify to speed it up, or just wait.
    async with sched.GPU.cond:
        sched.GPU.cond.notify_all()
        
    await asyncio.wait_for(acquire_called.wait(), timeout=2.0)
    assert acquire_called.is_set()
    t.cancel()

@pytest.mark.asyncio
async def test_acquire_gpu_ignores_busy_gpu_when_disabled(monkeypatch):
    """Verify that acquire_gpu continues immediately if the guard is disabled."""
    monkeypatch.setattr(sched, "_mem_available_gb", lambda: 500.0)
    
    # Mock config: DISABLED.
    config = {
        "pause_for_idle_gpu": False,
        "pause_idle_max_pct": 50,
        "pause_idle_samples": 3
    }
    monkeypatch.setattr(sched, "_load_scheduler_config", lambda: config)
    
    # GPU is busy (80%)
    for _ in range(3):
        sched.GPU.record_gpu_usage(80)
        
    # This should succeed immediately.
    async with sched.acquire_gpu("video", "wan2.5"):
        pass
    assert True # Reached here immediately
