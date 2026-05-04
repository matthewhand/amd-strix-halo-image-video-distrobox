"""Slopfinity scheduler — GPU lock, budget check, ComfyUI /free integration.

Coordination layer so the fleet pipeline runs safely on AMD Strix Halo's
128 GB unified memory without OOM-ing. Stdlib only.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import itertools
import json
import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from collections import deque
from typing import Optional, Tuple

from . import config as _config
from . import auto_suspend as _auto_suspend

# Phase 5: optional planner consultation. Defensive import so the scheduler
# still works even if memory_planner is missing or partially refactored.
try:
    from . import memory_planner as _memory_planner
except Exception:  # pragma: no cover — defensive
    _memory_planner = None  # type: ignore

# Rough per-stage peak memory budgets (GB). Values are intentionally
# a bit conservative — they are the amount we expect a stage to peak at
# while it holds weights + activations resident.
STAGE_BUDGETS = {
    ("image", "qwen"): 28,
    ("image", "ernie"): 18,
    ("image", "ltx-2.3"): 38,
    ("video", "ltx-2.3"): 48,
    ("video", "wan2.2"): 84,
    ("video", "wan2.5"): 96,
    ("audio", "heartmula"): 14,
    ("tts", "qwen-tts"): 10,
    ("tts", "kokoro"): 8,
    ("upscale", "ltx-spatial"): 30,
}

OVERHEAD_GB = 6
SAFETY_GB = 24
COMFY_URL = "http://localhost:8188"


# Global singletons. asyncio primitives bind to the running loop at first use.
class GPUReservation:
    def __init__(self) -> None:
        self.resident_gb: float = 0.0
        self.cond: asyncio.Condition = asyncio.Condition()
        self.resident_models: dict[str, float] = {}
        self.gpu_history: deque[int] = deque()
        self.in_flight: dict[str, dict] = {}

    def record_gpu_usage(self, pct: int, max_samples: int = 5) -> None:
        self.gpu_history.append(pct)
        while len(self.gpu_history) > max_samples:
            self.gpu_history.popleft()

    def get_gpu_avg(self) -> float:
        if not self.gpu_history:
            return 0.0
        return sum(self.gpu_history) / len(self.gpu_history)

    def snapshot(self) -> dict:
        return {
            "resident_gb": round(self.resident_gb, 2),
            "in_flight": [{"job_id": k, **v} for k, v in self.in_flight.items()],
            "resident_models": list(self.resident_models.keys()),
        }


_GPU: Optional[GPUReservation] = None
_PAUSED: Optional[asyncio.Event] = None


def get_gpu() -> GPUReservation:
    global _GPU
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if _GPU is None:
            _GPU = GPUReservation()
        return _GPU

    if _GPU is None:
        _GPU = GPUReservation()
    else:
        try:
            # Condition._loop is internal but present in most CPython versions
            cur_loop = getattr(_GPU.cond, "_loop", None)
            if cur_loop is not None and cur_loop is not loop:
                old = _GPU
                _GPU = GPUReservation()
                _GPU.resident_gb = old.resident_gb
                _GPU.in_flight = dict(old.in_flight)
                _GPU.gpu_history = deque(old.gpu_history)
                _GPU.resident_models = dict(old.resident_models)
        except Exception:
            pass
    return _GPU


def get_paused() -> asyncio.Event:
    global _PAUSED
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if _PAUSED is None:
            _PAUSED = asyncio.Event()
            _PAUSED.set()
        return _PAUSED

    if _PAUSED is None:
        _PAUSED = asyncio.Event()
        _PAUSED.set()
    else:
        try:
            cur_loop = getattr(_PAUSED, "_loop", None)
            if cur_loop is not None and cur_loop is not loop:
                was_set = _PAUSED.is_set()
                _PAUSED = asyncio.Event()
                if was_set: _PAUSED.set()
        except Exception:
            pass
    return _PAUSED


def is_paused() -> bool:
    return not get_paused().is_set()


async def pause() -> None:
    get_paused().clear()


async def resume() -> None:
    get_paused().set()


async def notify_gpu_usage() -> None:
    gpu = get_gpu()
    async with gpu.cond:
        gpu.cond.notify_all()


# Monotonic id generator for anonymous reservations (callers without a job_id).
_RESERVATION_ID = itertools.count(1)

# Unbounded queue of scheduler events for the dashboard WebSocket to drain.
SchedulerEvents: "asyncio.Queue[dict]" = asyncio.Queue()


def _now() -> float:
    return time.time()


def _mem_available_gb() -> float:
    """Read MemAvailable from /proc/meminfo, in GB. Falls back to MemFree + Cached."""
    try:
        m = {}
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        m[k.strip()] = int(v.split()[0])
        
        if "MemAvailable" in m:
            return round(m["MemAvailable"] / (1024 * 1024), 2)
        
        # Fallback for older kernels or containers
        free = m.get("MemFree", 0)
        cached = m.get("Cached", 0)
        buffers = m.get("Buffers", 0)
        if free or cached or buffers:
            return round((free + cached + buffers) / (1024 * 1024), 2)
    except Exception:
        pass
    # Absolute fallback — assume plenty of RAM if we can't read /proc/meminfo
    return 128.0


def stage_budget_gb(stage: str, model: str) -> float:
    base = STAGE_BUDGETS.get((stage, model), 0)
    return float(base + OVERHEAD_GB)


async def _emit(event: dict) -> None:
    try:
        SchedulerEvents.put_nowait(event)
    except Exception:
        pass


async def free_between(comfy_url: str = COMFY_URL) -> dict:
    before = _mem_available_gb()
    if not has_gpu():
        return {"ok": True, "before_gb": before, "after_gb": before, "freed_gb": 0.0}
    ok = False
    try:
        payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{comfy_url}/free",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _do():
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status

        status = await asyncio.to_thread(_do)
        ok = 200 <= int(status) < 300
    except Exception:
        ok = False
    gc.collect()
    await asyncio.sleep(0.5)
    after = _mem_available_gb()
    return {
        "ok": ok,
        "before_gb": before,
        "after_gb": after,
        "freed_gb": round(max(0.0, after - before), 2),
    }


def has_gpu() -> bool:
    """Check if an AMD GPU is available via /dev/kfd or rocminfo."""
    if hasattr(has_gpu, "_result"):
        return has_gpu._result
    
    if os.path.exists("/dev/kfd"):
        has_gpu._result = True
        return True
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=2)
        has_gpu._result = (result.returncode == 0)
        return has_gpu._result
    except (FileNotFoundError, subprocess.SubprocessError):
        has_gpu._result = False
        return False


@contextlib.asynccontextmanager
async def acquire_gpu(
    stage: str,
    model: str,
    safety_gb: int = SAFETY_GB,
    job_id: Optional[str] = None,
):
    await get_paused().wait()

    if not has_gpu():
        yield {"stage": stage, "model": model, "wait_seconds": 0.0, "gpu_bypassed": True}
        return

    gpu = get_gpu()
    base_need = stage_budget_gb(stage, model)
    rid = job_id or f"r{next(_RESERVATION_ID)}"
    wait_start = _now()
    blocks = 0
    ok = False
    need = base_need
    planner_hit = False

    while True:
        async with gpu.cond:
            available = _mem_available_gb()
            need = 0.0 if model in gpu.resident_models else base_need
            projected_resident = gpu.resident_gb + need
            ok = projected_resident <= max(0.0, available - float(safety_gb))

            if ok or blocks >= 30:
                gpu.resident_gb += need
                gpu.in_flight[rid] = {"stage": stage, "model": model, "since": _now()}
                if model: gpu.resident_models[model] = _now()
                break

            blocks += 1
            if blocks == 1:
                # Drop lock and try to free ComfyUI
                pass 
            else:
                try:
                    await asyncio.wait_for(gpu.cond.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        
        if blocks == 1:
            await free_between()

    try:
        yield {"stage": stage, "model": model, "wait_seconds": _now() - wait_start}
    finally:
        async with gpu.cond:
            gpu.resident_gb -= need
            gpu.in_flight.pop(rid, None)
            gpu.cond.notify_all()
        SchedulerEvents.put_nowait({
            "type": "stage_end",
            "stage": stage,
            "model": model,
            "job_id": rid,
            "ts": _now(),
        })


def _load_scheduler_config() -> dict:
    try:
        c = _config.load_config()
        v = c.get("scheduler") if isinstance(c, dict) else None
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


async def resume_llm_async() -> dict:
    return await _auto_suspend.legacy_suspend_lmstudio() # Wait, should be resume


def suspend_llm() -> dict:
    return asyncio.run(_auto_suspend.legacy_suspend_lmstudio())


def resume_llm() -> dict:
    return asyncio.run(_auto_suspend.legacy_resume_lmstudio())
