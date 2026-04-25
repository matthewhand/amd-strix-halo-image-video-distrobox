"""Slopfinity scheduler — GPU lock, budget check, ComfyUI /free integration.

Coordination layer so the fleet pipeline runs safely on AMD Strix Halo's
128 GB unified memory without OOM-ing. Stdlib only.
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from typing import Optional, Tuple

from . import config as _config
from . import auto_suspend as _auto_suspend

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
SAFETY_GB = 10
COMFY_URL = "http://localhost:8188"

# Global singletons. asyncio primitives bind to the running loop at first use.
gpu_lock = asyncio.Lock()
paused = asyncio.Event()
paused.set()  # start un-paused

# Unbounded queue of scheduler events for the dashboard WebSocket to drain.
SchedulerEvents: "asyncio.Queue[dict]" = asyncio.Queue()


def _now() -> float:
    return time.time()


def _mem_available_gb() -> float:
    """Read MemAvailable from /proc/meminfo, in GB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def stage_budget_gb(stage: str, model: str) -> float:
    """Return the estimated peak for (stage, model) incl. overhead."""
    base = STAGE_BUDGETS.get((stage, model), 0)
    return float(base + OVERHEAD_GB)


def check_budget(stage: str, model: str, safety_gb: int = SAFETY_GB) -> Tuple[bool, float, float]:
    """Return (ok, available_gb, needed_gb).

    `needed_gb` = stage peak + overhead + safety margin.
    `ok` iff MemAvailable >= needed_gb.
    """
    available = _mem_available_gb()
    needed = stage_budget_gb(stage, model) + float(safety_gb)
    return (available >= needed, available, needed)


async def _emit(event: dict) -> None:
    try:
        SchedulerEvents.put_nowait(event)
    except Exception:
        pass


async def free_between(comfy_url: str = COMFY_URL) -> dict:
    """POST /free to ComfyUI, gc.collect(), 500 ms quiesce.

    Returns {ok, before_gb, after_gb, freed_gb}.
    """
    before = _mem_available_gb()
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


async def emergency_free() -> dict:
    """free_between() then pkill stray model launchers."""
    result = await free_between()
    killed = []
    for pat in ("qwen_launcher.py", "ernie_launcher.py", "wan_launcher.py"):
        try:
            rc = await asyncio.to_thread(
                lambda p=pat: subprocess.run(
                    ["pkill", "-f", p], capture_output=True, text=True
                ).returncode
            )
            if rc == 0:
                killed.append(pat)
        except Exception:
            pass
    result["killed"] = killed
    await _emit({
        "type": "emergency_free",
        "stage": None,
        "model": None,
        "freed_gb": result.get("freed_gb", 0.0),
        "killed": killed,
        "ts": _now(),
    })
    return result


@contextlib.asynccontextmanager
async def acquire_gpu(stage: str, model: str, safety_gb: int = SAFETY_GB):
    """Async context manager serializing GPU-resident stages.

    1. Waits for `paused` to be set (un-paused).
    2. Acquires `gpu_lock`.
    3. Polls `check_budget` until satisfied (emits `budget_block`).
    4. Yields control to the stage.
    5. On exit, calls `free_between()` and emits `stage_end`.
    """
    await paused.wait()
    wait_start = _now()
    async with gpu_lock:
        blocks = 0
        while True:
            ok, available, needed = check_budget(stage, model, safety_gb=safety_gb)
            if ok:
                break
            blocks += 1
            await _emit({
                "type": "budget_block",
                "stage": stage,
                "model": model,
                "available_gb": available,
                "needed_gb": needed,
                "ts": _now(),
            })
            await free_between()
            await asyncio.sleep(1.0)
            if blocks >= 30:  # ~30s of retries — bail out and let caller OOM-retry
                await _emit({
                    "type": "oom_retry",
                    "stage": stage,
                    "model": model,
                    "available_gb": available,
                    "needed_gb": needed,
                    "ts": _now(),
                })
                break

        wait_seconds = round(_now() - wait_start, 2)
        await _emit({
            "type": "stage_start",
            "stage": stage,
            "model": model,
            "budget_gb": stage_budget_gb(stage, model),
            "available_gb": _mem_available_gb(),
            "wait_seconds": wait_seconds,
            "ts": _now(),
        })

        # Auto-suspend co-resident services (LM Studio, ComfyUI, Qwen-TTS, ...)
        # via the configured `auto_suspend` list. Fast methods (sigstop) take
        # ~10 ms; heavyweight methods (docker_stop) take a few seconds — we
        # accept that as the cost of a clean stage. See auto_suspend.py.
        as_entries = _load_auto_suspend_entries()
        try:
            sus_results = await _auto_suspend.suspend_all(as_entries)
            if sus_results:
                await _emit({
                    "type": "auto_suspend_start",
                    "stage": stage,
                    "model": model,
                    "results": sus_results,
                    "ts": _now(),
                })
        except Exception:
            sus_results = []

        peak_before = _mem_available_gb()
        start = _now()
        try:
            yield {"stage": stage, "model": model, "wait_seconds": wait_seconds}
        finally:
            peak_after = _mem_available_gb()
            peak_used = round(max(0.0, peak_before - peak_after), 2)
            try:
                await free_between()
            except Exception:
                pass
            try:
                res_results = await _auto_suspend.resume_all(as_entries)
                if res_results:
                    await _emit({
                        "type": "auto_suspend_end",
                        "stage": stage,
                        "model": model,
                        "results": res_results,
                        "ts": _now(),
                    })
            except Exception:
                pass
            await _emit({
                "type": "stage_end",
                "stage": stage,
                "model": model,
                "peak_mem_gb": peak_used,
                "duration_seconds": round(_now() - start, 2),
                "ts": _now(),
            })


async def pause() -> None:
    paused.clear()


async def resume() -> None:
    paused.set()


def is_paused() -> bool:
    return not paused.is_set()


# ---------------------------------------------------------------------------
# Auto-suspend integration — see slopfinity/auto_suspend.py for the four
# methods (sigstop / rest_unload / docker_stop / sigterm) and
# docs/auto-suspend-design.md for the design rationale.
# ---------------------------------------------------------------------------

def _load_auto_suspend_entries() -> list[dict]:
    """Read the `auto_suspend` list from config.json. Empty list on any error."""
    try:
        v = _config.load_config().get("auto_suspend")
        return v if isinstance(v, list) else []
    except Exception:
        return []


# Legacy compatibility shims — PR #40 exported `suspend_llm` / `resume_llm`
# and the fleet runner / external callers may still import them. They now
# dispatch through the new framework's lmstudio entry. New code should call
# `slopfinity.auto_suspend.suspend_all(...)` directly with the desired list.

async def suspend_llm_async() -> dict:
    return await _auto_suspend.legacy_suspend_lmstudio()


async def resume_llm_async() -> dict:
    return await _auto_suspend.legacy_resume_lmstudio()


def suspend_llm() -> dict:
    """PR #40 compatibility wrapper. Returns {"suspended": [pids]}."""
    return asyncio.run(_auto_suspend.legacy_suspend_lmstudio())


def resume_llm() -> dict:
    """PR #40 compatibility wrapper. Returns {"resumed": [pids]}."""
    return asyncio.run(_auto_suspend.legacy_resume_lmstudio())
