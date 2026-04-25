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
SAFETY_GB = 10
COMFY_URL = "http://localhost:8188"

# Global singletons. asyncio primitives bind to the running loop at first use.
#
# Phase 5 — `gpu_lock` is replaced by `GPUReservation`, a budget-accounted
# condition variable that allows N-way concurrency when summed reservations
# fit within `MemAvailable - SAFETY_GB`. The legacy `gpu_lock` symbol is
# retained as an alias for the reservation's underlying mutex so any
# external test code monkey-patching it keeps working.
class GPUReservation:
    """Budget-accounted reservation primitive.

    Multiple stages may run concurrently iff the sum of their reserved
    budgets fits within `MemAvailable - SAFETY_GB` at the moment of
    acquisition. The condition variable doubles as the mutex protecting
    `resident_gb` and `in_flight`.

    See docs/concurrent-mode-design.md for the full design.
    """

    def __init__(self) -> None:
        self.resident_gb: float = 0.0
        self.cond: asyncio.Condition = asyncio.Condition()
        # job_id -> {"stage", "model", "gb", "started_ts"}
        self.in_flight: dict[str, dict] = {}
        # Phase 5 — planner-managed cache of models the scheduler believes
        # are still GPU-resident from prior stages. `acquire_gpu` consults
        # this when `scheduler.use_planner` is True: a hit means we can
        # skip the reservation entirely (the model is already loaded).
        # Maps model name -> last_used_ts (for LRU-ish eviction).
        self.resident_models: dict[str, float] = {}

    def snapshot(self) -> dict:
        """Diagnostic snapshot — never raises, never blocks."""
        return {
            "resident_gb": round(self.resident_gb, 2),
            "in_flight": [
                {"job_id": k, **v} for k, v in self.in_flight.items()
            ],
            "resident_models": list(self.resident_models.keys()),
        }


GPU = GPUReservation()
# Legacy alias. The condition's underlying mutex *is* the new gpu_lock.
gpu_lock = GPU.cond

paused = asyncio.Event()
paused.set()  # start un-paused


def _ensure_loop_bound() -> None:
    """Rebind asyncio primitives to the running loop if they were created on
    a different (often already-closed) loop. Test suites that call
    `asyncio.run(...)` repeatedly otherwise see "attached to a different
    loop" RuntimeErrors on the next test that touches the scheduler.
    """
    global GPU, gpu_lock, paused
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return
    cond_loop = getattr(GPU.cond, "_loop", None)
    if cond_loop is not None and cond_loop is not running:
        # Preserve resident_gb / in_flight (they're plain Python state).
        old = GPU
        GPU = GPUReservation()
        GPU.resident_gb = old.resident_gb
        GPU.in_flight = dict(old.in_flight)
        gpu_lock = GPU.cond
    paused_loop = getattr(paused, "_loop", None)
    if paused_loop is not None and paused_loop is not running:
        was_set = paused.is_set()
        paused = asyncio.Event()
        if was_set:
            paused.set()

# Monotonic id generator for anonymous reservations (callers without a job_id).
_RESERVATION_ID = itertools.count(1)

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
async def acquire_gpu(
    stage: str,
    model: str,
    safety_gb: int = SAFETY_GB,
    job_id: Optional[str] = None,
):
    """Async context manager reserving GPU budget for one stage.

    Phase 5: replaces the binary `gpu_lock` with a budget-accounted
    `GPUReservation`. Multiple stages may run concurrently iff the sum of
    their reservations fits within `MemAvailable - safety_gb`.

    1. Waits for `paused` to be set (un-paused).
    2. Reserves `stage_budget_gb(stage, model)` against `GPU.resident_gb`,
       waiting on the condition variable until headroom exists.
    3. Yields control to the stage.
    4. On exit, releases the reservation, calls `free_between()`, emits
       `stage_end`, and notifies waiting stages.
    """
    _ensure_loop_bound()
    await paused.wait()

    base_need = stage_budget_gb(stage, model)
    rid = job_id or f"r{next(_RESERVATION_ID)}"
    wait_start = _now()
    blocks = 0

    # Phase 5 — consult the memory_planner BEFORE the budget loop. If the
    # operator opted in (`scheduler.use_planner == True`) AND the required
    # model is already resident from a prior stage, the planner skips the
    # cold-load: we treat the reservation as overhead-only (the model
    # weights are already counted under another stage's reservation).
    planner_hit = False
    if (
        _memory_planner is not None
        and model
        and _planner_enabled()
        and model in GPU.resident_models
    ):
        planner_hit = True
        # Only the OVERHEAD_GB safety pad is "fresh"; the model weights are
        # already accounted for in resident_gb.
        need = float(OVERHEAD_GB)
        await _emit({
            "type": "planner_hit",
            "stage": stage,
            "model": model,
            "saved_gb": round(base_need - need, 2),
            "ts": _now(),
        })
    else:
        need = base_need

    # ---- Reservation acquisition ------------------------------------------------
    async with GPU.cond:
        while True:
            available = _mem_available_gb()
            # Headroom we'd have AFTER this stage's reservation is added.
            projected_resident = GPU.resident_gb + need
            # Budget rule: projected_resident must fit under live MemAvailable
            # minus the configured safety margin. When `resident_gb == 0`
            # (no other stage running) this collapses to the legacy
            # check_budget() formula and is fully backwards compatible.
            ok = projected_resident <= max(0.0, available - float(safety_gb))
            if ok:
                break
            blocks += 1
            await _emit({
                "type": "budget_block",
                "stage": stage,
                "model": model,
                "available_gb": available,
                "needed_gb": round(projected_resident + float(safety_gb), 2),
                "resident_gb": round(GPU.resident_gb, 2),
                "ts": _now(),
            })
            # On the first block, attempt a free pass (releases ComfyUI cached
            # weights without holding the cond mutex while doing network I/O).
            if blocks == 1:
                # Drop the cond so other reservations can release while we
                # talk to ComfyUI. Re-acquire after.
                GPU.cond.release()
                try:
                    await free_between()
                finally:
                    await GPU.cond.acquire()
                continue
            # Otherwise wait for another stage to release (or ~1s tick to
            # re-poll live MemAvailable in case host pressure shifts).
            try:
                await asyncio.wait_for(GPU.cond.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            if blocks >= 30:  # ~30s — bail out and let caller OOM-retry
                await _emit({
                    "type": "oom_retry",
                    "stage": stage,
                    "model": model,
                    "available_gb": available,
                    "needed_gb": round(projected_resident + float(safety_gb), 2),
                    "ts": _now(),
                })
                break

        # Reserve our slice while still holding cond.
        GPU.resident_gb += need
        GPU.in_flight[rid] = {
            "stage": stage,
            "model": model,
            "gb": need,
            "started_ts": _now(),
            "planner_hit": planner_hit,
        }
        # Track this model in the resident-models set so a later stage's
        # planner consultation finds it. Used only when use_planner=True
        # (the dict is harmless to maintain unconditionally).
        if model:
            GPU.resident_models[model] = _now()

    wait_seconds = round(_now() - wait_start, 2)
    await _emit({
        "type": "stage_start",
        "stage": stage,
        "model": model,
        "budget_gb": need,
        "available_gb": _mem_available_gb(),
        "resident_gb": round(GPU.resident_gb, 2),
        "wait_seconds": wait_seconds,
        "ts": _now(),
    })

    # Auto-suspend co-resident services (LM Studio, ComfyUI, Qwen-TTS, ...)
    # via the configured `auto_suspend` list. See auto_suspend.py.
    # NOTE: with concurrent mode active, a second stage starting up will
    # call suspend_all again — the framework is idempotent.
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
        # Release the reservation BEFORE the slow free/resume work, so any
        # blocked stages can wake up and start their own /free pass in
        # parallel. We notify under the cond so wake-ups are race-free.
        async with GPU.cond:
            entry = GPU.in_flight.pop(rid, None)
            released = entry["gb"] if entry else need
            GPU.resident_gb = max(0.0, GPU.resident_gb - released)
            # Planner-managed cache: keep `model` resident if the operator
            # opted in. Otherwise drop it — free_between() below will tell
            # ComfyUI to unload, so the cache must reflect that.
            if not _planner_enabled():
                GPU.resident_models.pop(model, None)
            GPU.cond.notify_all()
        try:
            # Only call /free if the planner doesn't want to keep this
            # model resident. When use_planner=True the whole point is to
            # avoid unloading between stages.
            if not _planner_enabled():
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
            "resident_gb": round(GPU.resident_gb, 2),
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

def _load_scheduler_config() -> dict:
    """Read the `scheduler` block from config.json. Defaults on any error."""
    try:
        c = _config.load_config()
        v = c.get("scheduler") if isinstance(c, dict) else None
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    return {"use_planner": False, "memory_safety_gb": SAFETY_GB}


def _planner_enabled() -> bool:
    """Phase 5 toggle — defaults False so behavior is unchanged for users
    who haven't opted in."""
    try:
        return bool(_load_scheduler_config().get("use_planner", False))
    except Exception:
        return False


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
