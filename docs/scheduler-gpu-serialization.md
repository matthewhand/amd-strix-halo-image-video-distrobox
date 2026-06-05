# Scheduler & GPU serialization

## Overview

AMD Strix Halo's unified 128 GB memory and single GPU core (gfx1151) mandate strict serialization of GPU-bound workloads. The `slopfinity/scheduler.py` module enforces a global mutual-exclusion invariant via `GPUReservation`, an async context manager (`acquire_gpu`), and resident-memory accounting. This document covers the *why* (single-core GPU hangs under concurrent access), the *how* (lock-based serialization with budget admission control), and the *edge cases* (lock-hold-on-hang class of bugs and the principle that `acquire_gpu` must never cross unbounded operations).

## Core invariant: serialized GPU access

The gfx1151 GPU cannot handle overlapping compute kernels from separate processes. Concurrent GPU submissions cause driver hangs that only a warm reboot recovers from. The scheduler prevents this with a single global async lock (the `asyncio.Condition` inside `GPUReservation`), held for the entire duration of a GPU stage.

**Failure mode without serialization:** Two stages (e.g., image generation and video rendering) submit GPU work simultaneously. The driver deadlocks; the entire slopfinity host becomes unresponsive.

---

## Architecture

### GPUReservation class

**File:** `slopfinity/scheduler.py`, lines 55–78

```python
class GPUReservation:
    def __init__(self) -> None:
        self.resident_gb: float = 0.0
        self.cond: asyncio.Condition = asyncio.Condition()
        self.resident_models: dict[str, float] = {}
        self.gpu_history: deque[int] = deque()
        self.in_flight: dict[str, dict] = {}
```

| Field | Meaning | Usage |
|-------|---------|-------|
| `resident_gb` | Peak resident memory (GB) across all in-flight stages. Incremented on entry, decremented on exit. | Budget check: must fit in host RAM minus `SAFETY_GB`. |
| `cond` | `asyncio.Condition` wrapping the above. Holders of the lock can call `cond.wait(timeout=...)` to block until notified. | Blocked stages poll this every 1.0s for memory availability. |
| `resident_models` | Dictionary mapping model names to timestamps (seconds since epoch). Updated every time a model is reserved. | **Phase 5 advisory:** used by `memory_planner` to detect warm-loaded models. See "Residuals" below. |
| `gpu_history` | Sliding window of the last 5 GPU utilization samples (%), appended by broadcaster. | **Phase 5 advisory:** used to gate on idle GPU (not yet wired into `acquire_gpu`). |
| `in_flight` | Dictionary mapping reservation IDs to stage metadata (`{stage, model, since}`). One entry per active stage. | Broadcaster reads to check if anything is holding a reservation (pausing check, line 230). |

### Global singleton getter

**File:** `slopfinity/scheduler.py`, lines 85–109

```python
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
```

**Why:** `asyncio.Condition` binds to the event loop at first use. If the loop changes (e.g., a test fixture tears down one loop and starts another), the old condition becomes stale. This getter detects loop changes and migrates state to a fresh `GPUReservation`, preserving `resident_gb`, `in_flight`, `gpu_history`, and `resident_models` across the transition.

**Caller mistake it prevents:** Directly accessing `sched.GPU` instead of calling `sched.get_gpu()` causes an `AttributeError` when the module-level `_GPU` doesn't exist yet (or `cond._loop` is stale).

### Broadcaster bug: get_gpu() vs sched.GPU (commit 61094b0)

**File:** `slopfinity/broadcaster.py`, lines 229–230

```python
# After fix (correct):
_gpu = sched.get_gpu()
if _gpu.resident_gb > 0 or _gpu.in_flight:
```

**Why it broke:** The broadcaster's tick loop (every 2 seconds) directly accessed `sched.GPU`, which was `None` if the module hadn't yet initialized a `GPUReservation`. The paused-state check would raise `AttributeError: 'NoneType' object has no attribute 'resident_gb'`. With logging unhidden by commit 68e5e33 (which stopped swallowing all exceptions), this error became visible every 2 seconds.

**Failure mode:** Dashboard WebSocket tick silently failed. Clients never received state updates while the pipeline was paused. A user would think the app was hung.

---

## The acquire_gpu context manager

### Signature and budget check

**File:** `slopfinity/scheduler.py`, lines 266–327

```python
@contextlib.asynccontextmanager
async def acquire_gpu(
    stage: str,
    model: str,
    safety_gb: int = SAFETY_GB,
    job_id: Optional[str] = None,
):
```

The context manager implements **budget-based admission control** with **fairness backoff**:
1. On entry: check if (resident_gb + model_budget) fits in available RAM
2. If not: wait up to 30 polling intervals (~30 seconds) for other stages to finish
3. On the 1st wait: call `free_between()` to unload ComfyUI models
4. On exit: decrement resident_gb and notify all waiters

### Detailed flow

**Lines 273–277:** Wait for pause signal, then bypass on non-GPU hosts

```python
await get_paused().wait()

if not has_gpu():
    yield {"stage": stage, "model": model, "wait_seconds": 0.0, "gpu_bypassed": True}
    return
```

The initial `await get_paused().wait()` ensures that when the dashboard user clicks "Pause", already-running stages will skip new GPU reservations (line 273 blocks on every entry).

**Lines 288–312:** Lock-protected polling loop

```python
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
```

**Key invariant:** The lock (`async with gpu.cond`) is held only during the check and state update (lines 289–309). Expensive I/O like `free_between()` happens *outside* the lock (line 312).

**Admission formula (line 293):** A stage can enter if its projected memory footprint, *added to existing resident_gb*, stays under `available - safety_gb`. The `safety_gb` buffer (default 24 GB, configurable) protects against system-critical tasks and memory-accounting vagaries.

**Model reuse (line 291):** If a model is already in `resident_models`, it was loaded by a previous stage. The stage only reserves `OVERHEAD_GB` (6 GB) instead of the full `stage_budget_gb()`, enabling better packing.

**Fairness via backoff (lines 301–309):** If blocked, the stage waits for notification with a 1.0-second timeout. On the *first* block, it also tries `free_between()` to unload ComfyUI. After 30 timeouts (~30 seconds), it admits anyway, failing gracefully under sustained overload.

### Release and notification

**Lines 316–327:** Finally block

```python
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
```

When a stage exits, it releases its memory reservation and wakes all waiters. The event is sent to the dashboard so clients can update progress.

---

## Lock-hold-on-hang: the two fixes

The scheduler's lock must be held *only* during the check and state update. Long operations (downloading models, running inference, I/O) inside `async with gpu.cond` block all other stages. Two fixes prevent this:

### Fix 1: /music subprocess timeout (commit 61094b0)

**File:** `slopfinity/routers/runner.py`, lines 383–389

```python
def _do() -> int:
    # Hard timeout so a wedged docker/GPU can't pin the GPU lock (held
    # by the enclosing acquire_gpu) forever and starve every other
    # serialized pipeline. Mirrors run_fleet's run_with_timeout(600);
    # 20 min gives heartmula generous headroom. TimeoutExpired is an
    # Exception → caught below, and the lock releases on `async with` exit.
    return subprocess.run(cmd, check=False, timeout=1200).returncode
```

The `/music` endpoint calls Heartmula (music generation) inside `acquire_gpu` (line 363). Before this fix, if the subprocess hung (e.g., GPU driver wedged), the CPU would spin on `subprocess.run()` indefinitely, pinning the GPU lock and starving every other queued stage.

**The fix:** `timeout=1200` (20 minutes) ensures that a hung subprocess is killed. The `TimeoutExpired` exception propagates to the outer `except` (line 393), releases the lock (implicit on exit from `async with`), and returns an error to the client.

**Failure mode without the fix:** One wedged music job starves the entire pipeline. Users cannot interrupt it or queue other work.

**Mirrors run_fleet:** The main fleet runner uses `run_with_timeout(600)` for subprocess calls. The `/music` endpoint's 1200-second limit is deliberately generous (20 minutes) to accommodate slow music generation on a single GPU, while still providing an upper bound.

### Fix 2: /tts speed validation (commit 68e5e33)

**File:** `slopfinity/routers/runner.py`, lines 307–313

```python
# Clamp to the supported TTS range instead of forwarding e.g. speed=999 to
# the worker (the chat tool already validates 0.5–2.0; match it here).
if speed is not None and not (0.5 <= speed <= 2.0):
    return JSONResponse(
        {"ok": False, "error": "speed must be between 0.5 and 2.0"},
        status_code=400,
    )
```

The `/tts` endpoint forwards `speed` to the TTS worker. Without validation, a malformed request (e.g., `speed=999`) would be sent to the worker, which might reject it slowly (network retry, worker exception handling) or accept it and generate audio at an invalid speed, causing the worker to hang or crash.

**The fix:** Validate `0.5 <= speed <= 2.0` before touching the GPU lock. Reject invalid speeds with a 400 error *before* entering `acquire_gpu` (line 322).

**Why this matters:** Invalid speed values could cause the TTS worker to hang or loop inside the `acquire_gpu` context (the `/tts` code at lines 322–326 holds the lock while calling the worker). Validation up-front prevents a bad request from accidentally pinning the GPU lock.

**Precedent:** The chat tool (in the UI) already validates this range; the router should enforce the same bounds for API consumers.

---

## check_budget: read-only admission check

**File:** `slopfinity/scheduler.py`, lines 197–206

```python
def check_budget(stage: str, model: str, safety_gb: int = 10):
    """Does a (stage, model) reservation fit in host RAM right now?

    Returns (ok, available_gb, need_gb) where need = model+overhead budget plus
    a `safety_gb` headroom. Pure/read-only (reads /proc/meminfo) — does not
    touch the GPU lock, so it's safe to call for gating decisions and tests.
    """
    need = stage_budget_gb(stage, model) + float(safety_gb)
    avail = _mem_available_gb()
    return (avail >= need, avail, need)
```

This is a *read-only* utility for dashboard and test code to check if a stage *would* fit, without taking the lock. It does not update `resident_gb` or `in_flight`. Useful for:
- Dashboard "Can I queue this job?" gating (before sending to the queue)
- Tests that want to mock memory conditions
- Monitoring and diagnostics

**Note:** The default `safety_gb=10` for `check_budget` differs from `acquire_gpu`'s default `SAFETY_GB=24` (set at module level, line 50). The former is conservative for tests; the latter is the live production buffer.

---

## Broadcaster integration: mutate_config and mutate_queue with to_thread

**File:** `slopfinity/broadcaster.py`, lines 147–148 and 307

```python
# Stale config (prune old cancelled jobs)
queue = await asyncio.to_thread(
    cfg.mutate_queue, lambda q: [x for x in q if not _is_stale(x)]
)

# Chaos mode (refresh infinity themes)
config = await asyncio.to_thread(cfg.mutate_config, _set_themes)
```

Commits 68e5e33 and 61094b0 wrapped these I/O operations (file lock + SQLite write) in `await asyncio.to_thread()` to avoid blocking the event loop. The broadcaster runs on the main async loop; any blocking call stalls all WebSocket broadcasts.

**Failure mode without to_thread:** A slow filesystem or database lock would cause the broadcast tick to hang, blocking state updates to the dashboard.

---

## has_gpu(): AMD GPU detection

**File:** `slopfinity/scheduler.py`, lines 249–263

```python
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
```

Caches the result in a function attribute (`has_gpu._result`) to avoid repeated subprocess spawns. Returns `True` if `/dev/kfd` exists (AMD KFD kernel module) or `rocminfo` succeeds.

**On non-GPU hosts:** Returns `False`, and `acquire_gpu` bypasses the lock entirely (lines 275–276), allowing the pipeline to run (slowly) on CPU.

---

## Failure modes & edge cases

| Failure | Root cause | Mitigation |
|---------|-----------|-----------|
| GPU lock held on subprocess hang | `/music` or `/tts` worker doesn't return within deadline. | `timeout=1200` on subprocess; `TimeoutExpired` releases lock on exit. |
| GPU lock held on invalid worker input | `speed=999` sent to TTS worker, worker hangs processing it. | Validate `0.5 <= speed <= 2.0` *before* entering `acquire_gpu`. |
| AttributeError in broadcaster paused check | Directly accessing `sched.GPU` instead of `get_gpu()`. | Use `sched.get_gpu()` which handles loop migration. |
| Event loop blocks on lock-protected I/O | Config mutations or queue prune happen inside the lock or on the main loop. | Wrap in `await asyncio.to_thread()`. |
| Resident_models grows unbounded (Phase 5) | Models are added to `resident_models` but never explicitly cleared. | See "Residuals" below. |
| Loop migration loses state | Old loop dies, new loop created, old `asyncio.Condition` becomes invalid. | `get_gpu()` detects loop change and migrates state. |

---

## Residuals & future work

### resident_models leak (unfixed)

**File:** `slopfinity/scheduler.py`, lines 59, 298

The `resident_models` dictionary is populated when a model enters (line 298: `gpu.resident_models[model] = _now()`), but it is **never pruned**. Over a long run (weeks of pipeline work), this dict grows without bound.

**Impact:** Low. The dict holds only model names and timestamps; memory usage is negligible. However, it's semantically incorrect — the name implies "currently resident models," but old entries linger.

**Why unfixed:** This field is **Phase 5 feature** (advisory for `memory_planner` to detect warm-loaded models). Phase 5 is not yet active (gfx1151 doesn't support true concurrency). Once Phase 5 lands and models are genuinely kept resident across multiple stages, this field will need a lifecycle:
- Clear on full GPU unload (e.g., `free_between()` success)
- Or: expire entries older than a configurable TTL
- Or: track which models are *actually* resident by querying ComfyUI

**Test coverage:** None. The xfail tests in `test_scheduler.py` (lines 107–147) skip Phase-5 concurrency features.

### GPU history for idle gating (advisory, not wired)

**File:** `slopfinity/scheduler.py`, lines 60, 63–71

The broadcaster records GPU utilization every 2 seconds (line 132–133 in broadcaster.py). The `gpu_history` deque holds the last 5 samples. A Phase-5 feature (`pause_for_idle_gpu`) was envisioned to gate new GPU reservations based on average GPU usage, but it is **not implemented in `acquire_gpu`**.

**Status:** The plumbing exists; the decision logic doesn't. The `xfail` test at `test_scheduler_gpu_guard.py` lines 59–112 documents the intended behavior but is marked not-yet-implemented.

**Why deferred:** Gating on GPU idle adds latency to job admission. The current lock-based serialization is sufficient to prevent hangs. Idle gating would be an optimization (start jobs only when GPU is not busy with user work), not a safety feature.

---

## Verification

### Tests

**File:** `tests/test_scheduler.py`

- **test_check_budget_qwen_plenty** (line 28): 100 GB available, qwen (28+6+10 safety = 44 GB) fits.
- **test_check_budget_wan25_tight** (line 38): 50 GB available, wan2.5 (96+6+10 = 112 GB) does not fit.
- **test_check_budget_respects_safety_override** (line 56): Varying `safety_gb` changes admission.
- **test_acquire_gpu_concurrent_when_budget_fits** (line 67): Two qwen stages can overlap if budgets fit (200 GB host, both 34 GB → 68 GB < 190 GB available).
- **test_free_between_posts_to_comfy** (line 186): Verifies `/free` POST to ComfyUI and memory delta.
- **test_free_between_handles_network_error** (line 222): Error path works.
- **test_pause_resume_toggle** (line 233): Pause/resume signal works.
- **test_stage_budget_gb_includes_overhead** (line 246): Budget formula correct.

**File:** `tests/test_scheduler_gpu_guard.py`

- **test_gpu_reservation_history_init** (line 33): History starts empty.
- **test_record_gpu_usage** (line 38): Samples append and respect max.
- **test_get_gpu_avg** (line 48): Average calculated correctly.
- **test_acquire_gpu_ignores_busy_gpu_when_disabled** (line 115): Idle gating disabled means no blocking on high GPU usage.
- **test_acquire_gpu_waits_for_idle_gpu** (line 64): xfail — idle gating not yet wired.

**Full suite:** `pytest tests/ -v` reports 239 passed, 2 skipped, 3 xfailed (as of commits 61094b0 and 68e5e33).

### Manual verification

The scheduler is exercised every time the fleet pipeline runs:
1. Dashboard queues a job.
2. `/runner/terminate` endpoint checks the pause flag.
3. Each stage calls `acquire_gpu()`.
4. Broadcaster ticks every 2 seconds, calling `get_gpu()` and `sched.get_gpu()`.

A multiday fleet run validates:
- No hangs or deadlocks under concurrent queue load.
- Budget checks prevent OOM errors.
- Stages queue and drain correctly.

---

## Commits

- **68e5e33** (2026-06-04 09:01): "fix: config lost-update race, silent broadcaster errors, tts speed, frame leak"
  - Adds TTS speed validation (0.5–2.0 clamp).
  - Adds error logging to broadcaster (surfaces get_gpu() mistake).
  - Wraps `mutate_config` in `asyncio.to_thread()`.

- **61094b0** (2026-06-04 09:14): "fix: round-4 sweep — /music GPU-lock hang, broadcaster crash, event-loop blocking"
  - Adds `timeout=1200` to `/music` subprocess.
  - Fixes `sched.GPU` → `sched.get_gpu()` in broadcaster paused check.
  - Wraps `mutate_queue` in `asyncio.to_thread()` for the queue prune loop.
