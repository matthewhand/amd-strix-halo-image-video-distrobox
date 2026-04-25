# Concurrent Mode — Design Note

**Status:** UI toggle is currently **locked** (disabled). This document tracks the design that needs to land before unlocking.

## Why it's locked today

The `concurrent` toggle in the pipeline popup is a no-op with foot-gun potential:

1. **No reservation accumulator.** `slopfinity/scheduler.py::check_budget` weighs only the *one* stage being acquired against `MemAvailable`. Two stages can both pass the check and then collide on the GPU.
2. **`gpu_lock` is a binary `asyncio.Lock`, not a semaphore.** Even if the budget logic allowed concurrency, the lock itself serializes — concurrent never actually overlaps.
3. **The fleet runner bypasses the scheduler entirely.** `run_philosophical_experiments.py` shells out via `docker run --rm` directly without calling `acquire_gpu`, so today's serialization is theatrical for fleet jobs.
4. **The `concurrent` flag persists into `queue.json` items but no code branches on it.** It's a stored bool with zero consumers.

Letting users enable it without those gaps closed risks silent OOMs on the gfx1151 (128 GB unified memory shared with the host) — a single LTX-2.3 + Qwen-Image overlap can easily exceed budget.

## Target architecture

Replace the binary lock with a **budget-accounted condition variable**. Stages reserve their estimated peak GB; new stages can run concurrently iff their addition fits within `MemAvailable - SAFETY_GB`.

### Proposed `slopfinity/scheduler.py` changes

```python
class GPUReservation:
    def __init__(self):
        self.resident_gb: float = 0.0
        self.cond = asyncio.Condition()  # also serves as the mutex
        self.in_flight: dict[str, float] = {}  # job_id -> reserved_gb (for diagnostics)

GPU = GPUReservation()

@contextlib.asynccontextmanager
async def acquire_gpu(stage: str, model: str, job_id: str, safety_gb: int = SAFETY_GB):
    await paused.wait()
    need = stage_budget_gb(stage, model)
    async with GPU.cond:
        # Wait until reserving `need` would still leave SAFETY_GB headroom against
        # the LIVE MemAvailable reading (not a snapshot — re-poll on each wake).
        while GPU.resident_gb + need > _mem_available_gb() - safety_gb:
            await _emit({"type": "budget_block", ...})
            # Caller may want to call free_between() here on first block
            await GPU.cond.wait()  # released by another stage's exit
        GPU.resident_gb += need
        GPU.in_flight[job_id] = need

    try:
        await _emit({"type": "stage_start", "stage": stage, "model": model, "resident_gb": GPU.resident_gb, ...})
        yield
    finally:
        async with GPU.cond:
            GPU.resident_gb -= GPU.in_flight.pop(job_id, need)
            GPU.cond.notify_all()
        await _emit({"type": "stage_end", "stage": stage, "model": model, "resident_gb": GPU.resident_gb, ...})
```

### Properties

- **N-way concurrency** emerges naturally when budgets sum cleanly (e.g. Qwen image stage 22 GB + Heartmula audio stage 14 GB = 36 GB → fits next to most things).
- **Single-way fallback** when they don't (LTX 72 GB + Qwen 22 GB + safety 10 GB = 104 GB — fits within 128 GB but contests with host RAM, so live `_mem_available_gb()` will gate it).
- **Cannot OOM within budget accuracy.** The `STAGE_BUDGETS` table is the trust boundary — if it underestimates, the live `MemAvailable` re-poll catches up to reality on each wake.
- **Backwards compatible:** `concurrent=false` callers still get serialized behavior because they're still gated by the same budget check; without other stages running, `resident_gb=0` and they always pass.

### Failure modes to design for

| Failure | Mitigation |
|---|---|
| Stage exits abnormally without releasing reservation | Use `asyncio.shield(...)` and a `try/finally` around the `yield`; on cancellation the `finally` still runs |
| Budget table underestimates | Per-wake live `MemAvailable` re-poll; `safety_gb` static padding |
| Deadlock (all stages blocked, none can finish to release) | The first stage that fits acquires; budget table guarantees at least one stage always fits in 128 GB |
| Fleet runner bypasses | See "Step 2" below |

## Step 2 — wire the fleet runner through the same gate

The fleet runner (`run_philosophical_experiments.py`) launches stages via `docker run --rm`. To honor reservations:

**Option A: HTTP scheduler hook.** Fleet runner POSTs to a new dashboard endpoint `POST /scheduler/reserve` (body: `{stage, model, job_id}`); response holds connection open until budget is granted. On stage end, fleet runner POSTs `/scheduler/release`. Pro: simple, no shared state. Con: dashboard must be running.

**Option B: file-lock based.** Fleet runner uses `flock` on a shared lockfile in `/workspace/.gpu-reservations/<job_id>` and writes its budget. The dashboard scheduler reads these lockfiles to compute live `resident_gb`. Pro: dashboard-independent. Con: race-prone, harder to debug.

**Recommend A.** The fleet runner is already useless without the dashboard, so the dependency is fine.

## Step 3 — UI unlock

Once steps 1 and 2 land:
1. Remove `disabled` + `opacity-50` + tooltip from `#concurrent-on` in `slopfinity/templates/index.html`
2. Replace the `experimental · locked` badge with a small `experimental` badge (still warning the user, no longer blocked)
3. Optionally add a Diagnostics-tab readout showing `resident_gb / MemAvailable` so users can see the budget in real time

## Open questions

- Should the budget table be user-editable (Settings → Pipeline Advanced)? Probably yes for power users, but ship with sensible defaults.
- Should we expose `MAX_CONCURRENCY` as a hard cap regardless of budget? Useful for I/O-bound stages where memory isn't the bottleneck.
- Should `concurrent=true` jobs share the LLM auto-suspend logic from PR #40? Two GPU stages can't both have the host LLM suspended/unsuspended cleanly.

## Tracking

When you start implementing, branch from `main` as `feat/concurrent-mode-budget-semaphore`. Land in stages:
1. PR: replace `gpu_lock` with `GPUReservation`, no fleet-runner change yet
2. PR: scheduler `/reserve` and `/release` endpoints + fleet runner integration
3. PR: UI unlock + diagnostics tile
