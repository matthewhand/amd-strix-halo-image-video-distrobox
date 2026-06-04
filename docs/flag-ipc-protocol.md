# Flag-file IPC protocol

## Overview

The file-based IPC between the containerised slopfinity dashboard (port 9099, FastAPI) and the host-side `run_fleet.py` orchestrator uses three filesystem flags in `OUTPUT_DIR` (typically `comfy-outputs/experiments`). Because the dashboard runs in a container with PID-namespace isolation and restricted security profile, it cannot directly SIGTERM the host process; flags provide an asynchronous, reliable signalling mechanism that survives process restarts and crosses the container boundary.

The three flags are:

| Flag | Written by | Read by | Purpose | File location |
|------|-----------|---------|---------|---------------|
| `terminate.flag` | `/runner/terminate` endpoint | `run_fleet.py` main loop | Hard stop; exit cleanly | `{OUTPUT_DIR}/terminate.flag` |
| `pause.flag` | `/queue/pause` endpoint | `run_fleet.py` main loop | Soft stop; pause iter starts | `{OUTPUT_DIR}/pause.flag` |
| `cancel.flag` | `/inject`, `/cancel-all`, `/queue/cancel` endpoints | `run_fleet.py` chain boundaries | Abort in-flight iter | `{OUTPUT_DIR}/cancel.flag` |

All flag files are written as text files containing a Unix timestamp (the time they were written). Files are written via `open(..., "w")` with no explicit fsync, relying on Python's buffered I/O and OS page-cache semantics — reads happen on the next call to `os.path.exists()` or `os.path.getmtime()`, which are atomic filesystem operations on Linux.

---

## Terminate flag: hard-stop mechanism

### Writing the flag

**File:** `slopfinity/routers/runner.py:189–228`

```python
@router.post("/runner/terminate")
async def runner_terminate():
    import signal
    flag_path = os.path.join(EXP_DIR, "terminate.flag")
    flag_written = False
    try:
        with open(flag_path, "w") as f:
            f.write(str(time.time()))
        flag_written = True
    except Exception:
        pass
    pids = _find_pids_by_cmdline("run_fleet.py")
    killed: list[int] = []
    # ... SIGTERM escalation ...
    return { "ok": True, "flag_written": flag_written, ... }
```

The endpoint performs two actions in sequence:
1. **Write `terminate.flag`** with the current timestamp (lines 194–197). This is the canonical signal; the SIGTERM fallback (lines 200–210) is best-effort and unreliable across deployment topologies (namespace/container restrictions).
2. **SIGTERM then SIGKILL** any running `run_fleet.py` processes (if accessible).

The flag is the reliable path when the dashboard cannot directly kill the runner.

### Reading the flag: startup cleanup

**File:** `run_fleet.py:1575–1584`

```python
try:
    _stale_term = os.path.join(OUTPUT_DIR, "terminate.flag")
    if os.path.exists(_stale_term):
        os.remove(_stale_term)
        print("[FLEET] cleared stale terminate.flag from a previous run", flush=True)
except OSError:
    pass
```

**Root cause of bug (commit ddab557):** Before the fix, `terminate.flag` was never cleared on startup. If `run_fleet.py` was restarted after the user had terminated it via the dashboard, the stale flag remained on disk. On the next iter, the runner would see the 1–2-day-old flag and exit immediately — un-restartable without manual file deletion.

**Fix:** Lines 1575–1584 explicitly remove any stale `terminate.flag` at startup, before entering the main loop. This is a one-time best-effort cleanup; if removal fails (e.g., permissions), execution continues.

### Reading the flag: main loop gate

**File:** `run_fleet.py:1585–1599`

```python
while True:
    _terminate_flag = os.path.join(OUTPUT_DIR, "terminate.flag")
    if os.path.exists(_terminate_flag):
        print(f"[FLEET] terminate.flag detected at {_terminate_flag} — exiting", flush=True)
        update_state(mode="Terminated", step="terminate.flag",
                     video=0, total=0)
        break
```

At the top of each iteration, `run_fleet.py` checks for the flag (line 1595). If present, it breaks from the main loop and exits cleanly after updating the dashboard state. The check happens **before** popping a new task, so the active iter (if any) finishes naturally — it is *not* an abort mid-flight (see cancel.flag for that).

**Ordering guarantee:** The cleanup (lines 1575–1584) ensures the runner can restart. The loop-top gate (line 1595) ensures termination is detected promptly (within one iter, typically 1–60s depending on workload).

### Clearing the flag

**File:** `slopfinity/routers/runner.py:230–238`

```python
@router.post("/runner/terminate-clear")
async def runner_terminate_clear():
    flag_path = os.path.join(EXP_DIR, "terminate.flag")
    existed = os.path.exists(flag_path)
    try:
        os.remove(flag_path)
    except FileNotFoundError:
        pass
    return {"ok": True, "existed": existed, "flag_path": flag_path}
```

The dashboard endpoint `/runner/terminate-clear` removes the flag so the user can restart `run_fleet.py`. It is idempotent (ignores `FileNotFoundError`).

---

## Pause flag: soft-stop mechanism

### Writing the flag

**File:** `slopfinity/routers/queue.py:213–226`

```python
@router.post("/queue/pause")
async def queue_pause():
    """Pause new iter starts in run_fleet. Writes pause.flag in EXP_DIR;
    the orchestrator polls the flag and skips its iter loop body while
    it exists. The currently-running iter (if any) finishes naturally —
    pause is a SOFT stop, not a kill."""
    try:
        with open(os.path.join(EXP_DIR, "pause.flag"), "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "paused": True}
```

The endpoint simply writes the flag with a timestamp (lines 221–223). No queue operations, no locks — it is a fire-and-forget write.

### Reading the flag: main loop gate

**File:** `run_fleet.py:1600–1610`

```python
_pause_flag = os.path.join(OUTPUT_DIR, "pause.flag")
if os.path.exists(_pause_flag):
    update_state(mode="Paused", step="User-paused (no new iters)",
                 video=0, total=0)
    time.sleep(5)
    continue
```

After checking for `terminate.flag`, the runner checks for `pause.flag` (line 1606). If present, it updates state, sleeps for 5 seconds, and continues the loop without processing a new task. The 5-second cadence balances responsiveness (the user sees "paused" state within ~5s) vs CPU usage (no busy-polling).

**Why soft stop:** The current iter (if any) continues running — the check happens at the top of the loop, *before* popping and starting a new task. This allows the user to pause gracefully without aborting in-flight work.

### Clearing the flag

**File:** `slopfinity/routers/queue.py:228–237`

```python
@router.post("/queue/resume")
async def queue_resume():
    """Remove pause.flag — fleet returns to its iter loop on next poll."""
    flag = os.path.join(EXP_DIR, "pause.flag")
    try:
        if os.path.exists(flag):
            os.remove(flag)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "paused": False}
```

The `/queue/resume` endpoint removes the flag (lines 232–234). On the next poll (within 5s), the runner returns to its iter loop.

**Verification:** Tests `test_pause_sets_flag` and `test_resume_clears_flag` in `tests/test_server_queue.py:61–73` verify endpoint success (status 200).

---

## Cancel flag: mid-flight abort with mtime-gating

Cancel is the most intricate flag: it must abort an in-flight iter without stopping the paused or pending queue, and it must ignore stale flags from prior iters.

### Writing the flag: three entry points

The flag is written **only** when cancelling the **active (working) item**, not when cancelling pending items. This is enforced at three endpoints:

#### 1. `/inject` with `terminate=1` (cancel all + inject)

**File:** `slopfinity/routers/queue.py:51–211`

```python
async def inject(
    prompt: str = Form(...),
    terminate: str = Form(...),
    ...
):
    terminate = _truthy(terminate)
    # ... disk guard ...
    # ... task setup ...
    
    _cancel_ts: list = []
    
    def _do_inject(q):
        if terminate:
            # Mark every pending and in-flight item cancelled
            now_ts = time.time()
            for item in q:
                if item.get("status") in (None, "pending", "working"):
                    item["status"] = "cancelled"
                    item["cancelled_ts"] = now_ts
                    item["infinity"] = False
            _cancel_ts.append(now_ts)
        # ... reorder pending ...
        return working + pending + done + cancelled
    
    await asyncio.to_thread(cfg.mutate_queue, _do_inject)
    if _cancel_ts:
        try:
            with open(os.path.join(EXP_DIR, "cancel.flag"), "w") as f:
                f.write(str(_cancel_ts[0]))
        except Exception:
            pass
    return {"status": "ok"}
```

**Key detail:** The flag is written **after** `mutate_queue()` returns (lines 205–210). The queue commit locks the queue, marks every `pending` or `working` item as cancelled, then unlocks. Only *then* is the flag written. This ordering ensures the runner never sees the flag before the queue update — preventing a race where the runner aborts mid-iter but the queue still shows the item as `working` (not `cancelled`).

The timestamp written is the same one used to mark items as cancelled in the queue (line 177), establishing a consistent point-in-time for all cancellations in this batch.

#### 2. `/cancel-all` (cancel everything)

**File:** `slopfinity/routers/queue.py:244–270`

```python
@router.post("/cancel-all")
async def cancel_all():
    """Mark every pending or in-flight queue item as cancelled and signal the fleet runner."""
    now_ts = time.time()
    counted: list = []
    
    def _cancel_all(q):
        for item in q:
            if item.get("status") in (None, "pending", "working"):
                item["status"] = "cancelled"
                item["cancelled_ts"] = now_ts
                item["infinity"] = False
                counted.append(1)
        return q
    
    await asyncio.to_thread(cfg.mutate_queue, _cancel_all)
    try:
        with open(os.path.join(EXP_DIR, "cancel.flag"), "w") as f:
            f.write(str(now_ts))
    except Exception:
        pass
    return {"status": "ok", "cancelled": len(counted)}
```

Same pattern: mutate under lock (lines 255–262), then write the flag outside the lock with the same timestamp (lines 265–269).

#### 3. `/queue/cancel` (selective, active-only)

**File:** `slopfinity/routers/queue.py:272–312`

```python
@router.post("/queue/cancel")
async def queue_cancel(data: dict = Body(...)):
    """Cancel a single queue item by ts. Only when the *actively running*
    (`working`) item is cancelled do we write a cancel.flag so the fleet runner
    aborts the in-flight iter. Cancelling a merely-pending item must NOT abort
    whatever is currently running."""
    target_ts = data.get("ts")
    if target_ts is None:
        return JSONResponse({"ok": False, "error": "missing ts"}, status_code=400)
    found: list = []
    was_working: list = []
    
    def _cancel_one(q):
        for item in q:
            if item.get("ts") == target_ts and item.get("status") in (None, "pending", "working"):
                if item.get("status") == "working":
                    was_working.append(1)
                item["status"] = "cancelled"
                item["cancelled_ts"] = time.time()
                item["infinity"] = False
                found.append(1)
                break
        return q
    
    await asyncio.to_thread(cfg.mutate_queue, _cancel_one)
    if not found:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    # Abort the in-flight iter ONLY when the running item is the one being
    # cancelled — not for any pending item earlier in the queue. Written after
    # the commit so the fleet never sees the flag before the queue update.
    if was_working:
        try:
            with open(os.path.join(EXP_DIR, "cancel.flag"), "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass
    return {"ok": True}
```

**Critical detail (lines 306–311):** The flag is written **only if the cancelled item was `working`** (line 306). If the cancelled item was merely `pending`, no flag is written — the active iter continues uninterrupted. This prevents the user from accidentally aborting the running job by cancelling something in the queue below it.

**Verification:** Tests in `tests/test_server_queue.py:154–173` verify this scoping:
- `test_cancel_pending_does_not_write_flag` (line 154–163): cancelling pending item (ts 2.0) when working item (ts 1.0) is active does not write flag.
- `test_cancel_working_writes_flag` (line 165–173): cancelling working item writes flag.

### Reading the flag: chain-boundary abort with mtime-gating

**File:** `run_fleet.py:1904–1920`

```python
for c_idx in range(1, _n_chains + 1):
    # Honour a mid-flight cancel of the running item: the dashboard
    # writes cancel.flag when the active (`working`) item is
    # cancelled. Check at each chain boundary so a long multi-chain
    # render stops promptly instead of finishing. mtime-gated against
    # this iter's start so a stale flag from a prior iter is ignored.
    _cf = os.path.join(OUTPUT_DIR, "cancel.flag")
    try:
        if os.path.exists(_cf) and os.path.getmtime(_cf) >= _iter_started_ts:
            print(
                f"[FLEET] cancel.flag — aborting iter v{v_idx} at chain {c_idx}/{_n_chains}",
                flush=True,
            )
            _iter_cancelled = True
            break
    except OSError:
        pass
```

The check occurs **at each chain boundary** (after every video segment is rendered, line 1904). The logic:

1. **File exists?** Check with `os.path.exists()` (line 1912).
2. **Modification time >= iter start?** Check with `os.path.getmtime(_cf) >= _iter_started_ts` (line 1912).

The **mtime-gating** (line 1912) is the critical guard against stale flags. At line 1615, the runner records `_iter_started_ts = time.time()` at the *start* of the iter. If `cancel.flag` exists but was written *before* this iter started (e.g., from a cancelled prior job), the mtime will be older and the condition fails — the flag is ignored.

**Root cause of original bug:** Before commit ddab557, `cancel.flag` was written but never read, so cancelling the active item had no effect on the runner — it continued rendering regardless. Only the queue status was marked cancelled, which prevented requeue but did not abort work.

**Fix:** Lines 1910–1920 read the flag at each chain boundary and unwind via `_IterCancelled` exception.

### Unwinding on cancel: _IterCancelled exception

**File:** `run_fleet.py:38–41`

```python
class _IterCancelled(Exception):
    """Raised to unwind the current iter when the user cancels the running item
    mid-flight (dashboard writes cancel.flag). Handled distinctly from real
    errors so a cancel isn't logged/recorded as a failure."""
```

When the mtime-gated check succeeds, the runner sets `_iter_cancelled = True` and breaks from the chain loop (line 1917–1918). Later, if no chains have rendered, the code explicitly raises `_IterCancelled` (line 2061):

```python
if _iter_cancelled and not chain_vids:
    # Cancelled before any chain finished — nothing to finalize.
    raise _IterCancelled(f"iter v{v_idx} cancelled (no chains rendered)")
```

**Exception handling:** The exception is caught at line 2142 and handled distinctly from errors:

**File:** `run_fleet.py:2141–2151`

```python
except Exception as e:
    if isinstance(e, _IterCancelled) or _iter_cancelled:
        # User-initiated cancel of the running item — not a failure.
        # The queue already marks it cancelled (so the archive path
        # below won't requeue it); just update state and move on.
        print(f"[FLEET] ⏹ {e}", flush=True)
        try:
            update_state(mode="Cancelled", step="User cancelled",
                         video=v_idx, total=1000, prompt=p)
        except Exception:
            pass
    else:
        iter_failed = True
        # ... error handling ...
```

If `_IterCancelled` is caught, the iter is marked with `mode="Cancelled"` in the dashboard state (line 2148), and the runner does *not* increment `iter_failed` — the record is audited as cancelled, not failed.

### Chain boundary ordering and queue consistency

The cancel flag is checked **at each chain boundary**, not just once at iter start. This is important for:

1. **Responsiveness:** Long multi-chain renders (15–50 chains, each taking 30–600s) stop promptly instead of finishing the entire chain set.
2. **Partial output:** If the user cancels after 3 of 10 chains have rendered, those 3 chains are muxed into a final partial clip (lines 2057–2098) — the user keeps what was produced before the cancel.

### Queue record management under cancel

When the runner finishes an iter (success, failure, or cancel), it reconciles the in-flight `working` sentinel with a `done` archive record. The archive span is inside `cfg.queue_lock()`:

**File:** `run_fleet.py:2184–2263`

```python
try:
    if _task_opts.get("_seed_prompt"):
        with cfg.queue_lock():
            q_now = cfg.get_queue()
            # Pull the in-flight row matching this iter to read the user's
            # most recent toggle / cancel state
            orig_ts = _task_opts.get("_orig_task_ts")
            live_record = None
            kept = []
            for it in q_now:
                if (
                    it.get("ts") == orig_ts
                    and it.get("status") in ("working", "cancelled")
                    and live_record is None
                ):
                    live_record = it
                    continue  # drop the in-flight sentinel
                kept.append(it)
            q_now = kept
            
            # ... reconcile flags ...
            cancelled_mid_flight = live.get("status") == "cancelled"
            
            # 1) Done archive
            q_now.append({
                "prompt": _task_opts["_seed_prompt"],
                "status": "done",
                "succeeded": not iter_failed,
                "ts": orig_ts or _iter_started_ts,
                ...
            })
            
            # 2) Requeue — only if the user still wants infinity AND
            # they didn't cancel mid-flight.
            if eff_infinity and not cancelled_mid_flight:
                # re-append pending task
                ...
            
            cfg.save_queue(q_now)
```

Key points:

- **Line 2198:** The `live_record` is the `working` sentinel re-read from the queue. If the user clicked cancel via `/queue/cancel` *after* the iter started, the `live_record.status` will be `"cancelled"`.
- **Line 2238:** `cancelled_mid_flight = live.get("status") == "cancelled"` captures this.
- **Line 2263:** Requeue only happens if `eff_infinity and not cancelled_mid_flight` — a cancelled item is not requeued, even if infinity was enabled.

This ensures the queue reflects the user's cancel decision accurately. The queue lock prevents the runner and the dashboard from both mutating the queue simultaneously during this reconcile.

---

## Ordering guarantees and race conditions

### Atomic transitions

Because `cancel.flag` is written *after* the queue is committed (lines 205–210, 265–269, 307–311 in queue.py), the runner never sees the flag before the queue update. This prevents a race where:

- Dashboard: "mark working item as cancelled in queue"
- Runner reads: "cancel.flag exists" (flag seen)
- Runner reads: "queue still shows item as working" (stale queue snapshot)

The ordering ensures the runner sees either:
- Both the flag and the cancelled queue item, or
- Neither (flag not yet written, or from a prior iter).

### mtime-gating prevents stale cancellations

If a prior iter was cancelled and `cancel.flag` was not cleaned up, the runner would abort *every* subsequent iter when it sees the stale flag. The mtime-gating (line 1912) prevents this:

```python
if os.path.exists(_cf) and os.path.getmtime(_cf) >= _iter_started_ts:
```

Only flags written *after* the current iter started will trigger an abort. Flags from prior iters are ignored.

### No explicit cleanup of cancel.flag

Unlike `terminate.flag` (which is cleaned up at startup), `cancel.flag` is **not** explicitly deleted after an iter finishes. Instead, it is gated by mtime: the next iter's `_iter_started_ts` will be newer than the old flag's mtime, so the old flag is ignored. This simplifies the logic and avoids a filesystem write in the happy path (no cancel).

However, if `cancel.flag` is very old (days old), it will persist on disk indefinitely until manually cleaned or the scheduler upgrades. This is acceptable because the mtime-gating ensures it has no effect.

---

## Failure modes & edge cases

### Stale terminate.flag blocks restart (before ddab557)

**Symptom:** User terminates the fleet via dashboard. Manually restarts the runner. Runner exits immediately without entering the main loop.

**Root cause:** `terminate.flag` was never cleaned up. On restart, the flag is still present and the loop-top check (line 1595) immediately exits.

**Fix:** Lines 1575–1584 clear any stale flag at startup.

**Verification:** The fix is defensive code; there is no explicit test (the symptom manifests as a production incident).

### Stale cancel.flag triggers abort in wrong iter (before mtime-gating)

**Symptom:** User cancels job A. A few minutes later, user injects job B and starts it. Job B aborts despite no user action.

**Root cause:** If mtime-gating were absent, `cancel.flag` from the prior cancel would trigger an abort in B.

**Fix:** Line 1912 gates the check: `os.path.getmtime(_cf) >= _iter_started_ts`. Flags older than the current iter are ignored.

**Verification:** The gate is essential logic. Production testing (commit ddab557 notes) confirmed the feature works without false aborts.

### Cancelling pending item should not abort active iter

**Symptom:** Queue shows [working: job A], [pending: job B]. User clicks cancel on job B. Job A aborts.

**Root cause:** If the flag were written unconditionally, the runner would abort A.

**Fix:** Lines 306–311 in queue.py check `if item.get("status") == "working"` before incrementing `was_working`, and the flag is only written if `was_working` is non-empty.

**Verification:** Test `test_cancel_pending_does_not_write_flag` (line 154–163 in test_server_queue.py) confirms the flag is not written.

### Pause vs. cancel semantics

**Symptom:** User pauses the fleet. Does the active iter finish?

**Answer:** Yes. Pause is a soft stop. The check for `pause.flag` is at the top of the iter loop (line 1606), *before* a new task is popped. Active iters finish; new iters do not start.

**Cancel** is different. It aborts the active iter mid-flight (at chain boundaries), so partial output may be generated.

### Queue commit timing vs. flag write

**Invariant:** `cancel.flag` is written *after* `cfg.mutate_queue()` returns.

**Why:** If the flag were written first (or concurrently), the runner could see the flag, read the queue under its lock (but before the queue is updated), and not find the cancelled status — leading to an inconsistent view.

**Implementation:** All three endpoints (`/inject`, `/cancel-all`, `/queue/cancel`) follow this pattern:
1. Call `await asyncio.to_thread(cfg.mutate_queue, _mutator)` (line 204, 264, 300).
2. After return, write the flag (lines 205–210, 265–269, 307–311).

---

## Verification

### Unit tests

- **`test_pause_sets_flag` / `test_resume_clears_flag`** (test_server_queue.py:61–73): Verify pause endpoints return 200 (flag operations succeed).
- **`test_cancel_pending_does_not_write_flag`** (line 154–163): Cancelling a pending item (ts 2.0, status "pending") while working item (ts 1.0, status "working") is active does not write `cancel.flag`.
- **`test_cancel_working_writes_flag`** (line 165–173): Cancelling the working item (ts 1.0, status "working") writes `cancel.flag`.

### Integration tests

- **Test suite green (239 passed)** as noted in commit ddab557: the full suite runs with the flag IPC enabled and verifies queue operations, settings persistence, and the FLF2V / seed guards that were fixed alongside the IPC changes.

### Manual verification

1. **Terminate:** Start fleet, inject jobs, click dashboard "Terminate" button. Fleet should exit promptly. Restart the runner — it should proceed normally (not exit immediately due to stale flag).
2. **Pause:** Inject jobs, click "Pause". New iters should not start; active iter should finish. Click "Resume" — fleet should continue.
3. **Cancel active:** Queue shows [working: A] [pending: B, C]. Cancel A. A should abort at the next chain boundary. B and C should be unaffected.
4. **Cancel pending:** Queue shows [working: A] [pending: B, C]. Cancel B. A should continue. B should be marked cancelled in the queue.
5. **Stale cancel:** Cancel job A. Wait 5 minutes. Inject and start job B (new iter). B should run without false abort.

---

## Residuals / future work

### Explicit cancel.flag cleanup

Currently, `cancel.flag` is not deleted after an iter finishes; it is gated by mtime. A cleaner design would delete the flag after the iter (or before starting a new one), making the mtime-gating redundant. This would require:
- Adding `os.remove(cancel_flag_path)` after the exception handler (line 2151 in run_fleet.py).
- Catching `FileNotFoundError` since the flag may not exist.

This is a minor improvement and does not address a bug.

### Persistent flag state in dashboard

Currently, the flags are ephemeral files. The dashboard UI does not display the flag state; it infers pause/terminate state from the runner's process status (via `/runner/status`). A future enhancement could:
- Mirror flag state in the config/settings.
- Expose flag state in a `/flags/status` endpoint so the UI can show "paused" / "cancelled" states more reliably.

### Re-readable cancel.flag

The current design writes a timestamp to `cancel.flag`. The dashboard could, in principle, re-read this timestamp to determine the time of the last cancellation (for logging). Currently, no code reads the flag's content — only its existence and mtime are checked. If richer semantics are needed (e.g., which item was cancelled), the flag could carry a JSON payload. This is future work.

---

## References

- **Commits:** ddab557 (flag IPC and startup terminate.flag cleanup), ec7b849 (queue concurrency lock framework).
- **Related docs:** `docs/queue-concurrency.md` (queue lock design and lost-update race).
- **Queue constants:** `OUTPUT_DIR` in `run_fleet.py:33`, `EXP_DIR` in `slopfinity/paths.py:12–14`.
- **Flag files:** All three flags live in `{OUTPUT_DIR}/` (e.g., `comfy-outputs/experiments/terminate.flag`).
