# Queue concurrency: cross-process serialization of lost-update race

## Executive summary

The persisted job queue (`queue.json` + `QueueItem` in SQLite) is shared IPC between three concurrent writers: `run_fleet.py` (host process), the FastAPI dashboard + chat tools + broadcaster (container), and `slopfinity/workers/` (dormant Phase-4). Because `save_queue()` does a blind delete-all + reinsert of the entire list, two writers overlapping in their read→modify→write window silently lose each other's edits. Real-world data loss: injected jobs vanished; user edits were wiped; jobs ran twice. The race also manifested as `sqlalchemy.exc.StaleDataError` when concurrent delete paths collided.

**Fix:** Every read-modify-write site now serializes through a cross-process advisory lock (`queue_lock()`) on `<QUEUE_FILE>.lock`, using either `mutate_queue(mutator)` (preferred for one-shot RMW) or explicit `with queue_lock():` spans for conditional saves.

**Commits:** `ec7b849` (core fix) and `61094b0` (event-loop blocking regression fix).

---

## Root cause: save_queue's destructive rewrite

**File:** `slopfinity/config.py:619–648`

```python
def save_queue(q):
    """Save the queue to the database and sync to JSON."""
    init_db()
    with Session(engine) as session:
        # Delete all existing and replace (simulates the current list behavior)
        # In the future, we should do delta updates for performance.
        session.exec(select(QueueItem)).all() # load for session
        for item in session.exec(select(QueueItem)).all():
            session.delete(item)
        
        for item_dict in q:
            # _split_queue_item stamps a stable id (run_fleet appends iter rows
            # without one — otherwise the id default_factory churns a fresh uuid
            # every save and status tracking can't follow an item across saves)
            # and funnels non-column keys into `extra` so they aren't dropped.
            session.add(QueueItem(**_split_queue_item(item_dict)))
        session.commit()

    # Sync to JSON as backup
    try:
        os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
        tmp = QUEUE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(q, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, QUEUE_FILE)
    except Exception:
        pass
```

Every caller does `get_queue()` → `[modify list]` → `save_queue(q)`. The entire list is persisted in one operation: all old rows deleted, all new rows inserted. If two processes interleave this pattern:

```
Process A: reads queue → [5 items]
           modifies     → [5 items, new job #6]
Process B: reads queue → [5 items]
           modifies     → [5 items, item #3 cancelled]
           saves        → DB now: [5 items, #3 cancelled]
Process A: saves        → DB now: [5 items, new job #6] ← loses cancel
```

Process B's cancellation is silently overwritten. With concurrent SQLAlchemy `delete()` operations on overlapping row sets, the race also triggers `StaleDataError` (SQLAlchemy exception when a DELETE attempts to affect rows that were deleted between the load and the delete statement).

---

## Concurrent writers and observed loss patterns

| Writer | Process | Call sites | Loss modes |
|--------|---------|-----------|-----------|
| `run_fleet.py` | host runner | Claim next pending (line 307 `cfg.mutate_queue`); startup stale-`working` sweep (line 1568 `cfg.mutate_queue`); post-iter requeue/archive (line 2186 `with queue_lock`) | Edit wiped by concurrent claim; inject vanishes behind requeue; job marked done + pending simultaneously |
| FastAPI dashboard | container | `/inject` (204), `/cancel-all` (264), `/queue/cancel` (300), `/queue/edit` (339), `/queue/toggle-infinity` (367), `/queue/toggle-polymorphic` (394), `/queue/requeue` (485), `/queue/requeue-failed` (566), `/queue/clear-failed` (506), `/queue/clear-completed` (526) | Injected job clobbered by simultaneous claim; edit lost to finalize; toggle overwritten |
| Chat tools | container | `queue_clip` (221), `cancel_item` (271), `generate_image` (334) | Same as dashboard: mutation lost to overlapping writer |
| Broadcaster pruner | container | Stale-cancelled cleanup (lines 146–149) | Pre-check racy but re-filters under lock; tick loop runs every 2s so unconditional saves would hammer lock |
| Worker subprocesses | container (dormant Phase-4) | `claim_next` (workers/base.py:71 `with queue_lock`); `_finalize_sync` (workers/base.py:137 `with queue_lock`) | Concurrent claims both flip a stage to `working`; finalizer overwrites claim |

---

## The fix: queue_lock() + mutate_queue()

### Mechanism: cross-process flock

**File:** `slopfinity/config.py:537–551`

```python
_QUEUE_LOCK_FILE = QUEUE_FILE + ".lock"

@contextlib.contextmanager
def queue_lock():
    """Hold an OS-level advisory lock around a get_queue → save_queue
    round-trip. Cooperative across processes (host runner + container
    dashboard + worker subprocesses) since they all open the same
    .lock file. Non-recursive: nesting will deadlock — keep critical
    sections short and avoid calling out to other queue operations
    while holding."""
    os.makedirs(os.path.dirname(_QUEUE_LOCK_FILE), exist_ok=True)
    with open(_QUEUE_LOCK_FILE, "a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
```

Opens the same `queue.json.lock` file across all processes (they all resolve `SLOPFINITY_STATE_DIR` to the shared bind-mount). The `fcntl.flock(fd, LOCK_EX)` acquires an exclusive advisory lock; the OS kernel blocks the second acquirer until the first holder calls `fcntl.flock(fd, LOCK_UN)`. This serializes all writers.

**Data flow under the lock:**

```
[Acquire flock(LOCK_EX) on queue.json.lock]
  ↓
  get_queue()      ← read from DB/JSON (sees previous state or another's write)
  ↓
  [Critical section: in-memory list modifications]
  ↓
  save_queue(q)    ← delete-all + reinsert under same lock hold
  ↓
[Release flock(LOCK_UN)]
```

Subsequent writers block at the lock, then observe the previous writer's final state.

### Pattern 1: mutate_queue(mutator) — preferred

**File:** `slopfinity/config.py:650–671`

```python
def mutate_queue(mutator):
    """Atomic read→modify→write of the queue under the cross-process lock.

    save_queue does a blind delete-all + reinsert of the whole list, so two
    processes doing get_queue→modify→save_queue concurrently (host run_fleet +
    container dashboard + worker subprocesses) lose each other's edits. This
    runs the full cycle inside queue_lock() (an flock shared via the same
    SLOPFINITY_STATE_DIR), serialising all writers.

    `mutator(queue_list)` mutates the list in place and/or returns a new list;
    the resulting list is persisted and returned.

    CONTRACT: the mutator must NOT call get_queue/save_queue/mutate_queue
    (queue_lock is non-recursive → re-entry deadlocks) and must not block on
    network/subprocess/GPU work while the lock is held — keep it pure list work.
    """
    with queue_lock():
        q = get_queue()
        result = mutator(q)
        q = result if isinstance(result, list) else q
        save_queue(q)
        return q
```

**Usage example from `/inject`:** `slopfinity/routers/queue.py:171–204`

```python
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
    pending = [x for x in q if x.get("status") in (None, "pending")]
    working = [x for x in q if x.get("status") == "working"]
    done = [x for x in q if x.get("status") == "done"]
    cancelled = [x for x in q if x.get("status") == "cancelled"]
    
    if priority in ("now", "next"):
        for t in reversed(tasks_to_queue):
            pending.insert(0, t)
    else:
        for t in tasks_to_queue:
            pending.append(t)
    
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

The `_do_inject` mutator:
- Mutates the input list in place (flip statuses for terminate)
- Returns a new ordered list (working + pending + done + cancelled)
- Has **no early returns or conditions**—this is why it works as a pure function

For handlers that need to signal "not found" or return metadata:

```python
found = []
def _edit(q):
    for item in q:
        if item.get("ts") == target_ts and item.get("status") in (None, "pending", "working"):
            item["prompt"] = new_prompt
            item["seed_prompt"] = new_prompt
            found.append(1)
            break
    return q

await asyncio.to_thread(cfg.mutate_queue, _edit)
if not found:
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
return {"ok": True}
```

The `found` list (closure) captures the result inside the lock, and the handler inspects it **after** `mutate_queue` returns.

**Call sites using `mutate_queue`:**

| File | Endpoint/Function | Line |
|------|------------------|------|
| routers/queue.py | `/inject` | 204 |
| routers/queue.py | `/cancel-all` | 264 |
| routers/queue.py | `/queue/cancel` | 300 |
| routers/queue.py | `/queue/edit` | 339 |
| routers/queue.py | `/queue/toggle-infinity` | 367 |
| routers/queue.py | `/queue/toggle-polymorphic` | 394 |
| routers/queue.py | `/queue/requeue` | 485 |
| routers/queue.py | `/queue/requeue-failed` | 566 |
| routers/queue.py | `/queue/clear-failed` | 506 |
| routers/queue.py | `/queue/clear-completed` | 526 |
| routers/chat.py | `queue_clip` | 221 |
| routers/chat.py | `cancel_item` | 271 |
| routers/chat.py | `generate_image` | 334 |
| run_fleet.py | `generate_prompt` (claim next) | 307 |
| run_fleet.py | startup sweep (stale `working` cleanup) | 1568 |

### Pattern 2: Explicit `with queue_lock():` span — for conditional/early-return logic

**File:** `run_fleet.py:2186–2250` (excerpt)

```python
try:
    if _task_opts.get("_seed_prompt"):
        with cfg.queue_lock():
            q_now = cfg.get_queue()
            # Pull (and remove) the in-flight row matching this iter
            # so we can read the user's most recent toggle / cancel state.
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

            # Effective flags: the live record's flags win over the snapshot
            live = live_record or {}
            eff_infinity = bool(live.get("infinity", _task_opts.get("infinity")))
            eff_polymorphic = bool(
                live.get(
                    "polymorphic",
                    live.get("chaos", _task_opts.get("polymorphic", True)),
                )
            )
            # ... [reconciliation] ...
            cfg.save_queue(q_now)
```

Used for:
- **Run_fleet requeue/archive** (`run_fleet.py:2186`): The ~60-line reconciliation re-reads the live in-flight record to pick up user-made toggle/cancel state, rebuilds the queue, saves once. The lock is held for the entire reconcile so the re-read and final write happen atomically.
- **Worker claim_next** (`workers/base.py:71`): Loops through queue items looking for a claimable stage. Conditional save: only persists if it found work. Wrapping preserves the "only save if something changed" semantic.
- **Worker _finalize_sync** (`workers/base.py:137`): After a stage completes, finds the item by `id`, flips status to `done` or `failed`, saves. Conditional save inside the lock.
- **Broadcaster stale-cancelled pruner** (`broadcaster.py:146–149`): Runs every 2s and would unconditionally save if not guarded. Does a cheap unlocked pre-check (`any(_is_stale(x) …)`), then enters `mutate_queue` to re-filter fresh under the lock.

**Broadcaster pattern (defensive pre-check):** `broadcaster.py:139–149`

```python
def _is_stale(x):
    return x.get("status") == "cancelled" and (x.get("cancelled_ts") or 0) < cutoff

# This tick runs frequently; only take the cross-process lock + write
# when there's actually something to prune (the unlocked read above is
# just a cheap pre-check — the mutator re-filters fresh under the lock,
# so a concurrent writer can't be clobbered).
if any(_is_stale(x) for x in queue):
    queue = await asyncio.to_thread(
        cfg.mutate_queue, lambda q: [x for x in q if not _is_stale(x)]
    )
```

The unlocked pre-read is racy by design (it may see stale data). But the mutator **always** re-checks under the lock, so the final write is authoritative. This avoids holding the lock across every 2-second tick.

---

## Lock contract: non-recursive, short, side-effect-free

### Non-recursive: no nesting

**Critical invariant:** `queue_lock()` is **not reentrant**. A single process calling it twice on different file descriptors will deadlock:

```python
# DEADLOCK:
with cfg.queue_lock():        # fd1 → flock LOCK_EX
    # ... work ...
    with cfg.queue_lock():    # fd2 (different fd!) → flock tries again
        # Process hangs waiting for its own lock
```

Each call opens a fresh fd and takes `LOCK_EX` from scratch. The OS kernel blocks the second `flock()` call even though the same process already holds the lock (and cannot release it until the second call completes — deadlock).

**Contract violation examples:**

```python
# WRONG: mutator calls get_queue (re-entry):
def bad_mutator(q):
    q_fresh = cfg.get_queue()  # ← re-entrant call inside lock
    return q + q_fresh

cfg.mutate_queue(bad_mutator)  # DEADLOCK

# WRONG: explicit lock nesting:
with cfg.queue_lock():
    with cfg.queue_lock():     # DEADLOCK
        ...
```

All converted sites were audited to ensure mutators call **only** within the `mutate_queue` boundary, and never call `get_queue` / `save_queue` / `mutate_queue` from inside a mutator function or an open `with queue_lock()` span.

### Keep critical sections short and pure

The lock must never be held across:
- **Network calls** (LLM, TTS endpoints)
- **Subprocess execution** (ffmpeg, ComfyUI)
- **GPU acquisition** (`sched.acquire_gpu()`)
- **File I/O** beyond the queue read/write
- **Blocking on external services**

**Why:** A long-held lock starves other writers and can deadlock the system. On Strix Halo with a single GPU, holding the queue lock during a multi-minute generation blocks the dashboard from responding.

**Verified by code review:** All converted spans contain only in-memory list operations:
- Filtering by status, timestamp, id
- Inserting, removing, mutating fields
- Rebuilding list order

The queue lock is **released before** the LLM call that follows `generate_prompt`'s claim (line 308 onwards), and before any generation work in the handlers.

### Side effects after commit

Handlers that need to write flags (e.g., `cancel.flag` to abort the running job) do so **after** the locked commit:

```python
await asyncio.to_thread(cfg.mutate_queue, _do_inject)  # ← commit happens here
if _cancel_ts:
    try:
        with open(os.path.join(EXP_DIR, "cancel.flag"), "w") as f:
            f.write(str(_cancel_ts[0]))
    except Exception:
        pass
return {"status": "ok"}
```

This ensures the fleet runner never sees a flag before the queue update lands in the database, preventing race conditions where the runner acts on the flag but doesn't yet see the queue change.

---

## Event-loop blocking fix (commit 61094b0)

AsyncIO handlers calling synchronous blocking I/O (flock, SQLite commits) directly would block the entire event loop and hang the dashboard. The fix wraps all `mutate_queue` calls in async handlers with `asyncio.to_thread`:

**File:** `slopfinity/routers/queue.py:204` (example from `/inject`)

```python
await asyncio.to_thread(cfg.mutate_queue, _do_inject)
```

This offloads the critical section to a worker thread pool, preventing the main loop from stalling. All async handlers that perform queue mutations now use `await asyncio.to_thread(cfg.mutate_queue, ...)`.

**Affected sites:**

| File | Handler | Line |
|------|---------|------|
| routers/queue.py | `/inject` | 204 |
| routers/queue.py | `/cancel-all` | 264 |
| routers/queue.py | `/queue/cancel` | 300 |
| routers/queue.py | `/queue/edit` | 339 |
| routers/queue.py | `/queue/toggle-infinity` | 367 |
| routers/queue.py | `/queue/toggle-polymorphic` | 394 |
| routers/queue.py | `/queue/requeue` | 485 |
| routers/queue.py | `/queue/requeue-failed` | 566 |
| routers/queue.py | `/queue/clear-failed` | 506 |
| routers/queue.py | `/queue/clear-completed` | 526 |
| routers/chat.py | `queue_clip` | 221 |
| routers/chat.py | `cancel_item` | 271 |
| routers/chat.py | `generate_image` | 334 |
| broadcaster.py | stale-cancelled pruner | 147 |

The `run_fleet.py` synchronous calls to `mutate_queue` and explicit `with queue_lock():` are already in a synchronous context, so no wrapping is needed.

---

## Interaction with GPU lock and config lock

### GPU lock independence

`queue_lock()` guards queue *metadata* (the job list). `sched.acquire_gpu()` (scheduler.py) serializes *generation work* (loading models, running inference). **They are independent and must stay so:**

- The queue lock is never held across GPU acquisition
- A long generation doesn't starve queue mutations
- Dashboard remains responsive while the GPU generates

The two locks protect different resources; blocking one doesn't block the other.

### Config lock (separate)

`config_lock()` (config.py:677–687) is a distinct flock for `load_config()` ↔ `save_config()` round-trips. It's completely independent:

```python
@contextlib.contextmanager
def config_lock():
    """Cross-process advisory lock for a load_config → save_config round-trip.
    Separate flock from queue_lock; non-recursive (don't nest)."""
    os.makedirs(os.path.dirname(_CONFIG_LOCK_FILE), exist_ok=True)
    with open(_CONFIG_LOCK_FILE, "a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
```

Holding one lock does not block acquisition of the other.

---

## Failure modes before the fix

### Data loss

1. **Inject vanishes:** Dashboard injects a job while `run_fleet` is mid-requeue with a stale snapshot. The runner overwrites without the new job.
2. **Edit wiped:** User edits a pending prompt while the runner claims the next item. The runner's requeue snapshot overwrites the edit.
3. **Jobs run twice:** A worker flips a stage to `working`; a concurrent finalizer with a stale snapshot writes it back to `pending`.
4. **Cancel doesn't stick:** User cancels a job; concurrently the runner requeues with the old status, making the cancel disappear.

### StaleDataError

When two processes delete overlapping row sets in rapid succession, SQLAlchemy's optimistic locking (checking row-count changes) detects the collision:

```
Process A: SELECT * FROM queue_item        # 5 rows
Process B: SELECT * FROM queue_item        # 5 rows
Process B: DELETE FROM queue_item          # deletes all 5
Process B: INSERT (new set)
Process A: DELETE FROM queue_item WHERE id IN (old 5)  
           # ← only deletes 0 rows; statement.rowcount != expected
           # → sqlalchemy.exc.StaleDataError
```

This was the #1 high-priority correctness bug from the security audit.

---

## Verification

### Test: Exclusive lock (test_config_extras.py:58–71)

```python
def test_queue_lock_is_exclusive():
    # queue_lock holds an exclusive flock — a second non-blocking acquire fails.
    import fcntl
    import pytest
    from slopfinity import config as cfg
    with cfg.queue_lock():
        with open(cfg._QUEUE_LOCK_FILE, "a+") as fd:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # after release it acquires fine
    with open(cfg._QUEUE_LOCK_FILE, "a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
```

Proves that a second non-blocking flock attempt fails while the first is held, confirming serialization.

### Test: Atomic RMW (test_config_extras.py:109–120)

```python
def test_mutate_queue_reads_modifies_saves():
    from slopfinity import config as cfg
    try:
        cfg.save_queue([])
        out = cfg.mutate_queue(
            lambda q: q + [{"prompt": "m", "ts": 1.0, "status": "pending"}]
        )
        assert any(i["prompt"] == "m" for i in out)
        assert any(i["prompt"] == "m" for i in cfg.get_queue())
    finally:
        cfg.save_queue([])
```

Confirms that `mutate_queue` reads, modifies, and persists, and that the returned state matches what `get_queue` observes.

### Test: Cross-process no lost updates (test_config_extras.py:138–180)

```python
def test_mutate_queue_no_lost_updates_across_processes(tmp_path):
    """End-to-end proof the lost-update race is closed: N real subprocesses
    each append M rows concurrently; all N*M must land (no clobber)."""
    # ... spawn N=5 processes, each appending M=20 rows via mutate_queue ...
    # total, distinct = (int(x) for x in result.stdout.split())
    # assert total == N * M, f"lost updates: got {total} of {N * M}"
    # assert distinct == N * M, f"id collisions / drops: {distinct} distinct"
```

**5 × 20 concurrent proof:** 5 subprocesses each call `mutate_queue` 20 times, appending rows concurrently. Expected: 100 rows, 100 distinct ids. 

**Without the lock:** ~39/100 rows survived in test runs; the rest were clobbered by the blind delete-all+reinsert race, and `StaleDataError` exceptions occurred when concurrent deletes collided.

**With the lock:** All 100 rows land, 100 distinct, 0 errors. Tests pass repeatedly.

### Integration: Full test suite

- **227 passed, 2 skipped, 3 xfailed** (commit 61094b0)
- No regressions
- No deadlock detections
- All `/queue/*` endpoints exercise the new `mutate_queue` path via mocking and integration tests

---

## Residual and future work

### Row-level updates (future optimization)

`save_queue()` still rewrites the entire list—O(n) per write. Under the lock this is correct, but high-concurrency systems might benefit from delta updates:

```python
# Hypothetical future:
def update_queue_item(item_id, updates):
    with queue_lock():
        session.exec(
            update(QueueItem)
            .where(QueueItem.id == item_id)
            .values(**updates)
        )
        session.commit()
```

Not required for correctness; blind rewrite is fast enough for the current queue sizes (< 1 KB per item, typically 10–100 items).

### Phase-4 workers (dormant)

`slopfinity/workers/base.py` (claim_next, _finalize_sync) is locked defensively even though the workers are dormant until Phase-4 activates. Whoever integrates them inherits the queue-concurrency invariant.

### Dashboard settings direct to save_config

Dashboard settings endpoints (`/config/save`, etc.) currently call `save_config()` directly without routing through `mutate_config()`. They are not yet serialized against `run_fleet`'s infinity-theme rotations or `mutate_config` calls in the broadcaster. This is opt-in for now (backwards compatible); future work should route them through the lock.

---

## Code references

| Artifact | Location | Purpose |
|----------|----------|---------|
| `queue_lock()` | slopfinity/config.py:537–551 | Acquire/release cross-process flock |
| `mutate_queue(mutator)` | slopfinity/config.py:650–671 | One-shot locked RMW helper |
| `save_queue(q)` | slopfinity/config.py:619–648 | Delete-all + reinsert (blind) |
| `_split_queue_item(item_dict)` | slopfinity/config.py:554–574 | Funnel non-column fields into `extra` catch-all |
| `/inject` handler | slopfinity/routers/queue.py:171–204 | Wraps new job insertion in `mutate_queue` (line 204) |
| `/queue/cancel` handler | slopfinity/routers/queue.py:272–312 | Wraps item cancellation (line 300) |
| `generate_prompt` (claim) | run_fleet.py:283–401 | Wraps next-job claim in `mutate_queue` (line 307) |
| startup sweep | run_fleet.py:1559–1574 | Wraps stale-`working` cleanup in `mutate_queue` (line 1568) |
| requeue/archive | run_fleet.py:2185–2250 | Wraps reconcile in explicit `with queue_lock()` (line 2186) |
| broadcaster pruner | slopfinity/broadcaster.py:139–149 | Unlocked pre-check, then `mutate_queue` (line 147) |
| worker claim_next | slopfinity/workers/base.py:64–106 | Wraps claim attempt in `with queue_lock()` (line 71) |
| worker _finalize_sync | slopfinity/workers/base.py:136–170 | Wraps status update in `with queue_lock()` (line 137) |
| test: lock exclusive | tests/test_config_extras.py:58–71 | Proves second flock attempt blocks |
| test: mutate RMW | tests/test_config_extras.py:109–120 | Proves read–modify–write round-trip |
| test: 5×20 concurrent | tests/test_config_extras.py:138–180 | Proves all 100 rows land without clobber |
