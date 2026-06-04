# Queue concurrency: closing the lost-update race

**Status:** implemented
**Scope:** the persisted job queue (`config.get_queue` / `config.save_queue`),
shared as an IPC layer between three independent writers.

## The bug

`queue.json` (mirrored into SQLite via `QueueItem`) is written by
`config.save_queue(q)`, which does a **blind delete-all + reinsert of the whole
list**. There is no row-level update path — every writer reads the entire queue,
mutates a copy in memory, and rewrites the whole thing.

Three families of writers do this `get → modify → save` dance concurrently:

| Writer | Process | Examples |
|---|---|---|
| `run_fleet.py` | host runner | claim next pending, startup stale-`working` sweep, post-iter requeue/archive |
| FastAPI dashboard | container | `/inject`, `/cancel-all`, `/queue/cancel`, `/queue/edit`, `/queue/requeue`, `/queue/requeue-failed`, `/queue/clear-failed`, `/queue/clear-completed`, `/queue/toggle-infinity`, `/queue/toggle-polymorphic`, chat queue tools, broadcaster stale-cancelled pruner |
| `slopfinity/workers/base.py` | worker subprocs (Phase-4, dormant) | `claim_next`, `_finalize_sync` |

Because the whole list is rewritten, **any two writers overlapping in the
read→write window silently clobber each other**. Concrete losses observed/possible:

- The dashboard injects a job; `run_fleet` was mid-requeue with an older
  snapshot and rewrites the queue without the new job → **inject vanishes**.
- A user edits a pending prompt while the runner claims the next item → **edit
  wiped**.
- A worker flips a stage to `working`; a finalizer with a stale snapshot writes
  it back to `pending` → **job runs twice**.

This was the #1 high-priority correctness finding from the issue sweeps.

## The fix

A cross-process advisory lock already existed but was **defined and unused**:
`config.queue_lock()` — an `fcntl.flock(LOCK_EX)` on `<QUEUE_FILE>.lock`. Because
every process resolves the same path from `SLOPFINITY_STATE_DIR`, the lock
serialises the host runner, the container dashboard, and any worker subprocess.

Two call patterns now wrap **every** read-modify-write site:

### 1. `mutate_queue(mutator)` — the one-shot RMW helper (preferred)

```python
def mutate_queue(mutator):
    with queue_lock():
        q = get_queue()
        result = mutator(q)
        q = result if isinstance(result, list) else q
        save_queue(q)
        return q
```

`mutator(q)` mutates the list in place and/or returns a replacement list. The
read, the mutation, and the write all happen inside one held lock, so no other
writer can interleave. Used by the dashboard handlers and the two simple
`run_fleet` spans (claim, startup sweep).

Handlers that need to signal "not found" / return a value do so via a captured
list (closure), then inspect it *after* `mutate_queue` returns — e.g.

```python
found = []
def _edit(q):
    for item in q:
        if item.get("ts") == target_ts and ...:
            item["prompt"] = new_prompt
            found.append(1)
            break
    return q
cfg.mutate_queue(_edit)
if not found:
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
```

### 2. Explicit `with queue_lock():` span — for conditional / mid-block returns

A few sites can't be expressed as a single pure mutator without changing their
save semantics:

- **`run_fleet` requeue/archive** (the 130-line reconcile): re-reads the live
  record to pick up mid-flight cancel/toggle state, rebuilds the list, single
  commit. Wrapped as `with cfg.queue_lock(): q_now = get_queue(); …; save_queue(q_now)`.
- **`workers/base.py` `claim_next` / `_finalize_sync`**: mid-loop `return` after
  a conditional save; wrapping preserves "only save when something changed".
- **`broadcaster` stale-cancelled pruner**: runs on every WS tick, so it must
  *not* save unconditionally. It does a cheap unlocked pre-check
  (`any(_is_stale(x) …)`) and only then enters `mutate_queue` (which re-filters
  fresh under the lock). The pre-check is racy by design; the locked re-filter
  is authoritative.

### What intentionally stays lock-free

Pure reads — `get_queue()` on its own, `/queue/paginated`,
`/queue/pause-state`, the chat `list_queue` / `get_status` tools. A reader can
tolerate seeing the pre- or post-write state; it never rewrites.

## Lock contract (important)

- **Non-recursive.** `queue_lock()` opens a fresh fd and takes `LOCK_EX` each
  time; the same process taking it twice (even on different fds) deadlocks.
  → A mutator must **never** call `get_queue` / `save_queue` / `mutate_queue`,
  and you must never nest `queue_lock()` spans.
- **Keep critical sections pure + short.** No network, subprocess, GPU, or
  `acquire_gpu` calls while holding it. All converted spans were verified to
  contain only in-memory list work + the get/save pair. The lock is released
  *before* the LLM call that follows the `run_fleet` claim, and before any
  generation work.
- **Side effects go outside.** `cancel.flag` writes (terminate / cancel-active)
  happen *after* the locked commit, driven by state captured inside the
  mutator, so the fleet never sees the flag before the queue update lands.

## Interaction with the GPU lock

This change does **not** touch `scheduler.acquire_gpu`. The queue lock guards
queue *metadata*; the GPU lock serialises *generation*. They are independent and
must stay so — the queue lock is never held across a GPU acquisition (gfx1151
hangs under concurrent GPU work, and holding the queue lock across a multi-minute
generation would stall the whole dashboard).

## Verification

- `tests/test_config_extras.py::test_queue_lock_is_exclusive` — proves a second
  acquirer blocks until the first releases.
- `tests/test_config_extras.py::test_mutate_queue_reads_modifies_saves` — proves
  the helper round-trips read→modify→persist.
- `tests/test_server_queue.py` (`TestInjectFlagParsing`, `TestQueueCancelFlagScope`)
  — exercise `/inject` and `/queue/cancel` through the new `mutate_queue` path.
- Full suite: **227 passed, 2 skipped, 3 xfailed** — no regressions, no deadlocks.

## Residual / future work

- `save_queue` still rewrites the entire list. Under the lock this is correct but
  O(n) per write; a delta/row-update path would cut write amplification on large
  histories. Not required for correctness.
- The `workers/` + `coordinator.py` Phase-4 path is dormant; its writers are
  locked here defensively so that whoever activates it inherits the invariant.
