# slopfinity — remaining follow-ups

Status snapshot after queue-concurrency + round-3 + round-4 sweeps.
Test suite: **235 passed, 2 skipped, 3 xfailed**. Tree clean (only `slop_test/`
scratch dir untracked). HEAD `61094b0`.

Severity legend: 🔴 high · 🟠 med · 🟢 low · ⚪ housekeeping

---

## 1. Done — round-4 sweep (`wf_a968c3a5-8bf`, 10 confirmed, all fixed in `61094b0`)

- [x] 🟠 `/music` subprocess had no timeout inside `acquire_gpu` → GPU-lock
  starvation on a wedged docker/GPU. Added `timeout=1200`.
- [x] 🟠 `broadcaster` referenced non-existent `sched.GPU` → AttributeError on
  every paused tick. Use `get_gpu()`.
- [x] 🟠 `migrate_to_db.py` bypassed `_split_queue_item` (re-dropped extra
  fields). Routed through the helper.
- [x] 🟢 `_poll_comfy_history` empty-images IndexError guard.
- [x] 🟢 Fast Track sidecar reported pre-override snapshot; re-point
  `_CURRENT_ITER_CONFIG`.
- [x] 🟢 `_frames_per_chain` clamped to ≥1.
- [x] 🟢 Event-loop blocking: `mutate_queue`/`mutate_config` + chat tool handlers
  now run via `asyncio.to_thread` (queue.py, chat.py, broadcaster).
- [x] 🟢 dormant `workers/audio.py` defaulted output to `/tmp` → `EXP_DIR`.

---

## 2. Confirmed-but-deferred (conscious decisions)

- [ ] 🟢 **Legacy queue item missing `prompt`** — in `config.get_queue()` legacy
  JSON→DB migration, an item without `prompt` raises in `QueueItem(**…)` and is
  skipped+logged. *Deferred:* a prompt-less item is unusable; defaulting it to
  `""` would inject a degenerate runnable row. Skip+log is the safer behavior.
  Revisit only if real user data is being silently dropped.

---

## 3. Known residuals (carried from earlier sweeps, not yet done)

- [ ] 🟠 **`acquire_gpu` `resident_models` leak** — `slopfinity/scheduler.py`.
  The resident-model accounting grows / isn't evicted, degrading the planner's
  VRAM-budget accuracy over a long run. Touches the **live GPU lock** — must be
  done carefully and never weaken GPU serialization (gfx1151 hangs under
  concurrent GPU). Needs its own design note + test.
- [ ] 🟢 **`/queue/edit` accepts `done` items** — `slopfinity/routers/queue.py`.
  Editing a completed item's prompt is a no-op-ish footgun; should 400 on
  terminal-status items (mirror the cancel/requeue status guards).
- [ ] 🟢 **`/tts` hardcodes the `qwen-tts` budget role** — `routers/runner.py`
  `acquire_gpu("TTS", "qwen-tts", …)` even when the actual engine is
  kokoro/dramabox/heartmula, so the VRAM budget check uses the wrong model size.
  Pass the resolved engine through to the budget role.
- [ ] 🟢 **`/inject` priority not validated** — `routers/queue.py` accepts any
  `priority` string; only `now`/`next` front-insert, anything else silently
  appends. Validate against the known set and 400 otherwise.

## 4. Launcher dead-code (NOT-LIVE Phase-4 path — low urgency)

- [ ] 🟢 **Stale launcher paths in the dormant coordinator** —
  `slopfinity/worker_sh.py` / `slopfinity/workers/*` reference
  `/opt/{kokoro,ltx}_launcher.py` and the broken qwen `--out` flag. These are in
  the **not-live** `workers/` + `coordinator.py` refactor. Either delete the dead
  paths or fix them when/if Phase-4 is activated. No live impact today.

---

## 5. Engineering / future work (tracked in design notes)

- [ ] ⚪ **`save_queue` rewrites the entire list per write** (delete-all+reinsert).
  Correct under the new lock but O(n) write amplification on large histories.
  A delta/row-update path would cut it. See `docs/queue-concurrency.md` §Residual.
- [ ] ⚪ **Config locking is opt-in** — `config_lock`/`mutate_config` serialize the
  two background loops (run_fleet infinity-index ↔ broadcaster chaos_rotator) but
  the **dashboard settings endpoints still call `save_config` directly** and are
  not serialized against them. A settings save overlapping a background loop can
  still revert keys. Full coverage = route every config writer through
  `mutate_config` (larger change; queue-style migration).
- [ ] ⚪ **`workers/base.py` lock is async-blocking** — `claim_next` /
  `_finalize_sync` hold a blocking `with queue_lock()` (flock + SQLite) inside
  `async` context. Dormant, so left as-is; if Phase-4 goes live, run the locked
  body via `asyncio.to_thread`. (The LIVE async sites — queue.py, chat.py,
  broadcaster — were already wrapped in `61094b0`.)

---

## 6. Known-flaky test (not a regression — leave or fix upstream)

- [ ] 🟢 **`tests/test_ai_mock_integration.py`** — `test_subjects_suggest` /
  `test_enhance_distribute` intermittently time out (10s) on the mock-LLM
  endpoint under machine load; pass on retry and in isolation. Env-coupled
  subprocess timing, unrelated to any queue/run_fleet code. Could be hardened
  with a longer timeout / readiness wait, or marked `flaky`.

---

## 7. Verification artifacts (reference)

- `docs/queue-concurrency.md` — design note for the lost-update fix.
- `tests/test_config_extras.py` — queue/config lock + mutate_queue/mutate_config
  + extra-field round-trip + 5×20 cross-process no-lost-update test.
- `tests/test_comfy_poll.py` — ComfyUI poll deadline/timeout/error tests.
