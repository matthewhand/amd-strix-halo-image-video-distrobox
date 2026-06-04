# slopfinity — remaining follow-ups

Status snapshot after queue-concurrency + round-3/4/5 sweeps.
Test suite: **238 passed, 2 skipped, 3 xfailed**. Tree clean (only `slop_test/`
scratch dir untracked). HEAD `d9becc4`.

Severity legend: 🔴 high · 🟠 med · 🟢 low · ⚪ housekeeping

---

## 0. Done — round-5 sweep (`wf_69b498bb-62a`, 13 confirmed, 12 fixed in `ddab557`+`d9becc4`)

- [x] 🟠 terminate.flag never cleared → fleet un-restartable after a terminate.
- [x] 🟠 cancel.flag written but never read → cancelling the RUNNING item didn't
  abort it. Now checked at each chain boundary (mtime-gated).
- [x] 🟠 FLF2V with frames<9 placed end keyframe past latent → clamp to ≥9.
- [x] 🟠 `llm_cpu_mode` read from wrong namespace → always 'smart'. Fixed.
- [x] 🟠 `disk_min_pct`/`disk_min_gb` shown by GET but POST dropped them. Persist.
- [x] 🟠 SSRF guard now applied to `tts_worker_url`/`comfy_url`.
- [x] 🟠 `stage_prompts` {image,video,music} were persisted but never applied in
  run_fleet — now used per stage.
- [x] 🟢 per-chain seed mode with 1 seed → demote to per-task.
- [x] 🟢 requeue/requeue-failed now reset the failed run's `stages`.
- [x] 🟢 /tts default voice is engine-aware (qwen→ryan, not af_heart).
- [ ] 🟢 **fanout() doesn't pass `response_format`** (round-5 #12) — DEFERRED.
  `/enhance/distribute` isn't constrained to the JSON schema, but fanout already
  has retry + JSON-parse + seed-text fallback, so it's a marginal first-try
  reliability gain. Wiring it means changing the `llm_call(sys,user)` callback
  signature across callers + building the schema. Do it if enhance output proves
  flaky in practice.

---

## 0b. QA sweep across build/run/test (`wf_6ef4106a-9d2`, 40 confirmed)

Resolved (`0a4a407` build · `7cac0d9` run · `8f0042e` test):

- [x] 🟠 LLM provider detection heuristic (`"11434" in url`) → provider tagged at
  the pool source + read in both loops (env-overridable).
- [x] 🟢 Declare `pillow`+`numpy` (vae_grid); add `requirements-dev.txt` (test deps).
- [x] 🟢 Replace deprecated `datetime.utcnow()` everywhere (warnings 81→7).
- [x] 🟢 `ruff` F401: removed 41 unused imports (excl. `db.py` side-effect import
  + `__init__.py` re-exports); dropped unused `planner_hit`.
- [x] 🟢 `config.set_state` atomic write (tmp+fsync+replace) — no more torn reads
  / "Idle" flashes; narrowed bare `except:` in `get_state` + `stats.py`.
- [x] 🟢 Tests: new `test_qa_coverage.py` (stage-prompt resolution, cancel.flag
  mtime-gating, `extra` catch-all merge — via two new pure helpers in run_fleet);
  strengthened pause/resume (assert the flag file) + gpu-guard assertion.

Deferred (rationale):

- [ ] 🟢 **FastAPI `on_event` → `lifespan`** (server.py) — deprecated but still
  works; a startup/shutdown rewire is better done deliberately than AFK.
- [ ] 🟢 **Docker/compose env consistency** (HSA_OVERRIDE / FLASH_ATTENTION /
  `--force-fp16` mismatches across Dockerfile, docker-compose, start_docker.sh) —
  can't build/test docker here; needs a real image build to verify.
- [ ] 🟢 **Hardcoded endpoints** (`:8188` ComfyUI, `localhost:8010` TTS,
  import-time DB init) — work on the standard single-box setup; env-override has
  many touch points. Low impact until someone runs a non-standard topology.
- [ ] 🟢 **Integration-test harness** — `test_pipeline_slow` returns bool instead
  of asserting (only "passes" without a live server because of it; needs a
  skip-if-no-server guard); `sched.GPU` test-only-attribute pattern diverges from
  prod `get_gpu()`; xfail review; `test_ai_mock_integration._main` list. These are
  test-infra redesigns — a naive fix breaks the suite.
- [ ] 🟢 **Minor consistency** — `/settings/models` HTTP status codes; LLM timeout
  default 60 vs 120 (arguably intentional for chat tool-loops); the duplicated
  endpoint-selection block in `lmstudio_call`/`lmstudio_chat_raw` (the provider
  bug in it is fixed; the duplication itself remains).
- [ ] 🟢 **Deeper run_fleet test coverage** — FLF2V `max(9,…)` clamp, per-chain
  seed demotion, broadcaster 48h prune, polymorphic mid-flight demotion. Each
  needs a small pure-helper extraction (like the two added this round) or
  endpoint/seed-file fixtures to test without a GPU.

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

- [x] 🟢 **`tests/test_ai_mock_integration.py`** — FIXED (`f697ae2`). The 10s
  mock-LLM request timeout in `test_subjects_suggest` / `test_enhance_distribute`
  expired under full-suite load; bumped to 30s. 3/3 consecutive full-suite runs
  clean afterward.

---

## 7. Verification artifacts (reference)

- `docs/queue-concurrency.md` — design note for the lost-update fix.
- `tests/test_config_extras.py` — queue/config lock + mutate_queue/mutate_config
  + extra-field round-trip + 5×20 cross-process no-lost-update test.
- `tests/test_comfy_poll.py` — ComfyUI poll deadline/timeout/error tests.
