# Concurrent Iteration Execution — Implementation Plan

**Status:** Research + design pass. No code lands from this PR.
**Branch:** `agent/plan-concurrent-iterations`
**Audience:** Maintainer deciding how (and whether) to unlock the `concurrent-on`
toggle in `slopfinity/templates/index.html:1599` so two queue items can actually
overlap on Strix Halo's 128 GB unified memory.

This plan complements `docs/concurrent-mode-design.md` (which already covered the
GPU reservation primitive) and `docs/queueing-refactor-design.md` (which covered
the per-stage worker fan). Those two designs are *both partially landed*; the
gap that prevents real concurrency today is the orchestrator on top of them.

---

## 1. Resource model

### Per-stage memory cost

The reference table for memory budgeting is **`slopfinity/stats.py:95` (`_MODEL_GB`)**
and the harder per-stage budget table at **`slopfinity/scheduler.py:33` (`STAGE_BUDGETS`)**.
The two are not perfectly consistent — `stats.py` is used for the WILL-USE
breakdown the dashboard renders, `scheduler.py` is used for live admission
control. The scheduler numbers are the load-bearing ones; they include weights
+ activations, and `OVERHEAD_GB=6` (`slopfinity/scheduler.py:46`) gets added on
acquire.

| Stage | Model | Budget (GB) | Notes |
|-------|-------|-------------|-------|
| image | qwen | 28 | `STAGE_BUDGETS[("image","qwen")]` |
| image | ernie | 18 | |
| image | ltx-2.3 | 38 | base-image-via-LTX path |
| video | ltx-2.3 | 48 | per-chain |
| video | wan2.2 | 84 | |
| video | wan2.5 | 96 | |
| audio | heartmula | 14 | |
| tts | qwen-tts | 10 | |
| tts | kokoro | 8 | |
| upscale | ltx-spatial | 30 | |
| (all) | overhead | +6 | added on every acquire |
| safety | — | +10 | `SAFETY_GB`, `slopfinity/scheduler.py:47` |

### Co-residency analysis (128 GB ceiling)

Sums are budget + 2×OVERHEAD + SAFETY:

| Pair | Sum (GB) | Verdict |
|------|----------|---------|
| Image qwen + Audio heartmula | 64 | Fits comfortably. |
| Image qwen + TTS kokoro | 58 | Fits. |
| Video ltx-2.3 + Audio heartmula | 84 | Tight but fits. |
| Video ltx-2.3 + TTS qwen-tts | 80 | Fits. |
| Video ltx-2.3 + Upscale ltx-spatial | 100 | Marginal. |
| Video ltx-2.3 + Image qwen | 98 | Fits, contests host RAM. |
| Video wan2.2 + ANY GPU stage | 120+ | Single-tenant. |
| Concept (CPU) / Final Merge (CPU) + anything | n/a | GPU-free, free overlap. |

**Headline:** GPU is the constraint when both lanes hit it. CPU-only stages
(Concept via LM Studio; Final Merge via ffmpeg `-c copy` at `run_fleet.py:743`)
overlap with everything for free. The big win is job N+1's Concept while job
N is in Video Chains.

### Does the existing primitive support N-way?

**Yes.** `slopfinity/scheduler.py:57` (`GPUReservation`) already replaced the
binary `gpu_lock` with a budget-accounted condition variable. `acquire_gpu` at
`slopfinity/scheduler.py:236` increments `GPU.resident_gb`, suspends until
`projected_resident <= MemAvailable - SAFETY_GB`, and `notify_all`s on release.
Two simultaneous `async with acquire_gpu(...)` blocks already work
correctly *if the runner ever calls them concurrently*. The runner does not.

### Iterations vs. stages-within-an-iteration

Recommend **iteration-level**: pop two queue items, run their Concept→Final
Merge pipelines as independent async tasks, let `acquire_gpu` serialize where
budgets collide. Intra-iteration parallelism needs a DAG (Audio depends on
Final Merge; Post Process depends on the muxed output) — only Music+TTS are
meaningfully independent within one job. Iteration-level parallelism is
mostly stage-disjoint: while job N is in Video Chains (10 chains × LTX
video, the long pole), job N+1 can run Concept + Image + Audio + TTS in
parallel. The Phase-4 coordinator design (`slopfinity/coordinator.py:134`)
is already iteration-disjoint by construction — each `StageWorker` polls for
any item whose stage is `needs+prereqs-met`, regardless of iteration.

---

## 2. Runner refactor

### Today

`run_fleet.py:667` is a single `while True:` loop calling
`generate_prompt → run_image_gen → 10 × generate_video_ltx → ffmpeg merge → optional heartmula`.
There is exactly one in-flight iteration. The loop *does not* call
`acquire_gpu` — every stage shells out via `subprocess.run` against
ComfyUI/LTX/Wan launchers. Concurrency would require those subprocess calls to
be serialized through `acquire_gpu`, which they are not.

Meanwhile the alternative orchestrator at `slopfinity/coordinator.py:134`
already exists, with seven `StageWorker` subclasses
(`slopfinity/workers/{concept,image,video,audio,tts,post,merge}.py`) that *do*
plug into `acquire_gpu` via `slopfinity/workers/base.py:42`. But the coordinator
imports them through a non-existent `slopfinity.workers.stage_workers` module
(see `slopfinity/coordinator.py:37`); the import fails and the coordinator's
`run()` raises `RuntimeError` on day one. **The Phase-4 architecture is wired
but unboxed.**

### Three options

**(a) Sibling OS process when `concurrent_on` is set.** Cheap to imagine,
but `cfg.save_queue` (`slopfinity/config.py:289`) is a non-atomic
`open + json.dump` (two writers race trivially), and `GPUReservation` lives
in-process so cross-process budget needs an HTTP hook (Option A in
`docs/concurrent-mode-design.md`) or a file-locked reservation directory.
Also no shared `SchedulerEvents` queue for the WS.

**(b) `ThreadPoolExecutor(2)` inside `run_fleet.py`.** `acquire_gpu` is
asyncio-only, the legacy runner is sync subprocess-driven, and the runner's
mutable globals (`_iter_assets`, `_task_opts`, single `state.json` writer)
are not thread-safe. Lots of impedance.

**(c) The existing `slopfinity/coordinator.py` + per-stage workers.** Each
`StageWorker` is already async; `claim_next` (`slopfinity/workers/base.py:62`)
re-reads `queue.json` so two workers race on `pending` and the loser observes
the winner's `working` flip. One process, one event loop, one `GPU`
reservation singleton, one `SchedulerEvents` queue. Per-stage AND iteration
parallelism both fall out of the same fan. Caveats: Phase-4 is partially
landed — `slopfinity/coordinator.py:37` imports a `stage_workers` aggregator
module that *does not exist*; the legacy `run_fleet.py` and the new
coordinator both write to `state.json`; the infinite-requeue logic
(`run_fleet.py:773-893`, commit `6fcc549`) needs migrating into a post-merge
hook before the legacy runner can retire.

### Pick

**Option (c).** The codebase has been investing in Phase 1-4 of the
queueing refactor for months (see `docs/queueing-refactor-design.md` and
seven `slopfinity/workers/*.py` modules). (a) duplicates file-locking work;
(b) adds threading hazards on top of an already-async scheduler. (c) is the
path the codebase is already standing on — land the missing aggregator,
migrate requeue, decommission the legacy runner.

### Failure modes for (c)

| Failure | Mitigation |
|---|---|
| Two workers race to claim same item | Already handled — `claim_next` (`slopfinity/workers/base.py:62`) re-reads queue, sets `working` atomically, second worker re-scans and skips |
| `cfg.save_queue` not atomic, partial writes | Wrap with tempfile + os.replace in `slopfinity/config.py:289`; add `fcntl.flock` for inter-process safety even though we don't expect cross-process today |
| One worker crashes mid-stage, leaves item with `working` stage status | Add startup sweep (analogue of `run_fleet.py:653-664`) that resets `working` → `needs` on items whose `worker_id` no longer matches a live worker |
| Coordinator process dies, legacy runner restarts | Mutual exclusion via `coordinator.state.json` + `working` sentinel sweep on either runner's startup. Ship a single launch script that picks one |
| `acquire_gpu` budget table understates real peak; OOM under load | Already mitigated — live `_mem_available_gb()` re-poll on every wake (`slopfinity/scheduler.py:292`) |

### Working-sentinel + two workers

The infinite-requeue patch (`6fcc549`, `run_fleet.py:212-222`,
`run_fleet.py:773-893`) writes `status="working"` keyed by `task["ts"]` and
re-pulls it via `it.get("ts") == orig_ts`. Two issues with two workers:

1. **TS collision:** requeue stamps `ts = time.time() + 1e-6`
   (`run_fleet.py:874`). Two workers requeueing in the same microsecond
   collide. Migrate to keying on the Phase-1 schema's uuid `id` field.
2. **Ownership ambiguity:** the top-level `status="working"` row carries no
   `worker_id`. Use the per-stage `set_stage_status` (`slopfinity/workers/base.py:86`)
   which already records `worker`. The top-level `working` row becomes a
   back-compat artifact only during the migration window.

---

## 3. State schema

### Today

`slopfinity/config.py:266` (`set_state`) writes a flat:
```
{mode, step, video_index, total_videos, chain_index, total_chains, current_prompt, ts}
```
The runner calls `update_state(...)` at every transition (10+ call sites in
`run_fleet.py`, e.g. lines 670, 696, 714, 721, 732, 757; plus
`slopfinity/coordinator.py:172`).

### Target schema

```jsonc
{
  "mode": "Rendering",          // worst-of: max severity across active rows
  "active": [
    {
      "id": "<item-uuid>",       // queue item id (matches q.id)
      "step": "Video Chains",
      "role": "video",           // _STAGE_ROLE[step]
      "model": "ltx-2.3",
      "started_ts": 1740000000.123,
      "video_index": 17,
      "chain_index": 4,
      "total_chains": 10,
      "current_prompt": "…"
    },
    {
      "id": "<other-uuid>",
      "step": "Concept",
      "role": "llm",
      "model": "gemma-3-12b",
      "started_ts": 1740000010.456,
      "video_index": 18,
      "current_prompt": ""
    }
  ],
  "ts": 1740000020.789
}
```

### Who reads `state` today?

- `slopfinity/server.py:1601` `state = cfg.get_state()` — primary reader.
- `slopfinity/server.py:1603-1631` derives `_stage_track` / `_job_track` /
  `_job_stage_actuals` from `state.step` / `state.video_index` (singular).
  All those local-state-machines need to become per-`active[i].id`.
- `slopfinity/server.py:1696-1714` builds `render_heartbeat` from
  `state.step` / `state.mode`.
- `slopfinity/server.py:1115-1128` (`/asset/preview` prompt fallback) reads
  `state.current_prompt`.
- `slopfinity/static/app.js` — many call sites: `968-977` (modal prompt),
  `2464-2515` (header + progress), `2572-2575` (chain counter),
  `2654-2696` (per-stage spinner derivation), `2932` (run-row prompt),
  `3001-3046` (idle hint, stats), `4624-4640` (top-of-card chain counter).
- The `dark_server.py` shim (`dark_server.py:1` — 143 chars) does not read
  state.
- Clients also persist nothing — `STATE_FILE` (`slopfinity/config.py:259`)
  is recovered on dashboard restart, but every refresh starts a fresh WS.

### Migration / shim

Emit **both** legacy and new shapes for one release. The shim picks a
"primary" row from `active[]` (tie-break: prefer non-LLM, then most-recently
started) and writes its `step` / `video_index` / `chain_index` /
`total_chains` / `current_prompt` into the legacy top-level fields. Every
existing reader keeps working; new readers consume `state.active`. Shim
removed in a follow-up PR once all ~12-15 `app.js` sites are converted.

### `render_heartbeat`

`slopfinity/server.py:1710-1714` emits one event with one `text`. Evolve
to one event whose payload is an array: `{type, active: [{id, text,
expires_ts}, …]}`. Atomic w.r.t. the WS broadcast and cheaper than N
events per tick. Client loops, looks up each `runItem` by `id`.

---

## 4. UI changes

### Multiple synthetic running rows

`slopfinity/static/app.js:2944-2949` injects one `runItem` (ts:0) at the
top. Two options: (a) push N synthetics with ts:0+i; (b) render the live
`working` items in-place via the per-stage schema. Recommend **(b)** — the
queue is the truth, no parallel synthetic universe, no dual-world
confusion. Renderer checks `if any item.stages.<x>.status === 'working'`
and lifts those items to the top with active styling.

### Per-row spinner placement

`_configModelBadges` (`slopfinity/static/app.js:1921`) takes `activeRole`
+ `qTs`. The caller at `app.js:2824` already passes `q.ts` — only change
is to derive `activeRole` from `state.active.find(a => a.id === q.id)`
instead of the global `state.step`. Trivial once the Phase-1 schema's
stable `id` is plumbed.

### Top-of-card progress bar

`_buildActiveJobProgressBar` (`slopfinity/static/app.js:690`) renders ONE bar
for one active job. With two active jobs the question is: stack two bars,
or render one bar showing the slowest? Recommend **stack**: the card already
hosts the bar at `#active-job-progress-bar` (line 2982) which can become a
container for N bars. Each bar inherits the existing segmented-stage styling;
labels carry `id` so click-to-scroll works. Cap at 2 visible (the practical
ceiling — you can't fit a third LTX 48-GB job in 128 GB anyway).

---

## 5. Risks ranked

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | `cfg.save_queue` not atomic; two workers stomp each other's writes | High | Make `slopfinity/config.py:289` atomic (tempfile + `os.replace`). Add `fcntl.flock` if cross-process is ever real. The Phase-1 schema's per-item write-merge in `slopfinity/workers/base.py:_finalize_sync` re-reads, so two workers serializing through the file is correct *if* the write itself is atomic. |
| 2 | Working-sentinel TS collision with two workers requeueing in same µs | High | Migrate the requeue path (`run_fleet.py:874`) to key on item `id` (uuid) not `ts`. The `id` field already exists per Phase-1 schema. |
| 3 | VRAM thrash when both lanes hit GPU stages | Medium | `acquire_gpu` already serializes when budgets don't fit. But the live-poll fallback assumes ComfyUI's `/free` endpoint responds — `slopfinity/scheduler.py:319` calls `free_between()` on first block. Verify ComfyUI is reachable before declaring concurrent-mode safe. |
| 4 | Legacy state-reading JS paths see `step="Waiting"` while two real stages run | Medium | Compat shim (Section 3) emits both legacy and new fields. JS conversion is mechanical. |
| 5 | `scripts/orphan_vid_watcher.sh` heuristic mis-fires on a different chain mid-render | Medium | Today the watcher matches `vid_<ts>_*.png` patterns. With two parallel renders, two distinct `<ts>` directories coexist — but the watcher's per-ts encode-when-stable logic is per-ts-keyed, so it should be robust. **Verify** by manual test: trigger a parallel run with two image-only items to make sure the watcher doesn't grab job-2's frames as job-1 orphans. |
| 6 | `STATE_FILE` reload after restart shows the LAST iteration only, not the in-flight set | Low | The new `state.active` is rebuilt from `queue.json` on coordinator startup (each item's stages.* tells you what's `working`). The persisted `state.json` becomes a UI-recovery hint only, not authoritative. |
| 7 | LLM rewriter runs twice in parallel, hammers the LM Studio process | Low | LM Studio handles concurrent requests serially anyway; worst case is added latency, not failure. Add a `Concept` stage soft-cap in `STAGE_BUDGETS` if needed. |
| 8 | `update_state` calls happen from sync subprocess paths in `run_fleet.py`; threading them is not safe | Low | Only relevant if Option (b) were picked; we picked (c) so the legacy runner gets retired wholesale. |

---

## 6. Implementation sequencing

Each step is independently mergeable + testable. Effort in T-shirts.

| Step | Scope | Effort | Deliverable |
|------|-------|--------|-------------|
| **1** | Make `cfg.save_queue` atomic (tempfile + `os.replace`). Add a unit test that two threads writing simultaneously never produce malformed JSON. (`slopfinity/config.py:289`) | S | Latent prerequisite — also helps single-runner reliability today. |
| **2** | Land the multi-active state schema with shim. Modify `slopfinity/config.py:266` (`set_state`) to accept either the legacy positional args (back-compat) or a new `set_state_active(active, mode)` API. Emit BOTH new and legacy fields. Adapt `slopfinity/server.py:1601-1631` to read `active[]` if present. **No JS changes yet.** | M | Server-side ready, client unchanged — no behavior change. |
| **3** | Convert `app.js` readers to consume `state.active[]` with fallback to legacy fields. Site-by-site mechanical change at the call sites enumerated in §3. Renderer treats `state.active.length > 1` as a no-op (stays single-row) until the runner actually produces multi-row state. | M | Client ready, still single-row in practice. |
| **4** | Create the missing `slopfinity/workers/stage_workers.py` aggregator that the coordinator imports. Just re-exports the seven worker classes. Wire `slopfinity/coordinator.py:37` to import from it. Add a `--dry-run` mode that polls the queue but no-ops `run_stage`. | S | Coordinator becomes startable but inert. |
| **5** | Migrate the infinite-requeue logic out of `run_fleet.py:773-893` and into a `MergeWorker` post-finalize hook (or a separate `RequeueWorker`). Key on `item.id`, not `item.ts`. Preserve all toggle + cancel-mid-flight semantics from `run_fleet.py:809-823`. | M | Coordinator can run infinity items end-to-end with a single worker. |
| **6** | Add a feature flag (`config.scheduler.coordinator_workers = 1` default). When raised to 2, `slopfinity/coordinator.py:159` instantiates a second instance of *every* worker (or, more cheaply, just two `ConceptWorker`s + two `ImageWorker`s + two `VideoWorker`s — the long poles). Single-worker behavior preserved by default. | M | Code is in tree, off by default. |
| **7** | Wire the `concurrent-on` UI toggle to flip `coordinator_workers=2` (and back). Today's toggle (`slopfinity/templates/index.html:1599`) just persists a per-item `concurrent` bool; we generalize it to a global scheduler hint. | S | UI control over concurrency. |
| **8** | Decommission the legacy `run_fleet.py`. Replace with a thin shim that `exec`s `python -m slopfinity.coordinator`. Keep the file because `.gitignore`'d launchers reference it. | S | One orchestrator, one truth. |
| **9** | Remove the legacy-state shim from Step 2. (`slopfinity/config.py:266` becomes `set_state(active, mode)` only.) | S | Schema is clean. |

Total: ~5 PRs, the riskiest being Step 5 (requeue migration). Steps 1-3 are
zero-behavior-change refactors that can land at any time.

---

## 7. Quick-win alternative

A purely visual improvement ships in one PR with no backend changes.

**Goal:** make the queue card LOOK concurrent — show all completed-stage
badges as styled-done, the active one as glowing, future as dimmed.
The data is already there: `slopfinity/server.py:1610` populates
`stage_actuals` per video_index.

In `_configModelBadges` (`slopfinity/static/app.js:1921`), color each
badge by stage state (done / active / pending / skipped) from
`stage_actuals`. Optionally, when `concurrent-on` is set, pre-glow the
LLM badge on the *next* queue item while the current is in Final Merge —
cosmetic, but reads well.

Effort: **S**. No backend, no schema, no risk to working-sentinel logic.
Defers but does not avoid the proper refactor.

---

## RECOMMENDATION

**Proceed, on path (c) with sequencing 1→2→3→4→5→6→7→8→9.** The reservation
primitive (`slopfinity/scheduler.py:57`) and the per-stage worker classes
(`slopfinity/workers/*.py`) are already written and partially tested; the
gap is a missing aggregator (`stage_workers.py`), the legacy runner's
unmigrated requeue logic, and the `state` schema flattening. The riskiest
single change is **Step 5** (migrating infinite-requeue out of
`run_fleet.py:773-893`); everything else is mechanical. Steps 1-3 are
zero-behavior refactors that can land independently as wins on their own
even if the rest stalls. If the user wants a 1-week win without committing
to the full path, ship the Quick-Win in §7 first — it materially improves
the pipeline-readability today and does not foreclose the proper refactor
later.
