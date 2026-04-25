# Queueing Refactor — Design Doc

Status: Phase 1 (typed schema + migration) implemented. Phases 2–5 follow.

## Why

The current pipeline runs as a single linear process: one queue item at a
time, each item walking concept → image → video → audio → tts → post →
merge in a hardcoded sequence inside `slopfinity/workers.py`. This works
but is leaving substantial throughput on the floor:

- **No batching by model.** Two consecutive items both needing the LTX
  video model still pay the full model-load cost twice if any other stage
  ran between them. Conversely a wave of "image-only" items still waits
  in line behind an in-flight video job.
- **No stage-level parallelism.** While the GPU is busy on a video step,
  the CPU-bound `merge` (ffmpeg) for the previous item sits idle, and
  the LLM (`concept`) for the next item likewise waits. The
  memory-stage-planner (`docs/memory-stage-planner-design.md`) already
  knows which roles can co-exist; the runner just doesn't use it.
- **Failure granularity.** A failure in `merge` today re-runs the whole
  item including the (expensive) video step. Per-stage status would let
  us resume from the failed stage.
- **Auto-suspend churn.** Linear walking forces LM Studio / ComfyUI to
  suspend-resume for every item; a coordinator could group same-role
  work and suspend siblings once per batch.

## Target architecture

- **Typed queue items** — every item carries a `stages` map keyed by
  the seven canonical stages (concept, image, video, audio, tts, post,
  merge), each with its own status (`needs` | `working` | `done` |
  `failed` | `skipped`) and per-stage metadata (model, started_ts,
  completed_ts, error). The legacy top-level `status` / `succeeded`
  fields become a derived view (`overall_status`).
- **Per-role workers** — one worker process per role (`llm`, `image`,
  `video`, `audio`, `tts`, `post`, `ffmpeg`). Each worker pulls the
  next ready stage for its role from the queue, runs it, and writes
  back per-stage status. Workers stay alive so the model loaded into
  VRAM persists between adjacent same-role items (free batching).
- **Scheduler-aware coordinator** — a thin coordinator wakes workers
  when their role fits the current memory plan (see
  `docs/memory-stage-planner-design.md` and
  `docs/concurrent-mode-design.md`) and pauses them otherwise.
  Auto-suspend (`slopfinity/auto_suspend.py`) hangs off the same
  signal so co-resident services suspend per-role-batch instead of
  per-item.
- **Prerequisite graph** — declarative `PREREQS` (concept → image →
  video → post; concept → audio; concept → tts; merge needs post +
  audio + tts) replaces the sequential walk. The coordinator only
  hands a stage to a worker when its prereqs are `done` or `skipped`.

## Schema spec (v2)

Each queue item, post-migration:

```json
{
  "id": "q-1714000000000-ab12cd",
  "schema_version": 2,
  "config_snapshot": { ...same as today... },
  "stages": {
    "concept": {"status": "done",    "started_ts": ..., "completed_ts": ...},
    "image":   {"status": "done",    "model": "qwen", ...},
    "video":   {"status": "working", "model": "ltx-2.3", "started_ts": ...},
    "audio":   {"status": "needs",   "model": "heartmula"},
    "tts":     {"status": "needs",   "model": "qwen-tts"},
    "post":    {"status": "needs",   "model": "ltx-spatial"},
    "merge":   {"status": "needs"}
  }
}
```

Image-only items get `skipped` set on `video`/`audio`/`tts`/`post`/`merge`
at migration time so the coordinator never tries to schedule them.

`overall_status(item)` derives the legacy bulk status:
- all `done`/`skipped` (and at least one `done`) → `done`
- any `failed` → `failed`
- any `working` → `running`
- otherwise → `pending`

## Five-phase rollout

1. **Schema + migration (this PR).** Land `slopfinity/queue_schema.py`,
   wire `config.get_queue` to migrate legacy items in place on first
   read, ship tests. No runtime behaviour changes — the runner still
   ignores `stages` and reads the legacy fields.

2. **Worker base.** Introduce `slopfinity/workers/base.py` with a
   `RoleWorker` ABC: `claim_next()`, `run(item, stage)`, `mark_done()`,
   `mark_failed()`. Provide a `LinearDriver` that walks all roles
   single-threaded — equivalent to today's runner, but reading and
   writing per-stage status. Ship behind `SLOPFINITY_USE_WORKERS=1`.

3. **Stage workers.** Port each stage's existing logic out of
   `workers.py` into a dedicated `RoleWorker` subclass
   (`workers/llm.py`, `workers/image.py`, ...). Still single-threaded;
   the linear driver just dispatches by role.

4. **Coordinator.** Replace `LinearDriver` with `Coordinator` that
   consults `memory_planner` to pick the next role to run, and groups
   adjacent same-role stages into a batch (worker stays warm). Still
   one role active at a time, but auto-suspend now fires per-batch and
   model loads amortise across items.

5. **Concurrent unlock.** Allow N roles to run simultaneously when
   the memory plan permits (e.g. `merge` (ffmpeg, CPU) overlapping
   with `video` (GPU)). Roll out gated by `SLOPFINITY_CONCURRENT=1`.

## Migration notes

- `config.get_queue()` is the single read path for `queue.json`. It
  now migrates each item via `queue_schema.migrate_legacy(item)` and
  re-saves the file when any item lacked `schema_version: 2`.
  Migration is idempotent — items already at v2 are returned
  untouched.
- Legacy fields (`status`, `succeeded`, `image_only`) are preserved
  on the item; only `stages` and `schema_version` are added. Phase 2+
  workers will read `stages` exclusively; the dashboard keeps reading
  the legacy fields until Phase 4.
- `image_only` is the only non-trivial mapping: those items get
  `video`/`audio`/`tts`/`post`/`merge` marked `skipped` so the
  coordinator never offers them to a non-image worker.
- Done/failed/cancelled legacy items map to `done`/`failed`/`skipped`
  on **every** stage — this means a re-run won't retry old failures
  unless the user explicitly resets the per-stage status (a Phase 4
  dashboard control).
- The migration is safe to run on the live
  `comfy-outputs/experiments/queue.json`; the only mutation is adding
  `id`, `schema_version`, and `stages` keys.
