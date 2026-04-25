# Queueing refactor — design notes

This is the running design doc for the four-phase queueing refactor that
replaces the linear fleet runner with a fan of concurrent stage workers
backed by a per-stage queue.

## Motivation

The legacy `run_philosophical_experiments.py` fleet runner walks one job
through every stage in a fixed order before picking up the next item.
That serializes:

1. **Across-job concurrency** — even when stage budgets fit
   side-by-side (e.g. Heartmula audio for job N+1 while job N is
   still in Final Merge), nothing runs in parallel.
2. **CPU-only stages behind GPU stages** — Concept (LLM) and Final
   Merge (ffmpeg) sit in the same loop as Image/Video, so a long
   video render stalls the entire fleet.
3. **Recovery** — a single failure at any stage kills the whole job
   row; there's no per-stage retry surface.

## Phases

| Phase | Scope | PR / status |
|-------|-------|-------------|
| 1 | `queue_schema` — per-item `stages.<stage>.status` map (pending/running/done/failed/skipped), migration from legacy queue.json | TBD |
| 2 | `StageWorker` base class — poll loop, claim/release, status transitions, failure handling | TBD |
| 3 | Per-stage workers (`ConceptWorker`, `ImageWorker`, `VideoWorker`, `AudioWorker`, `TTSWorker`, `PostWorker`, `MergeWorker`) | TBD |
| 4 | **Coordinator** — concurrent worker fan, dashboard endpoints, dashboard rendering, CLI | This PR |

Phase 4 lands with **defensive imports** so it can merge before phases 1-3
(the `Coordinator.run()` raises a clear error until they land, and
`/coordinator/start` returns 503 with the underlying ImportError).

## Phase 4 — Coordinator

```python
class Coordinator:
    def __init__(self):
        self.workers = [
            ConceptWorker("llm-w0"), ImageWorker("img-w0"),
            VideoWorker("vid-w0"), AudioWorker("aud-w0"),
            TTSWorker("tts-w0"), PostWorker("post-w0"),
            MergeWorker("merge-w0"),
        ]
    async def run(self):
        tasks = [asyncio.create_task(w.loop(poll_interval_s=2.0)) for w in self.workers]
        await asyncio.gather(*tasks)
```

GPU serialization is unchanged — `scheduler.acquire_gpu` still gates
budget-conflicting stages. The coordinator only removes the
single-threaded *job ordering* constraint, not the GPU lock.

### Dashboard control

- `POST /coordinator/start` — spawn the worker fan (idempotent).
- `POST /coordinator/stop` — cancel worker tasks; honours `acquire_gpu`'s
  `try/finally` so the GPU lock is released cleanly.
- `GET /coordinator/status` — running flag, worker list, import health.

State persists to `comfy-outputs/experiments/coordinator.state.json` so a
slopfinity restart can show the previous running flag (auto-restart on
boot is intentionally **not** wired — the user opts in each session).

### Dashboard rendering

`renderPipelineStrip` and `_renderDoneItem` now read per-stage status
from `item.stages.<stage>.status`. Multiple items can be running at
once, each with different stages active. The legacy linear `state.step`
model is kept as a fallback for items written before the refactor lands.

A new `pipeline-seg-failed` segment class surfaces partial failures —
e.g. an item with Image done, Video failed, Audio done renders that
mosaic correctly instead of collapsing to a single "failed" badge.

### CLI

`python -m slopfinity.coordinator` runs the coordinator standalone,
replacing the legacy fleet runner for users who want the new
architecture. Flags:

- `--poll-interval SECONDS` — StageWorker queue-poll cadence (default 2.0).
- `--log-level LEVEL` — Python logging level (default INFO).

## Migration

> **The legacy fleet runner (`run_philosophical_experiments.py`) is
> deprecated as of Phase 4.**

Both code paths remain shipped during the transition:

- **Recommended:** `python -m slopfinity.coordinator` (or click "Start
  Coordinator" on the dashboard once the UI button lands).
- **Legacy:** `python run_philosophical_experiments.py` continues to
  work for users who haven't migrated yet.

Do not run both at once — they would race on the same `queue.json`.

The deprecation is not yet annotated in the legacy script itself; that
will land alongside the Phase-1 schema migration so the runner can read
the per-stage status it needs to write.

## Open questions

- Should the coordinator auto-start on slopfinity boot when the
  persisted running flag is `true`? Current answer: no, the user opts
  in each session to avoid surprises after a crash.
- What's the right number of workers per stage? Today it's 1-of-each;
  CPU stages (Concept, Post Process, Final Merge) could profitably run
  N>1 once the queue claim semantics from Phase 2 are in place.
- Does the existing `concurrent` flag on queue items still mean
  anything once the coordinator is the default? Probably no — it
  becomes a no-op and we can remove it in a follow-up.
