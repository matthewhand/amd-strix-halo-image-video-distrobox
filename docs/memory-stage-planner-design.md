# Memory-stage Planner — Design

## Problem

Slopfinity runs a fixed sequence of GPU stages per video:

```
Concept (LLM) → Base Image → Video Chains → Audio (Music) → TTS (Voice)
              → Post Process → Final Merge
```

Each stage uses **exactly one model** drawn from a known role
(`image`, `video`, `audio`, `tts`, `upscale`, …). Within one video, no model
repeats. Across videos in a fleet, the **same model often recurs** — every
video's Image stage uses Qwen, every video's Video stage uses LTX, etc.

Today `scheduler.py::acquire_gpu` serializes stages with a single `gpu_lock`
and calls `free_between()` (POSTs `/free` to ComfyUI + `gc.collect()`) after
every stage. That's safe but **always cold-loads the next model** — paying
the ~90 s aiter JIT build + ~90 s checkpoint load on every transition.

We have 128 GB of unified memory. We can keep multiple models resident at
once and *skip* the cold-load when the next stage's model is already there.
But we have to evict aggressively when the next stage needs more headroom.

This is **cache replacement**. The pipeline gives us perfect lookahead for
the active job, plus the queue tells us 1–N future jobs. So we can do better
than LRU — we can do **Belady's MIN**.

## Pipeline as a sequence

A fleet run flattens into a sequence of `StageStep`s, each carrying a peak
GB footprint:

```
v0/Image    qwen        28 GB
v0/Video    ltx-2.3     48 GB
v0/Audio    heartmula   14 GB
v0/TTS      qwen-tts    10 GB
v1/Image    qwen        28 GB    ← same model as v0/Image
v1/Video    ltx-2.3     48 GB
…
```

The `gb` values come from the existing `STAGE_BUDGETS` table in
`scheduler.py` (peak resident). The total budget is

```
budget = MEM_AVAILABLE - SAFETY_GB - OVERHEAD_GB
       ≈ 128 - 10 - 6 = 112 GB
```

at boot, less whatever the OS / dashboard already hold.

## Algorithm: Belady's MIN over the lookahead window

For each step `i` we hold a **resident set** `R_i`. Walking left to right:

1. If `step.model ∈ R_{i-1}`: free hit. `R_i = R_{i-1}`.
2. Else, while `sum(gb of R_{i-1}) + step.gb > budget`:
   - **Pick a victim**: the resident model whose **next reference in the
     remainder of the sequence is furthest** (or that never recurs). Tie
     break in favour of evicting the largest GB (frees more room per
     eviction).
   - Drop it from `R_{i-1}`.
3. Add `step.model` to the resident set.

This is Belady's MIN — provably optimal *offline* for a known sequence
(Belady, 1966). Slopfinity's queue gives us exactly that: full visibility of
the active job's stages and the next queued items.

### Why not LRU?

LRU would evict the model used least recently. That misses the obvious
case where Qwen was just used for v0/Image and is about to be used for
v1/Image — LRU would happily evict it after v0/Audio uses Heartmula, and
we'd cold-load it again. MIN sees `v1/Image: qwen` ahead and keeps it.

### Why not full clairvoyant MIN over the whole queue?

We *could* — but the queue can be very long, and replanning cost grows
linearly. **Sliding-window MIN over `current_job + next_1_or_2_queued`** is
the sweet spot: ~12–36 stages of lookahead, identical decisions to full MIN
in 99 % of cases (the optimal eviction victim is almost always determined
by what happens in the next handful of stages, not 50 stages out).

## Trade-offs

- **Single-job benefit: zero.** Each model is used exactly once per video.
  MIN inside one video collapses to "evict the model we already finished
  with" — same as the current `free_between()` after each stage.
- **Multi-job benefit: ~90 s × N saved boundaries.** If Qwen Image is the
  base model for 5 consecutive videos, we save ~5×90 s = 7.5 minutes of
  cold-loads (roughly — the actual savings depend on whether Qwen and the
  next stage's model fit together).
- **Memory pressure events.** When the user enables `wan2.5` (84 GB peak)
  the planner will be forced to evict almost everything else — report
  these eviction events so the user sees *why* the pipeline cold-loaded.
- **Mid-fleet model swaps.** If the user toggles `base_model` from `qwen`
  to `ernie` mid-run, the planner replans from the next step. Already-in-
  flight stages are not interrupted.
- **Pessimistic budgets.** `STAGE_BUDGETS` are peaks during inference; a
  quiescent resident model uses less. The planner uses the peak so it
  never plans an OOM, which means it sometimes evicts more than strictly
  necessary. Acceptable trade.

## Where this slots in

- **Phase 1 (this PR): advisory only.** Add `slopfinity/memory_planner.py`
  + `GET /pipeline/plan` + a "Load plan" expandable in the Pipeline popup.
  The dashboard surfaces the plan; the actual scheduler keeps its current
  free-after-every-stage behaviour. The user inspects the plan and decides
  whether to wire it in.
- **Phase 2 (follow-up):** make `acquire_gpu` honour the plan — skip the
  `free_between()` call when the next step's model is the same as the just-
  released one, and trigger eviction only when the plan says so. Requires
  thinking through interaction with the budget-accounted Condition from
  PR #61's concurrent-mode design.

## API (`slopfinity/memory_planner.py`)

```python
@dataclass
class StageStep:
    stage: str   # "Base Image"
    role: str    # "image"
    model: str   # "qwen"
    gb: float    # peak GB, from STAGE_BUDGETS

@dataclass
class PlanDecision:
    step: StageStep
    load: list[str]            # cold-loaded before this step
    keep: list[str]            # already resident (free)
    evict: list[str]           # dropped to make room
    resident_after: list[str]  # snapshot post-step

def plan_resident_set(
    sequence: list[StageStep],
    budget_gb: float,
    initial_resident: list[StageStep] | None = None,
) -> list[PlanDecision]: ...
```

Pure stdlib; no async; no I/O. The dashboard endpoint composes the
sequence from `cfg.load_config()` + `cfg.get_queue()` and passes
`MEM_AVAILABLE - SAFETY_GB - OVERHEAD_GB` as the budget.

## Tests

- single video → no inter-stage savings (every step is a load)
- two-video sequence sharing Image model → planner keeps Qwen across the
  boundary, exactly one fewer load than naïve
- budget too small for two big models → exactly one eviction at the right
  step; never an OOM
- never evicts a model that's about to be used in the next step (MIN
  guarantees this, the test asserts it)
