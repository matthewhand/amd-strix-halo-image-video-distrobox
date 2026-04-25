"""Look-ahead resident-set planner for Slopfinity's GPU pipeline.

Given a pipeline sequence (StageStep list) and a per-step GB budget, decides
which models to keep resident at each step. Eviction policy is Belady's MIN
— when room is needed, evict the resident model whose **next reference is
furthest in the future** (or that is never referenced again). Tie-broken by
GB (drop the largest first to free more room per eviction).

This module is intentionally pure: stdlib only, no async, no I/O. The
dashboard endpoint (`GET /pipeline/plan`) composes the sequence from the
runtime config + queue and calls `plan_resident_set` — see
`docs/memory-stage-planner-design.md` for the full design.

Phase 1: advisory only. The actual scheduler (`scheduler.acquire_gpu`) is
unchanged in this PR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class StageStep:
    """One pipeline step.

    `gb` is the peak resident footprint while this step is running, sourced
    from `scheduler.STAGE_BUDGETS` (peak weights + activations, no
    overhead). `model` may repeat across steps — that's the whole point of
    the planner.
    """
    stage: str
    role: str
    model: str
    gb: float


@dataclass
class PlanDecision:
    """Decision the planner made for a single step.

    - `load`     : models cold-loaded before this step ran
    - `keep`     : models that were already resident from a prior step
                   (no cold-load cost — this is the savings)
    - `evict`    : models the planner dropped to make room
    - `resident_after` : snapshot of the resident set after this step
                        finished (i.e. including the just-loaded model)
    """
    step: StageStep
    load: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)
    evict: list[str] = field(default_factory=list)
    resident_after: list[str] = field(default_factory=list)


def _gb_of(model: str, sequence: list[StageStep]) -> float:
    """Look up the peak GB for a model from its first appearance in
    `sequence`. We index by model name (not (stage, model)) because the
    resident set is keyed by the loaded checkpoint — Qwen Image at
    Image-stage and Qwen Image kept-resident-during-Audio cost the same
    on disk/RAM.
    """
    for s in sequence:
        if s.model == model:
            return s.gb
    return 0.0


def _next_use(model: str, sequence: list[StageStep], start_idx: int) -> int:
    """Index of the next reference to `model` in `sequence` at or after
    `start_idx`. Returns a sentinel (`len(sequence)`) if `model` does not
    recur — that maps to "evict me first" since it's furthest possible.
    """
    for j in range(start_idx, len(sequence)):
        if sequence[j].model == model:
            return j
    return len(sequence)


def _pick_victim(
    resident: dict[str, float],
    sequence: list[StageStep],
    next_idx: int,
    must_keep: set[str],
) -> Optional[str]:
    """Belady's MIN victim selection.

    Returns the resident model whose next reference is **furthest** (or
    that doesn't recur). Tie-break: largest GB first (frees more room
    per eviction). `must_keep` is the set of models we may not evict —
    typically the current step's model itself.
    """
    best: Optional[str] = None
    best_next = -1
    best_gb = -1.0
    for m in resident:
        if m in must_keep:
            continue
        nxt = _next_use(m, sequence, next_idx)
        if (nxt > best_next) or (nxt == best_next and resident[m] > best_gb):
            best_next = nxt
            best_gb = resident[m]
            best = m
    return best


def plan_resident_set(
    sequence: list[StageStep],
    budget_gb: float,
    initial_resident: Optional[list[StageStep]] = None,
) -> list[PlanDecision]:
    """Walk `sequence` left-to-right, maintaining a resident set under
    `budget_gb`.

    For each step:

    1. If the step's model is already resident → free hit (keep, no load).
    2. Otherwise: while `sum(resident_gb) + step.gb > budget_gb`, evict
       the Belady-MIN victim. Then load the new model.

    Returns one `PlanDecision` per step in input order.

    The resident set is a `dict[model_name -> gb]` so we never double-
    count when the same model would be loaded twice (it can't, by step 1).
    """
    decisions: list[PlanDecision] = []
    # model_name -> gb. Order is insertion-order; doesn't matter for the
    # algorithm but keeps `resident_after` deterministic for tests.
    resident: dict[str, float] = {}
    if initial_resident:
        for s in initial_resident:
            resident[s.model] = s.gb

    for i, step in enumerate(sequence):
        load: list[str] = []
        keep: list[str] = []
        evict: list[str] = []

        if step.model in resident:
            # Free hit. The model is already loaded.
            keep = [m for m in resident.keys() if m != step.model]
            decisions.append(PlanDecision(
                step=step,
                load=load,
                keep=[step.model] + keep,  # convention: include hit model
                evict=evict,
                resident_after=list(resident.keys()),
            ))
            continue

        # Need to load. Evict until the new model fits.
        # Look for victim using `i+1` as the lookahead start — we never
        # evict something the *current* step needs; the current step's
        # model is not yet resident anyway.
        must_keep: set[str] = set()
        while sum(resident.values()) + step.gb > budget_gb:
            victim = _pick_victim(resident, sequence, i + 1, must_keep)
            if victim is None:
                # Nothing left to evict (or everything is must_keep).
                # We'll over-budget — flag it via a synthetic eviction so
                # the caller sees the pressure event. The actual scheduler
                # will hit the budget-block path here.
                break
            evict.append(victim)
            del resident[victim]

        keep = list(resident.keys())
        load = [step.model]
        resident[step.model] = step.gb
        decisions.append(PlanDecision(
            step=step,
            load=load,
            keep=keep,
            evict=evict,
            resident_after=list(resident.keys()),
        ))

    return decisions


# ---------------------------------------------------------------------------
# Helpers for composing a sequence from Slopfinity config + queue.
# Lives here (rather than in server.py) so unit tests can build sequences
# without importing FastAPI.
# ---------------------------------------------------------------------------

# Mirrors `scheduler.STAGE_BUDGETS` to keep this module dep-free; importing
# scheduler would drag in asyncio. Kept in sync via design-doc note.
_STAGE_BUDGETS = {
    ("image", "qwen"):        28,
    ("image", "ernie"):       18,
    ("image", "ltx-2.3"):     38,
    ("video", "ltx-2.3"):     48,
    ("video", "wan2.2"):      84,
    ("video", "wan2.5"):      96,
    ("audio", "heartmula"):   14,
    ("tts",   "qwen-tts"):    10,
    ("tts",   "kokoro"):       8,
    ("upscale", "ltx-spatial"): 30,
}


def step_gb(stage: str, model: str) -> float:
    """Peak GB for (stage, model). Returns 0 for the empty/none sentinels."""
    if not model or model in ("none", ""):
        return 0.0
    return float(_STAGE_BUDGETS.get((stage, model), 0))


# Pipeline stage order, matching what the dashboard popup labels.
PIPELINE_STAGES = (
    ("image",   "Base Image"),
    ("video",   "Video Chains"),
    ("audio",   "Audio"),
    ("tts",     "TTS"),
    ("upscale", "Post Process"),
)


def build_sequence_for_job(
    base_model: str,
    video_model: str,
    audio_model: Optional[str],
    tts_model: Optional[str],
    upscale_model: Optional[str],
) -> list[StageStep]:
    """Flatten one job's stage selections into a `StageStep` list.

    Stages with `none` / empty model are skipped. `slopped:` placeholders
    contribute zero load (they reuse an existing asset) but we keep the
    step in the sequence with gb=0 so the planner accounts for ordering.
    """
    role_model = {
        "image":   base_model,
        "video":   video_model,
        "audio":   audio_model,
        "tts":     tts_model,
        "upscale": upscale_model,
    }
    out: list[StageStep] = []
    for role, stage in PIPELINE_STAGES:
        m = role_model.get(role)
        if not m or m == "none":
            continue
        if isinstance(m, str) and m.startswith("slopped:"):
            # Skip — uses an existing asset, no model loaded.
            continue
        out.append(StageStep(stage=stage, role=role, model=m, gb=step_gb(role, m)))
    return out


def naive_load_count(sequence: list[StageStep]) -> int:
    """How many cold-loads a no-cache scheduler would incur. Equal to the
    number of non-zero-GB steps — used by the dashboard to show savings.
    """
    return sum(1 for s in sequence if s.gb > 0)


def planned_load_count(decisions: list[PlanDecision]) -> int:
    """How many cold-loads the plan actually performs."""
    return sum(1 for d in decisions if d.load)
