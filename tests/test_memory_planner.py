"""Unit tests for `slopfinity.memory_planner`.

Covers the four invariants from the design doc:

1. Single-job sequence has zero inter-stage savings (every step is a load).
2. Two videos sharing the Image model → planner keeps it across the
   boundary, exactly one fewer load than naïve.
3. Budget too small for two big models simultaneously → exactly one
   eviction at the right step; never an OOM-by-plan.
4. Belady's MIN never evicts a model used by the *very next* step.
"""
import os
import sys

# Allow `python -m pytest tests/test_memory_planner.py` from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slopfinity.memory_planner import (
    PlanDecision,
    StageStep,
    build_sequence_for_job,
    naive_load_count,
    plan_resident_set,
    planned_load_count,
)


def _qwen():     return StageStep("Image", "image", "qwen", 28)
def _ltx():      return StageStep("Video", "video", "ltx-2.3", 48)
def _heartm():   return StageStep("Audio", "audio", "heartmula", 14)
def _qtts():     return StageStep("TTS",   "tts",   "qwen-tts", 10)


# ---------------------------------------------------------------------------
# 1. Single video → no inter-stage savings
# ---------------------------------------------------------------------------
def test_single_video_every_step_is_a_load():
    seq = [_qwen(), _ltx(), _heartm(), _qtts()]
    plan = plan_resident_set(seq, budget_gb=112)
    # Every step uses a different model, so every step must load.
    assert [d.load for d in plan] == [["qwen"], ["ltx-2.3"], ["heartmula"], ["qwen-tts"]]
    # No evictions if the budget fits everything (28+48+14+10 = 100 < 112).
    assert all(d.evict == [] for d in plan)
    assert planned_load_count(plan) == naive_load_count(seq) == 4


# ---------------------------------------------------------------------------
# 2. Two videos sharing Image model → keep Qwen across the boundary
# ---------------------------------------------------------------------------
def test_shared_image_model_kept_across_boundary():
    v0 = [_qwen(), _ltx(), _heartm(), _qtts()]
    v1 = [_qwen(), _ltx(), _heartm(), _qtts()]
    seq = v0 + v1
    plan = plan_resident_set(seq, budget_gb=112)

    # Step 4 (v1/Image, qwen) must be a HIT, not a load — that's the savings.
    v1_image = plan[4]
    assert v1_image.step.model == "qwen"
    assert v1_image.load == []
    assert "qwen" in v1_image.keep

    # Total cold-loads should be < naive.
    assert planned_load_count(plan) < naive_load_count(seq)


# ---------------------------------------------------------------------------
# 3. Tight budget → exactly one eviction at the right step
# ---------------------------------------------------------------------------
def test_tight_budget_forces_single_eviction():
    # Budget = 60 GB. Qwen (28) + LTX (48) = 76 > 60 → must evict before LTX.
    seq = [_qwen(), _ltx()]
    plan = plan_resident_set(seq, budget_gb=60)

    assert plan[0].evict == []          # first load always fits
    assert plan[1].evict == ["qwen"]    # exactly one eviction
    assert plan[1].load == ["ltx-2.3"]
    # Resident set after step 2 contains only LTX.
    assert plan[1].resident_after == ["ltx-2.3"]


# ---------------------------------------------------------------------------
# 4. MIN never evicts a model the next step needs
# ---------------------------------------------------------------------------
def test_never_evicts_immediate_next_step_model():
    # Construct a case where naive LRU would evict Qwen (oldest), but Qwen
    # is needed in 2 steps. Budget is just-tight enough to require ONE
    # eviction before the 4th step.
    seq = [
        _qwen(),    # 28
        _heartm(),  # 14   — total resident 42
        _ltx(),     # 48   — would push to 90; if budget=60, evict something
        _qwen(),    # next reference to qwen
    ]
    # Budget 80: qwen(28)+heartm(14)=42, plus ltx(48) → 90 > 80, must evict
    # to fit ltx. Qwen+ltx (76) fits in 80; heartm+ltx (62) also fits. The
    # candidates are qwen (next-use=3) and heartmula (next-use=∞). MIN picks
    # heartmula (furthest future) and keeps qwen resident.
    plan = plan_resident_set(seq, budget_gb=80)
    assert plan[2].evict == ["heartmula"]
    # Step 3 (qwen again) is a HIT — proof MIN preserved it correctly.
    assert plan[3].load == []
    assert "qwen" in plan[3].keep


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------
def test_build_sequence_skips_none_and_slopped():
    seq = build_sequence_for_job(
        base_model="qwen",
        video_model="ltx-2.3",
        audio_model="none",
        tts_model="slopped:foo.wav",
        upscale_model="",
    )
    assert [s.model for s in seq] == ["qwen", "ltx-2.3"]
    # Footprints come from the budget table.
    assert seq[0].gb == 28
    assert seq[1].gb == 48


def test_naive_vs_planned_load_count_savings():
    # 5-video synthetic benchmark. All videos share qwen + ltx + heartmula
    # + qwen-tts. Naive cold-loads every stage. Planner keeps the recurring
    # models resident.
    one_video = [_qwen(), _ltx(), _heartm(), _qtts()]
    seq = []
    for _ in range(5):
        seq.extend(one_video)
    plan = plan_resident_set(seq, budget_gb=112)

    # Sum of one video's models = 28+48+14+10 = 100 <= 112, so once all
    # four are loaded they stay resident forever and every subsequent step
    # is a hit.
    assert naive_load_count(seq) == 20
    assert planned_load_count(plan) == 4    # one cold-load per distinct model
    # Savings = 16 cold-loads avoided across 5 videos.
    assert naive_load_count(seq) - planned_load_count(plan) == 16


if __name__ == "__main__":
    # Allow `python tests/test_memory_planner.py` for ad-hoc smoke runs.
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
