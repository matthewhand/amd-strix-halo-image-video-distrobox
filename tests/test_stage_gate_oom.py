"""TDD: prove the loading plan cannot OOM the host.

Hard invariants:

1. Belady plan never keeps a resident set larger than budget (no OOM-by-plan).
2. stage_gate refuses to start if MemAvailable < need + SAFETY after reclaim.
3. When short on headroom, reclaim runs BEFORE any ensure_up.
4. Exclusive peers are parked before a heavy stage starts.
5. On stage exit without keep, the service is parked (ensure_down).
6. Stacked realistic fleet sequence (qwen + heartmula + ltx + tts) under a
   tight UMA budget always evicts before exceeding budget.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slopfinity.memory_planner import StageStep, plan_resident_set
from slopfinity import stage_gate as gate


# ---------------------------------------------------------------------------
# Pure plan: never OOM-by-plan
# ---------------------------------------------------------------------------

def _seq_full_job():
    return [
        StageStep("Image", "image", "qwen", 28),
        StageStep("Audio", "audio", "heartmula", 14),
        StageStep("TTS", "tts", "dramabox", 18),
        StageStep("Video", "video", "ltx-2.3", 48),
    ]


def test_plan_resident_never_exceeds_budget():
    """After every step, sum(resident peaks) <= budget (except break path)."""
    seq = _seq_full_job() * 3  # three fleet iters
    budget = 64.0  # tight UMA slice — cannot hold everything
    plan = plan_resident_set(seq, budget_gb=budget)
    for d in plan:
        # resident_after models' gb from original sequence
        total = 0.0
        for m in d.resident_after:
            for s in seq:
                if s.model == m:
                    total += s.gb
                    break
        # Planner may briefly overshoot only if nothing left to evict; for
        # this sequence victims always exist → must fit.
        assert total <= budget + 1e-6, (
            f"OOM-by-plan at {d.step.stage}/{d.step.model}: "
            f"resident={d.resident_after} total={total} budget={budget}"
        )


def test_plan_evicts_before_ltx_when_others_warm():
    """qwen(28)+heartmula(14)+dramabox(18)=60; +ltx(48)=108 > 70 → must evict."""
    seq = _seq_full_job()
    plan = plan_resident_set(seq, budget_gb=70)
    # Last step is LTX — must have evicted something to fit 48
    ltx = plan[-1]
    assert ltx.step.model == "ltx-2.3"
    assert ltx.evict, "must evict peers before loading LTX under 70GB budget"
    assert "ltx-2.3" in ltx.resident_after
    # Cannot still hold all four
    assert len(ltx.resident_after) < 4


def test_plan_never_keeps_qwen_heartmula_ltx_together_under_80gb():
    """Classic OOM stack from the NSFW campaign — plan must refuse keep-all."""
    seq = [
        StageStep("Image", "image", "qwen", 28),
        StageStep("Audio", "audio", "heartmula", 14),
        StageStep("Video", "video", "ltx-2.3", 48),
    ]
    plan = plan_resident_set(seq, budget_gb=80)
    # After LTX load, resident cannot be {qwen, heartmula, ltx} (90 > 80)
    after = set(plan[-1].resident_after)
    if after >= {"qwen", "heartmula", "ltx-2.3"}:
        pytest.fail(f"plan kept all three heavy models: {after}")


# ---------------------------------------------------------------------------
# Hard floor helpers (pure)
# ---------------------------------------------------------------------------

def test_can_start_requires_need_plus_safety():
    """Pre-load: free_after_load = available - need must stay ≥ 10GB."""
    assert gate.can_start(available_gb=50, need_gb=28, safety_gb=10) is True  # 22 left
    assert gate.can_start(available_gb=37, need_gb=28, safety_gb=10) is False  # 9 left
    assert gate.can_start(available_gb=38, need_gb=28, safety_gb=10) is True  # 10 left
    assert gate.can_start(available_gb=0.5, need_gb=28, safety_gb=10) is False


def test_remaining_after_load_and_post_load_floor():
    assert gate.remaining_after_load(50, 28) == 22
    # Post-load only cares about measured free ≥ 10
    assert gate.has_safety_after_load(12, already_loaded=True, safety_gb=10) is True
    assert gate.has_safety_after_load(9, already_loaded=True, safety_gb=10) is False
    # Pre-load estimate
    assert gate.has_safety_after_load(40, need_gb=28, safety_gb=10, already_loaded=False) is True
    assert gate.has_safety_after_load(37, need_gb=28, safety_gb=10, already_loaded=False) is False


def test_need_gb_from_budgets_includes_dramabox():
    assert gate.need_gb("image", "qwen") == 28
    assert gate.need_gb("video", "ltx-2.3") == 48
    assert gate.need_gb("tts", "dramabox") >= 12  # must be budgeted (not 0)
    assert gate.need_gb("tts", "kokoro") == 8


# ---------------------------------------------------------------------------
# stage_gate context — reclaim + refuse (with fakes)
# ---------------------------------------------------------------------------

@dataclass
class FakeMem:
    available: float
    history: List[float] = field(default_factory=list)

    def read(self) -> float:
        self.history.append(self.available)
        return self.available


@dataclass
class FakeRegistry:
    up: set = field(default_factory=set)
    ensure_up_calls: List[str] = field(default_factory=list)
    ensure_down_calls: List[str] = field(default_factory=list)

    def ensure_up(self, sid: str, **k: Any) -> dict:
        self.ensure_up_calls.append(sid)
        self.up.add(sid)
        return {"ok": True, "id": sid}

    def ensure_down(self, sid: str) -> dict:
        self.ensure_down_calls.append(sid)
        self.up.discard(sid)
        return {"ok": True, "id": sid}

    def ensure_down_group(self, group: str) -> dict:
        # pretend uma-heavy members
        members = {"qwen-image", "heartmula", "comfyui"}
        for m in list(self.up):
            if m in members:
                self.ensure_down(m)
        return {"ok": True, "group": group}


def test_stage_gate_refuses_when_still_short_after_reclaim():
    """Belt-and-braces: never ensure_up if reclaim cannot free enough."""
    mem = FakeMem(available=5.0)  # host OOM state
    reg = FakeRegistry(up={"heartmula", "comfyui", "qwen-tts"})

    def reclaim(reason: str) -> dict:
        # reclaim frees a little but not enough for LTX 48+10
        mem.available = 20.0
        reg.ensure_down_group("uma-heavy")
        return {"ok": True, "after_gb": mem.available, "actions": ["reclaim"]}

    with pytest.raises(gate.InsufficientMemoryError) as ei:
        with gate.stage_gate(
            role="video",
            model="ltx-2.3",
            mem_reader=mem.read,
            registry=reg,
            reclaim_fn=reclaim,
            safety_gb=10,
        ):
            pass

    assert "ltx-2.3" in str(ei.value).lower() or "insufficient" in str(ei.value).lower()
    # Must NOT have started comfy/video service
    assert "comfyui" not in reg.ensure_up_calls
    # Must have attempted reclaim
    assert mem.available == 20.0


def test_stage_gate_reclaims_then_starts_when_headroom_ok():
    mem = FakeMem(available=30.0)  # short for 48+10=58
    reg = FakeRegistry(up={"heartmula", "qwen-tts"})

    def reclaim(reason: str) -> dict:
        reg.ensure_down("heartmula")
        reg.ensure_down("qwen-tts")
        mem.available = 90.0  # freed pre-load
        return {"ok": True, "after_gb": 90.0, "actions": ["stop-heartmula", "stop-tts"]}

    # After ensure_up, simulate load consuming 48GB → free becomes 42 (≥10 OK)
    orig_up = reg.ensure_up

    def up_and_consume(sid, **k):
        r = orig_up(sid, **k)
        if sid == "comfyui":
            mem.available = 42.0  # 90 - 48 estimate realized
        return r

    reg.ensure_up = up_and_consume  # type: ignore

    entered = []
    with gate.stage_gate(
        role="video",
        model="ltx-2.3",
        mem_reader=mem.read,
        registry=reg,
        reclaim_fn=reclaim,
        safety_gb=10,
        service_id="comfyui",
    ):
        entered.append(True)
        assert "comfyui" in reg.up
        assert mem.available >= 10  # free after load floor

    assert entered == [True]
    assert "comfyui" in reg.ensure_up_calls
    # After exit without keep, park the service we started
    assert "comfyui" in reg.ensure_down_calls


def test_stage_gate_aborts_if_free_below_10gb_after_load():
    """Post-load measured free must stay ≥ 10GB or we park and fail."""
    mem = FakeMem(available=70.0)  # 70-48=22 ≥ 10 pre-OK
    reg = FakeRegistry()

    orig_up = reg.ensure_up

    def up_eats_all(sid, **k):
        r = orig_up(sid, **k)
        mem.available = 3.0  # collapsed after load
        return r

    reg.ensure_up = up_eats_all  # type: ignore

    with pytest.raises(gate.InsufficientMemoryError) as ei:
        with gate.stage_gate(
            role="video",
            model="ltx-2.3",
            mem_reader=mem.read,
            registry=reg,
            reclaim_fn=lambda r: {"ok": True, "after_gb": 70.0, "actions": []},
            safety_gb=10,
            service_id="comfyui",
        ):
            pass
    assert "after load" in str(ei.value).lower() or "below floor" in str(ei.value).lower()
    assert "comfyui" in reg.ensure_down_calls  # rolled back


def test_stage_gate_parks_peers_before_ensure_up():
    """uma-heavy peers must be down before loading LTX."""
    mem = FakeMem(available=100.0)
    reg = FakeRegistry(up={"heartmula", "qwen-image"})
    order: List[str] = []

    real_up = reg.ensure_up
    real_down = reg.ensure_down

    def track_up(sid, **k):
        order.append(f"up:{sid}")
        return real_up(sid, **k)

    def track_down(sid):
        order.append(f"down:{sid}")
        return real_down(sid)

    reg.ensure_up = track_up  # type: ignore
    reg.ensure_down = track_down  # type: ignore

    def reclaim(reason: str) -> dict:
        # stage_gate should call ensure_down_group / peer park itself
        return {"ok": True, "after_gb": mem.available, "actions": []}

    with gate.stage_gate(
        role="video",
        model="ltx-2.3",
        mem_reader=mem.read,
        registry=reg,
        reclaim_fn=reclaim,
        safety_gb=10,
        service_id="comfyui",
        exclusive_group="uma-heavy",
        exclusive_members=("qwen-image", "heartmula", "comfyui"),
    ):
        pass

    # Peers parked before comfy up
    down_idxs = [i for i, x in enumerate(order) if x.startswith("down:")]
    up_idxs = [i for i, x in enumerate(order) if x == "up:comfyui"]
    assert up_idxs, order
    assert down_idxs, order
    assert min(down_idxs) < min(up_idxs), f"must park peers before start: {order}"


def test_stage_gate_keep_skips_ensure_down_on_exit():
    mem = FakeMem(available=100.0)
    reg = FakeRegistry()
    with gate.stage_gate(
        role="image",
        model="qwen",
        mem_reader=mem.read,
        registry=reg,
        reclaim_fn=lambda r: {"ok": True, "after_gb": 100.0, "actions": []},
        safety_gb=10,
        service_id="qwen-image",
        keep_after=True,
    ):
        pass
    assert "qwen-image" in reg.ensure_up_calls
    assert "qwen-image" not in reg.ensure_down_calls


def test_reclaim_all_stops_warm_services(monkeypatch):
    """Default reclaim_all parks known warm services via registry."""
    reg = FakeRegistry(up={"heartmula", "qwen-tts", "comfyui"})
    mem = FakeMem(available=10.0)

    def after_stop():
        # simulate free after stops
        if not reg.up:
            mem.available = 80.0

    # wrap ensure_down to bump mem when empty
    orig = reg.ensure_down

    def ed(sid):
        r = orig(sid)
        after_stop()
        return r

    reg.ensure_down = ed  # type: ignore

    result = gate.reclaim_all(
        reason="test",
        mem_reader=mem.read,
        registry=reg,
        free_comfy_fn=lambda: {"ok": True},
        kill_launchers_fn=lambda: ["qwen_launcher.py"],
    )
    assert "heartmula" not in reg.up
    assert "comfyui" not in reg.up
    assert result["after_gb"] >= result["before_gb"]
    assert any("heartmula" in a or "comfyui" in a or "kill" in a for a in result["actions"])


def test_stacked_campaign_sequence_never_oom_by_plan():
    """Replay NSFW-like sequence: image-only then full with music+tts+video."""
    # 7 image-only qwen, then full pipeline with heartmula+dramabox+ltx
    seq = []
    for _ in range(7):
        seq.append(StageStep("Image", "image", "qwen", 28))
    seq.extend(_seq_full_job())
    budget = 70.0  # forced tight — like half of UMA after OS/services
    plan = plan_resident_set(seq, budget_gb=budget)
    for d in plan:
        total = sum(
            next(s.gb for s in seq if s.model == m)
            for m in d.resident_after
        )
        assert total <= budget + 1e-6, (
            f"campaign plan OOM at {d.step.model}: {d.resident_after} = {total}GB"
        )
