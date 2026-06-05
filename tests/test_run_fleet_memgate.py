"""Pre-stage memory gate (run_fleet._wait_for_free_memory) — wait for enough
unified memory to free before a heavy stage, toggle-gated."""
import run_fleet as rf
from slopfinity import config as cfg


def test_mem_gate_satisfied_accounts_for_safety():
    assert rf._mem_gate_satisfied(40, 20, 8) is True
    assert rf._mem_gate_satisfied(27, 20, 8) is False   # 27 < 20+8


def test_model_gb_known_and_default():
    assert rf._model_gb("qwen") == 20.0
    assert rf._model_gb("does-not-exist", 38) == 38.0


def test_toggle_off_is_noop(monkeypatch):
    cfg.save_config({"scheduler": {}})                  # gate disabled
    monkeypatch.setattr(rf, "_available_gb", lambda: 1.0)   # would block if gated
    monkeypatch.setattr("time.sleep", lambda s: None)
    rf._wait_for_free_memory(20, "x")                   # returns immediately


def test_returns_immediately_when_enough(monkeypatch):
    cfg.save_config({"scheduler": {"wait_for_free_memory": True}})
    monkeypatch.setattr(rf, "_available_gb", lambda: 999.0)
    rf._wait_for_free_memory(20, "x")                   # plenty → no wait


def test_waits_then_proceeds_when_memory_frees(monkeypatch):
    cfg.save_config({"scheduler": {"wait_for_free_memory": True}})
    calls = {"n": 0}

    def avail():
        calls["n"] += 1
        return 999.0 if calls["n"] >= 3 else 1.0        # frees up on the 3rd poll

    monkeypatch.setattr(rf, "_available_gb", avail)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(rf, "update_state", lambda **k: None)
    rf._wait_for_free_memory(20, "x")
    assert calls["n"] >= 3                              # it actually waited


def test_times_out_then_proceeds(monkeypatch):
    cfg.save_config({"scheduler": {"wait_for_free_memory": True,
                                   "wait_for_free_memory_timeout_s": 0.05}})
    monkeypatch.setattr(rf, "_available_gb", lambda: 1.0)   # never enough
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(rf, "update_state", lambda **k: None)
    rf._wait_for_free_memory(20, "x")                   # must return (not wedge)
