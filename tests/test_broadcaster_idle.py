"""Idle ComfyUI VRAM-reclaim decision (broadcaster._idle_free_due)."""
from slopfinity.broadcaster import _idle_free_due

IDLE_MIN = 10
SEC = IDLE_MIN * 60


def test_active_never_frees():
    assert _idle_free_due(active=True, last_active=0.0, now=99999.0,
                          idle_min=IDLE_MIN, already_freed=False) is False


def test_idle_below_threshold_waits():
    # idle for 9 min < 10 min threshold
    assert _idle_free_due(False, last_active=0.0, now=9 * 60,
                          idle_min=IDLE_MIN, already_freed=False) is False


def test_idle_at_threshold_frees():
    assert _idle_free_due(False, last_active=0.0, now=SEC,
                          idle_min=IDLE_MIN, already_freed=False) is True
    assert _idle_free_due(False, last_active=0.0, now=SEC + 120,
                          idle_min=IDLE_MIN, already_freed=False) is True


def test_latch_prevents_refree():
    # already freed this idle period → don't free again until activity resets it
    assert _idle_free_due(False, last_active=0.0, now=SEC * 5,
                          idle_min=IDLE_MIN, already_freed=True) is False


def test_disabled_when_zero():
    assert _idle_free_due(False, last_active=0.0, now=SEC * 99,
                          idle_min=0, already_freed=False) is False


def test_no_baseline_yet():
    assert _idle_free_due(False, last_active=None, now=SEC * 99,
                          idle_min=IDLE_MIN, already_freed=False) is False
