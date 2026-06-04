"""Coverage for round-5 fixes via the pure helpers extracted for testability:
stage-prompt resolution, cancel.flag mtime-gating, and the queue `extra`
catch-all's defensive merge."""
import os

import run_fleet
from slopfinity import config as cfg


# --- stage_prompts application (run_fleet._resolve_stage_prompt) -------------

def test_stage_prompt_override_used_when_present():
    sp = {"image": "  a portrait  ", "video": "", "music": "drums"}
    assert run_fleet._resolve_stage_prompt(sp, "image", "MAIN") == "a portrait"
    assert run_fleet._resolve_stage_prompt(sp, "music", "MAIN") == "drums"


def test_stage_prompt_falls_back_when_blank_or_missing():
    sp = {"image": "   ", "video": ""}
    assert run_fleet._resolve_stage_prompt(sp, "image", "MAIN") == "MAIN"   # whitespace-only
    assert run_fleet._resolve_stage_prompt(sp, "video", "MAIN") == "MAIN"   # empty
    assert run_fleet._resolve_stage_prompt(sp, "tts", "MAIN") == "MAIN"     # missing key
    assert run_fleet._resolve_stage_prompt(None, "image", "MAIN") == "MAIN"  # no overrides


# --- cancel.flag mtime-gating (run_fleet._cancel_requested) ------------------

def test_cancel_requested_fresh_flag_aborts(tmp_path):
    flag = tmp_path / "cancel.flag"
    flag.write_text("now")
    iter_started = os.path.getmtime(flag) - 1.0  # iter started before the flag
    assert run_fleet._cancel_requested(str(flag), iter_started) is True


def test_cancel_requested_stale_flag_ignored(tmp_path):
    flag = tmp_path / "cancel.flag"
    flag.write_text("old")
    iter_started = os.path.getmtime(flag) + 5.0   # iter started AFTER the flag → stale
    assert run_fleet._cancel_requested(str(flag), iter_started) is False


def test_cancel_requested_no_flag(tmp_path):
    assert run_fleet._cancel_requested(str(tmp_path / "nope.flag"), 0.0) is False


# --- queue `extra` catch-all defensive merge (config._split_queue_item) ------

def test_split_queue_item_funnels_unknown_fields():
    out = cfg._split_queue_item({"prompt": "p", "ts": 1.0, "seed_image": "s.png"})
    assert out["prompt"] == "p"
    assert out["extra"] == {"seed_image": "s.png"}
    assert "seed_image" not in {k for k in out if k != "extra"}
    assert out.get("id")  # stable id stamped


def test_split_queue_item_merges_preexisting_extra():
    # An item that already carries an `extra` dict (defensive path) must merge
    # it with freshly-detected unknowns rather than dropping either.
    out = cfg._split_queue_item({"prompt": "p", "extra": {"a": 1}, "b": 2})
    assert out["extra"] == {"a": 1, "b": 2}
    assert "b" not in {k for k in out if k != "extra"}
