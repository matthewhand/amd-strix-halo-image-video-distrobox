"""Unit tests for the AI Magic fan-out module.

Covers:
  - preserve-token auto-detection (quotes + proper nouns)
  - lock respect (missing words re-appended verbatim)
  - JSON parse fallback (prose-wrapped + retry nudge)
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import fanout  # noqa: E402


# ---------- preserve-token detection ----------------------------------------

def test_extract_preserve_tokens_quotes_and_proper_nouns():
    core = 'A shot of Aragorn saying "valar morghulis" near Minas Tirith'
    stages = {"image": "", "video": "The sky darkens", "music": "", "tts": ""}
    toks = fanout.extract_preserve_tokens(core, stages)
    # Quoted phrase preserved verbatim
    assert "valar morghulis" in toks
    # Proper nouns kept
    assert "Aragorn" in toks
    assert "Minas" in toks
    assert "Tirith" in toks
    # Sentence-initial stopword ("The") ignored
    assert "The" not in toks


def test_extract_preserve_tokens_dedups_case_insensitive():
    core = 'Aragorn'
    stages = {"image": "aragorn again", "video": "", "music": "", "tts": ""}
    toks = fanout.extract_preserve_tokens(core, stages)
    lowered = [t.lower() for t in toks]
    assert lowered.count("aragorn") == 1


# ---------- lock respect ----------------------------------------------------

def test_verify_locked_reinjects_missing_words():
    stages = {"image": "A red dragon named Smaug", "video": "", "music": "", "tts": ""}
    returned = {
        "image": "A fearsome winged lizard coils in gold",  # DROPS "Smaug"
        "video": "cam slowly pulls back",
        "music": "orchestral doom",
        "tts": "hello",
    }
    patched, dropped = fanout.verify_locked(returned, stages, ["image"])
    assert "Smaug" in patched["image"]
    assert "Smaug" in dropped
    # Unrelated stages untouched
    assert patched["video"] == "cam slowly pulls back"


def test_verify_locked_ignores_unlocked_stages():
    stages = {"image": "Aragorn rides", "video": "", "music": "", "tts": ""}
    returned = {"image": "Someone rides", "video": "", "music": "", "tts": ""}
    patched, dropped = fanout.verify_locked(returned, stages, [])  # nothing locked
    assert "Aragorn" not in patched["image"]
    assert dropped == []


# ---------- JSON parse fallback ---------------------------------------------

def test_fanout_parses_prose_wrapped_json():
    def fake_llm(sys_p, user_msg):
        return (
            "Sure! Here you go:\n\n"
            '{"image": "a blue sky", "video": "clouds drift",'
            ' "music": "ambient drone", "tts": "look up"}\n\n'
            "Hope that helps."
        )
    r = fanout.fanout(
        core="sky",
        stages={"image": "", "video": "", "music": "", "tts": ""},
        locked=[],
        preserve_tokens=[],
        llm_call=fake_llm,
    )
    assert r["ok"]
    assert r["stages"]["image"] == "a blue sky"
    assert r["stages"]["tts"] == "look up"


def test_fanout_retries_when_unparseable_then_succeeds():
    calls = {"n": 0}

    def flaky_llm(sys_p, user_msg):
        calls["n"] += 1
        if calls["n"] == 1:
            return "I refuse to return JSON, here is a haiku instead."
        return json.dumps({
            "image": "x", "video": "y", "music": "z", "tts": "w",
        })

    r = fanout.fanout(
        core="idea",
        stages={"image": "", "video": "", "music": "", "tts": ""},
        locked=[],
        preserve_tokens=[],
        llm_call=flaky_llm,
        max_retries=2,
    )
    assert calls["n"] >= 2
    assert r["stages"]["image"] == "x"


def test_fanout_falls_back_to_seed_when_all_retries_fail():
    def garbage_llm(sys_p, user_msg):
        return "no json ever"
    seed = {"image": "SEED-IMG", "video": "SEED-VID", "music": "SEED-MUS", "tts": "SEED-TTS"}
    r = fanout.fanout(
        core="idea",
        stages=seed,
        locked=["image"],
        preserve_tokens=[],
        llm_call=garbage_llm,
        max_retries=1,
    )
    # Fell back to seed; lock still honored (seed words present)
    assert "SEED-IMG" in r["stages"]["image"]


def test_fanout_end_to_end_lock_and_preserve():
    def llm(sys_p, user_msg):
        # Mimic an LLM that drops the locked seed word "Smaug"
        return json.dumps({
            "image": "a winged lizard on gold",
            "video": "slow dolly in",
            "music": "dread brass",
            "tts": "beware the beast",
        })
    r = fanout.fanout(
        core='"doom approaches" for Smaug',
        stages={"image": "Smaug the dragon", "video": "", "music": "", "tts": ""},
        locked=["image"],
        preserve_tokens=[],
        llm_call=llm,
    )
    assert "Smaug" in r["stages"]["image"]
    assert r["preserved_ok"] is False
    assert "Smaug" in r["preserved_dropped"]
