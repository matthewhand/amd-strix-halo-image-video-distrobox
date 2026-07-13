"""Unit tests for DramaBox engine selection in qwen_tts_serve (no GPU)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "scripts" / "qwen_tts_serve.py"


def _load_pick_engine():
    """Load only the pure helpers from qwen_tts_serve without FastAPI side-effects."""
    src = SERVE.read_text()
    # Cut before http_worker_app import (needs fastapi)
    cut = src.split("from http_worker_app")[0]
    ns: dict = {
        "__name__": "qwen_tts_serve_partial",
        "__file__": str(SERVE),
        "os": __import__("os"),
        "sys": sys,
        "time": __import__("time"),
        "uuid": __import__("uuid"),
        "subprocess": __import__("subprocess"),
        "annotations": True,
    }
    # from __future__ already in cut; provide common names
    exec(compile(cut, str(SERVE), "exec"), ns)
    return ns["_pick_engine"], ns["DRAMABOX_VOICES"], ns["QWEN_VOICES"]


def test_pick_engine_explicit_dramabox():
    pick, _, _ = _load_pick_engine()
    assert pick("af_heart", "dramabox") == "dramabox"
    assert pick("narrator-female", "DRAMABOX") == "dramabox"


def test_pick_engine_voice_implies_dramabox():
    pick, dvoices, _ = _load_pick_engine()
    assert "narrator-female" in dvoices
    assert pick("narrator-female", None) == "dramabox"
    assert pick("narrator-male", None) == "dramabox"
    assert pick("kid", None) == "dramabox"


def test_pick_engine_qwen_and_kokoro_still_work():
    pick, _, qvoices = _load_pick_engine()
    assert pick("ryan", None) == "qwen"
    assert pick("af_heart", "kokoro") == "kokoro"
    assert pick("af_heart", None) == "kokoro"  # default when not qwen/dramabox voice


def test_dramabox_launcher_exists():
    assert (ROOT / "scripts" / "dramabox_launcher.py").is_file()
    assert (ROOT / "scripts" / "dramabox_voices" / "narrator-female.wav").is_file()
