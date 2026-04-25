"""Tests for slopfinity.workers.tts::TTSWorker."""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from slopfinity.workers import tts as tts_mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"tts worker not importable: {exc}", allow_module_level=True)


def _mk_item(text="hello world", voice="ryan", v_idx=1, out_dir=None,
             override=None):
    return {
        "v_idx": v_idx,
        "config_snapshot": {"tts_voice": voice, "out_dir": out_dir or "/tmp"},
        "stages": {
            "concept": {"output": text, "status": "done"},
            "tts": {"status": "pending", "prompt_override": override},
        },
    }


def test_tts_worker_role():
    w = tts_mod.TTSWorker()
    assert w.role == "tts"


def test_tts_worker_uses_override_when_present():
    w = tts_mod.TTSWorker()
    item = _mk_item(text="from concept", override="from override")
    assert w._resolve_text(item) == "from override"


def test_tts_worker_falls_back_to_concept():
    w = tts_mod.TTSWorker()
    item = _mk_item(text="from concept")
    assert w._resolve_text(item) == "from concept"


def test_tts_worker_run_stage_success(tmp_path, monkeypatch):
    item = _mk_item(out_dir=str(tmp_path), v_idx=9)
    fake_wav = b"RIFF\x00\x00\x00\x00WAVEdata"

    async def fake_post(self, payload):  # noqa: ARG001
        return fake_wav

    monkeypatch.setattr(tts_mod.TTSWorker, "_post", fake_post)

    w = tts_mod.TTSWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["asset"].endswith("v9_tts.wav")
    with open(result["asset"], "rb") as f:
        assert f.read() == fake_wav


def test_tts_worker_handles_request_failure(tmp_path, monkeypatch):
    item = _mk_item(out_dir=str(tmp_path))

    async def fake_post(self, payload):  # noqa: ARG001
        return None

    monkeypatch.setattr(tts_mod.TTSWorker, "_post", fake_post)

    w = tts_mod.TTSWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is False
    assert "tts worker" in (result.get("error") or "")
