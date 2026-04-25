"""Tests for slopfinity.workers.audio::AudioWorker (stub for Phase 3)."""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from slopfinity.workers import audio as audio_mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"audio worker not importable: {exc}", allow_module_level=True)


def test_audio_worker_role_and_can_claim():
    w = audio_mod.AudioWorker()
    assert w.role == "audio"
    item = {"stages": {"audio": {"status": "pending"}}}
    assert w.can_claim(item) is True


def test_audio_worker_stub_returns_skipped():
    w = audio_mod.AudioWorker()
    item = {"stages": {"audio": {"status": "pending"}}}
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["asset"] is None
    assert "skipped" in result
    assert "Phase 4" in result["skipped"]
