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


def test_audio_worker_role():
    w = audio_mod.AudioWorker()
    assert w.role == "audio"


def test_audio_worker_skips_with_no_prompt():
    """With no concept/music prompt, the worker returns an explicit,
    clearly-marked skip (not a produced asset)."""
    w = audio_mod.AudioWorker()
    item = {"stages": {"audio": {"status": "pending"}}}
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    # No asset must be produced when skipping, so downstream merge does
    # not mistake the skip for real audio.
    assert result["asset"] is None
    # Explicit skip marker + human-readable reason.
    assert result["skipped"] is True
    assert result["reason"] == "no music prompt"
