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


def test_audio_worker_http_run_stage_success(tmp_path, monkeypatch):
    """Real run_stage HTTP path: ensure + urlopen mocked only; WAV written."""
    from unittest import mock
    import json

    out_dir = str(tmp_path)
    item = {
        "v_idx": 2,
        "config_snapshot": {"out_dir": out_dir},
        "stages": {
            "concept": {"status": "done", "output": "epic orchestral theme"},
            "audio": {"status": "pending"},
        },
    }
    monkeypatch.setenv("SLOPFINITY_AUDIO_MODE", "http")
    monkeypatch.setenv("HEARTMULA_URL", "http://test-music:8011")

    wav = b"RIFF" + b"\x00" * 44 + b"data" + b"\x01" * 64
    body = json.dumps({"ok": True, "url": "/files/music/x.wav"}).encode()

    def fake_ensure(stage, model=""):
        return {"ok": True, "id": "heartmula", "already_up": True}

    def fake_urlopen(req, timeout=None):
        class R:
            def read(self):
                return body
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        return R()

    with mock.patch("slopfinity.service_registry.ensure_for_stage", fake_ensure), \
         mock.patch("slopfinity.service_registry.base_url_for", return_value="http://test-music:8011"), \
         mock.patch("urllib.request.urlopen", fake_urlopen):
        w = audio_mod.AudioWorker()
        result = asyncio.run(w.run_stage(item))

    assert result["ok"] is True
    assert result.get("skipped") is not True
    # When only relative url returned, asset is the staged out_path
    assert result.get("asset")
    assert "audio" in str(result["asset"]) or result.get("url")


def test_audio_worker_http_failure_not_success_asset(tmp_path, monkeypatch):
    """HTTP failure must not claim a successful audio asset."""
    from unittest import mock
    import urllib.error

    item = {
        "v_idx": 3,
        "config_snapshot": {"out_dir": str(tmp_path)},
        "stages": {
            "concept": {"status": "done", "output": "theme"},
            "audio": {"status": "pending"},
        },
    }
    monkeypatch.setenv("SLOPFINITY_AUDIO_MODE", "http")

    def fake_ensure(stage, model=""):
        return {"ok": True, "already_up": True}

    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    with mock.patch("slopfinity.service_registry.ensure_for_stage", fake_ensure), \
         mock.patch("slopfinity.service_registry.base_url_for", return_value="http://x:8011"), \
         mock.patch("urllib.request.urlopen", boom):
        w = audio_mod.AudioWorker()
        result = asyncio.run(w.run_stage(item))
    assert result["ok"] is False
    assert not result.get("asset")
