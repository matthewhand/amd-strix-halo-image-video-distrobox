"""Tests for slopfinity.workers.merge::MergeWorker (ffmpeg final mux)."""
from __future__ import annotations

import asyncio
import os
import shutil
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from slopfinity.workers import merge as merge_mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"merge worker not importable: {exc}", allow_module_level=True)


def _mk_item(video, audio=None, tts=None, post_asset=None, v_idx=1, out_dir=None):
    return {
        "v_idx": v_idx,
        "config_snapshot": {"out_dir": out_dir or "/tmp", "music_gain_db": -8},
        "stages": {
            "video": {"status": "done", "asset": video},
            "post": {"status": "done", "asset": post_asset},
            "audio": {"status": "done", "asset": audio},
            "tts": {"status": "done", "asset": tts},
            "ffmpeg": {"status": "pending"},
        },
    }


def test_merge_worker_role():
    w = merge_mod.MergeWorker()
    assert w.role == "ffmpeg"


def test_merge_worker_video_only_copies(tmp_path, monkeypatch):
    video = tmp_path / "v1.mp4"
    video.write_bytes(b"MP4")
    monkeypatch.setattr(merge_mod, "_ffmpeg_available", lambda: True)

    item = _mk_item(str(video), out_dir=str(tmp_path), v_idx=1)
    w = merge_mod.MergeWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["mode"] == "video-only"
    assert os.path.exists(result["asset"])


def test_merge_worker_tts_only_uses_ffmpeg_mux(tmp_path, monkeypatch):
    video = tmp_path / "v2.mp4"
    video.write_bytes(b"MP4")
    tts = tmp_path / "v2.wav"
    tts.write_bytes(b"WAV")
    monkeypatch.setattr(merge_mod, "_ffmpeg_available", lambda: True)

    called = {}

    def fake_mux(v, a, o, **kw):
        called["args"] = (v, a, o, kw)
        with open(o, "wb") as f:
            f.write(b"OUT")
        return True

    monkeypatch.setattr(merge_mod.ffmpeg_mux, "mux", fake_mux)

    item = _mk_item(str(video), tts=str(tts), out_dir=str(tmp_path), v_idx=2)
    w = merge_mod.MergeWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["mode"] == "tts-only"
    assert called["args"][0] == str(video)
    assert called["args"][1] == str(tts)


def test_merge_worker_music_plus_voice(tmp_path, monkeypatch):
    video = tmp_path / "v3.mp4"; video.write_bytes(b"MP4")
    music = tmp_path / "v3_m.wav"; music.write_bytes(b"MUS")
    voice = tmp_path / "v3_v.wav"; voice.write_bytes(b"VOC")
    monkeypatch.setattr(merge_mod, "_ffmpeg_available", lambda: True)

    async def fake_run(cmd):
        out_path = cmd[-1]
        with open(out_path, "wb") as f:
            f.write(b"FINAL")
        return 0

    monkeypatch.setattr(merge_mod, "_run", fake_run)

    item = _mk_item(str(video), audio=str(music), tts=str(voice),
                    out_dir=str(tmp_path), v_idx=3)
    w = merge_mod.MergeWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["mode"] == "music+voice"
    assert result["asset"].endswith("FINAL_3.mp4")


def test_merge_worker_prefers_post_asset(tmp_path, monkeypatch):
    raw = tmp_path / "raw.mp4"; raw.write_bytes(b"RAW")
    upscaled = tmp_path / "up.mp4"; upscaled.write_bytes(b"UP")
    monkeypatch.setattr(merge_mod, "_ffmpeg_available", lambda: True)

    item = _mk_item(str(raw), post_asset=str(upscaled),
                    out_dir=str(tmp_path), v_idx=4)
    w = merge_mod.MergeWorker()
    chosen = w._resolve_video_in(item)
    assert chosen == str(upscaled)


def test_merge_worker_missing_video(tmp_path):
    item = _mk_item("/nonexistent/path.mp4", out_dir=str(tmp_path))
    w = merge_mod.MergeWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is False
