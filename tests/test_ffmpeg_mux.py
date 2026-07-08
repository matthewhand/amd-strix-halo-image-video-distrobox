"""Real-path tests for slopfinity.workers.ffmpeg_mux — produce a non-empty muxed file.

Uses the shipped `ffmpeg_mux.mux` entrypoint with tiny fixtures generated via
ffmpeg (not a re-implemented mux command in the assertion path).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity.workers import ffmpeg_mux  # noqa: E402


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)


def _make_tiny_video(path: str, seconds: float = 0.5) -> None:
    """Generate a minimal H.264 MP4 via system ffmpeg (fixture only)."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s=64x64:d={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(seconds),
        path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _make_tiny_wav(path: str, seconds: float = 0.5) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-c:a", "pcm_s16le",
        path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_ffmpeg_mux_build_cmd_shape():
    cmd = ffmpeg_mux.build_cmd("/v.mp4", "/a.wav", "/o.mp4")
    assert cmd[0] == "ffmpeg"
    assert "/v.mp4" in cmd and "/a.wav" in cmd and "/o.mp4" in cmd
    assert "-c:v" in cmd and "copy" in cmd


def test_ffmpeg_mux_real_output_file(tmp_path):
    """Shipped mux() writes a non-empty MP4 from real tiny fixtures."""
    video = str(tmp_path / "in.mp4")
    audio = str(tmp_path / "in.wav")
    out = str(tmp_path / "muxed.mp4")
    _make_tiny_video(video)
    _make_tiny_wav(audio)
    assert os.path.getsize(video) > 0
    assert os.path.getsize(audio) > 0

    ok = ffmpeg_mux.mux(video, audio, out, pad_to_video=True)
    assert ok is True
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    # Record path for harness capture consumers
    print(f"MUX_OUT={out} size={os.path.getsize(out)}")


def test_merge_worker_real_mux_run_stage(tmp_path, monkeypatch):
    """MergeWorker.run_stage with real ffmpeg_mux.mux (no fake mux)."""
    from slopfinity.workers import merge as merge_mod

    video = str(tmp_path / "v.mp4")
    tts = str(tmp_path / "t.wav")
    _make_tiny_video(video)
    _make_tiny_wav(tts)

    monkeypatch.setattr(merge_mod, "_ffmpeg_available", lambda: True)
    # Do NOT mock ffmpeg_mux.mux — use the real helper.

    item = {
        "v_idx": 9,
        "config_snapshot": {"out_dir": str(tmp_path), "music_gain_db": -8},
        "stages": {
            "video": {"status": "done", "asset": video},
            "post": {"status": "done", "asset": None},
            "audio": {"status": "done", "asset": None},
            "tts": {"status": "done", "asset": tts},
            "ffmpeg": {"status": "pending"},
        },
    }
    w = merge_mod.MergeWorker()
    import asyncio
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True, result
    assert result.get("asset")
    assert os.path.exists(result["asset"])
    assert os.path.getsize(result["asset"]) > 0
    print(f"MERGE_OUT={result['asset']} size={os.path.getsize(result['asset'])}")


def test_stage_order_covers_all_asset_kinds():
    """Schema still defines the full multi-kind pipeline."""
    from slopfinity.queue_schema import STAGE_ORDER, PREREQS
    assert STAGE_ORDER == ["concept", "image", "video", "audio", "tts", "post", "merge"]
    assert "concept" in STAGE_ORDER[0]
    assert PREREQS["merge"] == ["post", "audio", "tts"]
    assert PREREQS["image"] == ["concept"]
    assert PREREQS["video"] == ["image"]
