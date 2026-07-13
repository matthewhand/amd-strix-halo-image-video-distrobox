"""Real ffmpeg mux path used by POST /mux and fleet merge."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slopfinity.workers.ffmpeg_mux import build_cmd, mux  # noqa: E402


def _have_ffmpeg() -> bool:
    return subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_build_cmd_includes_copy_and_aac():
    cmd = build_cmd("/v.mp4", "/a.wav", "/o.mp4", pad_to_video=True)
    assert cmd[0] == "ffmpeg"
    assert "-c:v" in cmd and "copy" in cmd
    assert "-c:a" in cmd and "aac" in cmd
    assert "/o.mp4" in cmd


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_mux_real_tiny_streams(tmp_path: Path):
    """Generate minimal video+audio with ffmpeg, then mux via shipped helper."""
    video = tmp_path / "v.mp4"
    audio = tmp_path / "a.wav"
    out = tmp_path / "out.mp4"
    # 0.5s color video + 0.3s sine wav
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=f=440:d=0.3",
            str(audio),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert video.is_file() and audio.is_file()
    ok = mux(str(video), str(audio), str(out), pad_to_video=True)
    assert ok is True
    assert out.is_file()
    assert out.stat().st_size > 500
    # ffprobe: has video and audio streams
    probe = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", str(out),
        ],
        text=True,
    )
    types = {ln.strip() for ln in probe.splitlines() if ln.strip()}
    assert "video" in types
    assert "audio" in types
