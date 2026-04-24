"""ffmpeg audio/video mux helper.

Primary use case: take a silent Wan/LTX video and a Qwen-TTS WAV and produce
a muxed MP4. Duration mismatch handling is explicit — either pad the audio
with silence up to video length (default) or loop it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import List


def build_cmd(
    video_path: str,
    audio_path: str,
    out_path: str,
    *,
    loop_audio: bool = False,
    pad_to_video: bool = True,
) -> List[str]:
    """Build the ffmpeg argv. Separated for unit testing."""
    cmd: List[str] = ["ffmpeg", "-y"]

    # Input 0: video.
    cmd += ["-i", video_path]

    # Input 1: audio (optionally looped at the demuxer).
    if loop_audio:
        cmd += ["-stream_loop", "-1", "-i", audio_path]
    else:
        cmd += ["-i", audio_path]

    # Padding with silence: apad on the audio filter chain + -shortest.
    if pad_to_video and not loop_audio:
        cmd += [
            "-filter_complex", "[1:a]apad[a]",
            "-map", "0:v:0",
            "-map", "[a]",
        ]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]

    cmd += [
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    return cmd


def mux(
    video_path: str,
    audio_path: str,
    out_path: str,
    *,
    loop_audio: bool = False,
    pad_to_video: bool = True,
) -> bool:
    """Mux audio onto video. Returns True on success.

    - `loop_audio=True`: loop the audio track until video ends.
    - `pad_to_video=True` (default): pad audio with silence to video length.
    - Both False: classic `-shortest` (whichever stream ends first wins).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = build_cmd(
        video_path,
        audio_path,
        out_path,
        loop_audio=loop_audio,
        pad_to_video=pad_to_video,
    )
    proc = subprocess.run(cmd, capture_output=True)
    return proc.returncode == 0 and os.path.exists(out_path)
