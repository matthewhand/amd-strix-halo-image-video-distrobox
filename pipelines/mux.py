"""Audio muxing helpers for chain-joined mp4s.

Thin ffmpeg wrappers that run inside the comfyui-ltx23 container via
`comfy_container.exec_ffmpeg()`. All paths are translated from host
OUTPUT_DIR to the container's /opt/ComfyUI/output mount.
"""
import os

from . import comfy_container, config


def _rel(p):
    return p.replace(config.OUTPUT_DIR, "/opt/ComfyUI/output")


def replace_audio(video_path: str, audio_path: str, out_path: str) -> bool:
    """Swap the video's audio track with `audio_path`, looping if shorter."""
    rc = comfy_container.exec_ffmpeg([
        "-i", _rel(video_path),
        "-stream_loop", "-1", "-i", _rel(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-shortest",
        _rel(out_path),
    ]).returncode
    return rc == 0 and os.path.exists(out_path)


def mix_audio(
    video_path: str,
    audio_path: str,
    out_path: str,
    video_audio_gain: float = 0.3,
    music_gain: float = 0.8,
) -> bool:
    """Mix the video's own audio (ducked) under `audio_path` as music bed."""
    flt = (
        f"[0:a]volume={video_audio_gain}[a1];"
        f"[1:a]volume={music_gain}[a2];"
        f"[a1][a2]amix=inputs=2:duration=longest[aout]"
    )
    rc = comfy_container.exec_ffmpeg([
        "-i", _rel(video_path),
        "-stream_loop", "-1", "-i", _rel(audio_path),
        "-filter_complex", flt,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-shortest",
        _rel(out_path),
    ]).returncode
    return rc == 0 and os.path.exists(out_path)


def add_fadeout(audio_path: str, out_path: str, fade_duration_s: float = 2.0) -> bool:
    """Apply an afade tail fade to the audio file."""
    probe = comfy_container.exec_ffmpeg([
        "-i", _rel(audio_path),
        "-f", "null", "-",
    ])
    dur = None
    for line in probe.stderr.splitlines():
        if "Duration:" in line:
            t = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = t.split(":")
            dur = int(h) * 3600 + int(m) * 60 + float(s)
            break
    start = max(0.0, (dur or fade_duration_s) - fade_duration_s)
    rc = comfy_container.exec_ffmpeg([
        "-i", _rel(audio_path),
        "-af", f"afade=t=out:st={start}:d={fade_duration_s}",
        _rel(out_path),
    ]).returncode
    return rc == 0 and os.path.exists(out_path)
