"""MergeWorker — pure ffmpeg job combining post + audio + tts → FINAL_v{N}.mp4.

No GPU; doesn't call `acquire_gpu`. Uses the existing
`slopfinity.workers.ffmpeg_mux` helper for the audio mux step. When both
audio (music) and tts (voiceover) assets are present, music is mixed
under the voiceover via a second ffmpeg pass.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from ._compat import StageWorker, config_snapshot_get, item_v_idx, stage_get

try:
    from . import ffmpeg_mux
except Exception:  # pragma: no cover
    ffmpeg_mux = None  # type: ignore[assignment]


def _workspace() -> str:
    return os.environ.get("SLOPFINITY_WORKSPACE") or os.getcwd()


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def _run(cmd: List[str]) -> int:
    def _do() -> int:
        return subprocess.run(cmd, capture_output=True).returncode
    return await asyncio.to_thread(_do)


def _mix_two_audio_cmd(
    video_path: str,
    music_path: str,
    voice_path: str,
    out_path: str,
    music_gain_db: int = -8,
) -> List[str]:
    """ffmpeg argv mixing music (ducked) + voice over a video track."""
    return [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", music_path,
        "-i", voice_path,
        "-filter_complex",
        f"[1:a]volume={music_gain_db}dB[m];[m][2:a]amix=inputs=2:duration=first:dropout_transition=0[a]",
        "-map", "0:v:0",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]


class MergeWorker(StageWorker):
    """Stage worker for the `ffmpeg` role — final mux into FINAL_v{N}.mp4."""

    role = "ffmpeg"

    def __init__(self, role: str = "ffmpeg") -> None:
        super().__init__(role=role)

    def _resolve_video_in(self, item: Any) -> str:
        # Prefer the upscaled output; fall back to the raw video stage asset.
        post_asset = stage_get(item, "post", "asset")
        if post_asset:
            return str(post_asset)
        return str(stage_get(item, "video", "asset", "") or "")

    def _resolve_audio(self, item: Any) -> Optional[str]:
        return stage_get(item, "audio", "asset")

    def _resolve_tts(self, item: Any) -> Optional[str]:
        return stage_get(item, "tts", "asset")

    def _resolve_out_path(self, item: Any) -> str:
        v_idx = item_v_idx(item)
        out_dir = config_snapshot_get(item, "out_dir", _workspace()) or _workspace()
        return os.path.join(str(out_dir), f"FINAL_{v_idx}.mp4")

    async def run_stage(self, item: Any) -> Dict[str, Any]:
        video = self._resolve_video_in(item)
        audio = self._resolve_audio(item)
        tts = self._resolve_tts(item)
        out_path = self._resolve_out_path(item)
        music_gain = int(config_snapshot_get(item, "music_gain_db", -8) or -8)

        if not video or not os.path.exists(video):
            return {"ok": False, "output": None, "asset": None,
                    "error": f"merge worker: video not found ({video!r})"}
        if not _ffmpeg_available():
            return {"ok": False, "output": None, "asset": None,
                    "error": "merge worker: ffmpeg not in PATH"}

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Branch on which audio sources are present.
        if audio and tts and os.path.exists(audio) and os.path.exists(tts):
            cmd = _mix_two_audio_cmd(video, audio, tts, out_path,
                                     music_gain_db=music_gain)
            rc = await _run(cmd)
            ok = rc == 0 and os.path.exists(out_path)
            return {"ok": ok, "output": out_path if ok else None,
                    "asset": out_path if ok else None, "rc": rc,
                    "mode": "music+voice"}

        single = tts if (tts and os.path.exists(tts)) else (
            audio if (audio and os.path.exists(audio)) else None)
        if single is None:
            # No audio at all — just copy the video to the FINAL filename.
            try:
                shutil.copyfile(video, out_path)
                return {"ok": True, "output": out_path, "asset": out_path,
                        "mode": "video-only"}
            except OSError as exc:
                return {"ok": False, "output": None, "asset": None,
                        "error": str(exc)}

        if ffmpeg_mux is None:
            return {"ok": False, "output": None, "asset": None,
                    "error": "merge worker: ffmpeg_mux module unavailable"}

        def _mux() -> bool:
            return ffmpeg_mux.mux(video, single, out_path,
                                  loop_audio=False, pad_to_video=True)
        ok = await asyncio.to_thread(_mux)
        return {
            "ok": bool(ok),
            "output": out_path if ok else None,
            "asset": out_path if ok else None,
            "mode": "tts-only" if single == tts else "music-only",
        }
