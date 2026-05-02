"""AudioWorker — Heartmula music generation (Phase 4 wired).

Calls run_audio_heartmula() which shells out to the heartmula_launcher
inside the amd-strix-halo-image-video-toolbox Docker container. If that
container / script isn't present, the call returns a non-zero exit code
and the queue item is marked failed (not silently skipped).

Output WAV path lands in `item.stages.audio.asset`.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from ._compat import StageWorker, stage_get, item_v_idx


class AudioWorker(StageWorker):
    """Stage worker for the `audio` role — Heartmula music."""

    role = "audio"

    def __init__(self, role: str = "audio") -> None:
        super().__init__(role=role)

    async def run_stage(self, item: Any) -> Dict[str, Any]:
        from slopfinity.workers import run_audio_heartmula  # lazy import avoids circular

        prompt = stage_get(item, "concept", "output") or ""
        if not prompt:
            # No concept output — fall through silently so the queue advances.
            return {
                "ok": True,
                "output": None,
                "asset": None,
                "skipped": "audio skipped — no concept prompt available",
            }

        v_idx = item_v_idx(item)
        out_dir = (
            (item.get("config_snapshot") or {}).get("out_dir")
            or os.environ.get("SLOPFINITY_OUT_DIR", "/tmp")
        )
        out_path = os.path.join(out_dir, f"v{v_idx}_audio.wav")

        try:
            rc = await run_audio_heartmula(prompt, out_path)
        except Exception as exc:
            return {"ok": False, "error": f"heartmula launch error: {exc}"}

        if rc == 0:
            return {"ok": True, "asset": out_path}
        return {"ok": False, "error": f"heartmula exited with code {rc}"}
