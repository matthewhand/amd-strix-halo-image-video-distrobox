"""AudioWorker — Heartmula music generation (stub for Phase 3).

Heartmula isn't easily callable from the dashboard process today (the
launcher lives in the toolbox container and is invoked by the fleet
runner). Per the Phase 3 spec we ship this as a no-op that lets the
queue advance; Phase 4's coordinator will wire in the real call.

Output WAV path will land in `item.stages.audio.asset` once wired.
"""
from __future__ import annotations

from typing import Any, Dict

from ._compat import StageWorker


class AudioWorker(StageWorker):
    """Stage worker for the `audio` role — Heartmula music. STUB."""

    role = "audio"

    def __init__(self, role: str = "audio") -> None:
        super().__init__(role=role)

    async def run_stage(self, item: Any) -> Dict[str, Any]:  # noqa: ARG002
        # TODO(phase-4): replace with real Heartmula docker run via
        #   `slopfinity.workers.run_audio_heartmula(prompt, out)` once the
        #   coordinator passes through prompt/out paths from queue items.
        return {
            "ok": True,
            "output": None,
            "asset": None,
            "skipped": "audio worker stub — Phase 4 wires in HeartMuLa",
        }
