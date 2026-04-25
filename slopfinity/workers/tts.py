"""TTSWorker — POST to the local Qwen3-TTS worker on :8010 → WAV asset.

Reads voice from `item.config_snapshot["tts_voice"]` (fallback: 'ryan').
Reads text from `item.stages.tts.prompt_override` else
`item.stages.concept.output`.
Writes the saved WAV path to `item.stages.tts.asset`.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ._compat import StageWorker, config_snapshot_get, item_v_idx, stage_get


TTS_URL = os.environ.get("TTS_WORKER_URL", "http://localhost:8010/tts")


def _workspace() -> str:
    return os.environ.get("SLOPFINITY_WORKSPACE") or os.getcwd()


class TTSWorker(StageWorker):
    """Stage worker for the `tts` role — HTTP POST to local Qwen3-TTS."""

    role = "tts"

    def __init__(self, role: str = "tts", url: str = TTS_URL) -> None:
        super().__init__(role=role)
        self.url = url

    def _resolve_text(self, item: Any) -> str:
        text = stage_get(item, "tts", "prompt_override")
        if not text:
            text = stage_get(item, "concept", "output")
        return str(text or "")

    def _resolve_voice(self, item: Any) -> str:
        return str(config_snapshot_get(item, "tts_voice", "ryan") or "ryan")

    def _resolve_out_path(self, item: Any) -> str:
        v_idx = item_v_idx(item)
        out_dir = config_snapshot_get(item, "out_dir", _workspace()) or _workspace()
        return os.path.join(str(out_dir), f"v{v_idx}_tts.wav")

    async def _post(self, payload: dict) -> Optional[bytes]:
        def _do() -> Optional[bytes]:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                return None
        return await asyncio.to_thread(_do)

    async def run_stage(self, item: Any) -> Dict[str, Any]:
        text = self._resolve_text(item)
        voice = self._resolve_voice(item)
        out_path = self._resolve_out_path(item)

        if not text:
            return {"ok": False, "output": None, "asset": None,
                    "error": "tts worker: empty text"}

        body = await self._post({"text": text, "voice": voice})
        if body is None:
            return {"ok": False, "output": None, "asset": None,
                    "error": f"tts worker: request to {self.url} failed"}

        # The TTS worker returns either raw audio bytes or a JSON envelope
        # with a server-side path. Keep the simple raw-bytes contract here.
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(body)
            ok = os.path.getsize(out_path) > 0
        except OSError as exc:
            return {"ok": False, "output": None, "asset": None, "error": str(exc)}

        return {
            "ok": ok,
            "output": out_path if ok else None,
            "asset": out_path if ok else None,
            "voice": voice,
        }
