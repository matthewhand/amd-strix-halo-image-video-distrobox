"""AudioWorker — Heartmula music generation (Phase 4 wired).

Calls run_audio_heartmula() which shells out to the heartmula_launcher
inside the amd-strix-halo-image-video-toolbox Docker container. If that
container / script isn't present, the call returns a non-zero exit code
and the queue item is marked failed (not silently skipped).

Output WAV path lands in `item.stages.audio.asset`.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from ._compat import StageWorker, stage_get, item_v_idx

logger = logging.getLogger(__name__)


class AudioWorker(StageWorker):
    """Stage worker for the `audio` role — Heartmula music."""

    role = "audio"

    def __init__(self, role: str = "audio") -> None:
        super().__init__(role=role)

    async def run_stage(self, item: Any, stage: str = "audio") -> Dict[str, Any]:
        prompt = stage_get(item, "concept", "output") or ""
        if not prompt:
            # No concept/music prompt available. Intentionally skip music
            # generation so the queue advances, but make the skip explicit
            # (logged + flagged) so the pipeline/UI can distinguish an
            # intentional skip from a successful run that produced an asset.
            v_idx = item_v_idx(item)
            reason = "no music prompt"
            logger.info("AudioWorker: skipping music for v%s — %s", v_idx, reason)
            return {
                "ok": True,
                "output": None,
                # `asset` stays None so downstream merge.py treats this as
                # "no audio" rather than a produced asset.
                "asset": None,
                "skipped": True,
                "reason": reason,
            }

        v_idx = item_v_idx(item)
        out_dir = (
            (item.get("config_snapshot") or {}).get("out_dir")
            or os.environ.get("SLOPFINITY_OUT_DIR", "/tmp")
        )
        out_path = os.path.join(out_dir, f"v{v_idx}_audio.wav")

        mode = (os.environ.get("SLOPFINITY_AUDIO_MODE") or "http").lower()
        if mode != "docker":
            try:
                from .. import service_registry as _svc
                import asyncio
                import json
                import urllib.request

                ens = await asyncio.to_thread(_svc.ensure_for_stage, "audio", "heartmula")
                if not ens.get("ok") and not ens.get("skipped"):
                    return {"ok": False, "error": f"heartmula ensure failed: {ens}"}

                base = _svc.base_url_for("heartmula") or os.environ.get("HEARTMULA_URL", "http://127.0.0.1:8011")
                base = base.rstrip("/")
                url = base if base.endswith("/music") else f"{base}/music"
                payload = json.dumps({"prompt": prompt, "duration": 30}).encode("utf-8")
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )

                def _post():
                    with urllib.request.urlopen(req, timeout=600) as r:
                        return r.read()

                raw = await asyncio.to_thread(_post)
                # JSON envelope {ok, url} or raw wav
                try:
                    data = json.loads(raw.decode("utf-8"))
                    if not data.get("ok", True):
                        return {"ok": False, "error": data.get("error") or str(data)}
                    # Worker may write under shared workspace; prefer returned path/url.
                    # If only relative url, leave asset to dashboard files route.
                    asset = data.get("audio_path") or data.get("path") or out_path
                    if data.get("url") and not os.path.exists(str(asset)):
                        return {"ok": True, "asset": out_path, "url": data.get("url"), "http": True}
                    return {"ok": True, "asset": asset, "http": True}
                except Exception:
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(raw)
                    return {"ok": os.path.getsize(out_path) > 0, "asset": out_path, "http": True}
            except Exception as exc:
                if mode == "http":
                    return {"ok": False, "error": f"heartmula http error: {exc}"}
                # fall through to docker mode on hybrid
                pass

        # Lazy import (only when we actually have a prompt to generate from)
        # avoids a circular import and keeps the skip path above dependency-free.
        from slopfinity.workers import run_audio_heartmula

        try:
            rc = await run_audio_heartmula(prompt, out_path)
        except Exception as exc:
            return {"ok": False, "error": f"heartmula launch error: {exc}"}

        if rc == 0:
            return {"ok": True, "asset": out_path}
        return {"ok": False, "error": f"heartmula exited with code {rc}"}
