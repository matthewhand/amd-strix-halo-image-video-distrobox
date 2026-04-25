"""PostWorker — ltx-spatial upscaler invoked via docker run on a video MP4.

Reads `item.stages.video.asset` as input and writes the upscaled MP4 path
to `item.stages.post.asset`. Acquires the GPU via `acquire_gpu("upscale",
"ltx-spatial")` so it serializes against image/video stages.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any, Dict, List

from ._compat import StageWorker, config_snapshot_get, item_v_idx, stage_get

try:
    from ..scheduler import acquire_gpu
except Exception:  # pragma: no cover
    acquire_gpu = None  # type: ignore[assignment]


IMAGE = "amd-strix-halo-image-video-toolbox:latest"


def _hf_cache() -> str:
    return os.path.expanduser("~/.cache/huggingface")


def _workspace() -> str:
    return os.environ.get("SLOPFINITY_WORKSPACE") or os.getcwd()


def _docker_cmd(in_path: str, out_path: str) -> List[str]:
    return [
        "docker", "run", "--rm",
        "-v", f"{_workspace()}:/workspace",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface",
        "-w", "/workspace",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        IMAGE,
        "python3", "/opt/ltx_launcher.py",
        "--mode", "upscale",
        "--input", in_path,
        "--out", out_path,
    ]


async def _run(cmd: List[str]) -> int:
    def _do() -> int:
        return subprocess.run(cmd, check=False).returncode
    return await asyncio.to_thread(_do)


class PostWorker(StageWorker):
    """Stage worker for the `post` role — ltx-spatial upscale of the video."""

    role = "post"

    def __init__(self, role: str = "post", model: str = "ltx-spatial") -> None:
        super().__init__(role=role)
        self.model = model

    def _resolve_input(self, item: Any) -> str:
        return str(stage_get(item, "video", "asset", "") or "")

    def _resolve_out_path(self, item: Any) -> str:
        v_idx = item_v_idx(item)
        out_dir = config_snapshot_get(item, "out_dir", _workspace()) or _workspace()
        return os.path.join(str(out_dir), f"v{v_idx}_upscaled.mp4")

    async def run_stage(self, item: Any) -> Dict[str, Any]:
        in_path = self._resolve_input(item)
        out_path = self._resolve_out_path(item)

        if not in_path:
            return {"ok": False, "output": None, "asset": None,
                    "error": "post worker: no input video (stages.video.asset missing)"}

        cmd = _docker_cmd(in_path, out_path)

        if acquire_gpu is None:
            rc = await _run(cmd)
        else:
            async with acquire_gpu("upscale", self.model):
                rc = await _run(cmd)

        ok = rc == 0 and os.path.exists(out_path)
        return {
            "ok": ok,
            "output": out_path if ok else None,
            "asset": out_path if ok else None,
            "rc": rc,
            "model": self.model,
        }
