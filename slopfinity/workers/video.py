"""VideoWorker — i2v / t2v video generation (LTX-2.3, Wan2.2, Wan2.5).

Phase 3 placeholder. The existing fleet runner embeds a JSON ComfyUI
workflow that wires base image + prompt → LTX/Wan; that JSON construction
will move to the Phase 4 coordinator. For now we shell out to a docker
launcher, mirroring the existing `slopfinity.workers.run_video_*` shape.

Output MP4 path is written to `item.stages.video.asset`.
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


def _launcher_for(model: str) -> str:
    if model.startswith("wan"):
        return "/opt/wan_launcher.py"
    return "/opt/ltx_launcher.py"


def _docker_cmd(model: str, prompt: str, in_img: str, out_path: str) -> List[str]:
    base = [
        "docker", "run", "--rm",
        "-v", f"{_workspace()}:/workspace",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface",
        "-w", "/workspace",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        IMAGE,
        "python3", _launcher_for(model),
    ]
    if model.startswith("wan"):
        base += [
            "--prompt", prompt,
            "--image", in_img,
            "--out", out_path,
            "--model", model,
        ]
    else:
        base += [
            "--mode", "video",
            "--prompt", prompt,
            "--image", in_img,
            "--out", out_path,
        ]
    return base


async def _run(cmd: List[str]) -> int:
    def _do() -> int:
        return subprocess.run(cmd, check=False).returncode
    return await asyncio.to_thread(_do)


class VideoWorker(StageWorker):
    """Stage worker for the `video` role.

    NOTE: The runner currently embeds a JSON ComfyUI workflow inside the
    fleet pipeline. Phase 4's coordinator will lift that JSON out and pass
    it here; for now we use the launcher CLI shape (placeholder).
    """

    role = "video"

    def __init__(self, role: str = "video") -> None:
        super().__init__(role=role)

    def _resolve_model(self, item: Any) -> str:
        return str(
            config_snapshot_get(item, "video_model")
            or config_snapshot_get(item, "model")
            or "ltx-2.3"
        )

    def _resolve_prompt(self, item: Any) -> str:
        prompt = stage_get(item, "video", "prompt_override")
        if not prompt:
            prompt = stage_get(item, "concept", "output")
        return str(prompt or "")

    def _resolve_input_image(self, item: Any) -> str:
        return str(stage_get(item, "image", "asset", "") or "")

    def _resolve_out_path(self, item: Any) -> str:
        v_idx = item_v_idx(item)
        out_dir = config_snapshot_get(item, "out_dir", _workspace()) or _workspace()
        return os.path.join(str(out_dir), f"v{v_idx}_video.mp4")

    async def run_stage(self, item: Any) -> Dict[str, Any]:
        model = self._resolve_model(item)
        prompt = self._resolve_prompt(item)
        in_img = self._resolve_input_image(item)
        out_path = self._resolve_out_path(item)

        if not prompt:
            return {"ok": False, "output": None, "asset": None,
                    "error": "video worker: empty prompt"}

        cmd = _docker_cmd(model, prompt, in_img, out_path)

        if acquire_gpu is None:
            rc = await _run(cmd)
        else:
            async with acquire_gpu("video", model):
                rc = await _run(cmd)

        ok = rc == 0 and os.path.exists(out_path)
        return {
            "ok": ok,
            "output": out_path if ok else None,
            "asset": out_path if ok else None,
            "rc": rc,
            "model": model,
        }
