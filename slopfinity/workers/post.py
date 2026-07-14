"""PostWorker — ltx-spatial upscale via ComfyUI (LTXVLatentUpsampler).

Reads `item.stages.video.asset`, ensures comfyui, runs seed-frame + LTX
spatial upscale graph (slopfinity.ltx_comfy.upscale_video). Docker fallback
when SLOPFINITY_UPSCALE_MODE=docker.
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

try:
    from .. import service_registry as _svc
except Exception:  # pragma: no cover
    _svc = None  # type: ignore[assignment]

try:
    from .. import ltx_comfy
except Exception:  # pragma: no cover
    ltx_comfy = None  # type: ignore[assignment]


IMAGE = "amd-strix-halo-image-video-toolbox:latest"


def _hf_cache() -> str:
    return os.path.expanduser("~/.cache/huggingface")


def _workspace() -> str:
    return os.environ.get("SLOPFINITY_WORKSPACE") or os.getcwd()


def _docker_cmd(in_path: str, out_path: str) -> List[str]:
    ws = _workspace()
    c_in, c_out = in_path, out_path
    if in_path.startswith(ws):
        c_in = "/workspace" + in_path[len(ws):]
    if out_path.startswith(ws):
        c_out = "/workspace" + out_path[len(ws):]
    return [
        "docker", "run", "--rm",
        "-v", f"{ws}:/workspace",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface",
        "-w", "/workspace",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        IMAGE,
        "python3", "/opt/ltx_launcher.py",
        "--mode", "upscale",
        "--input", c_in,
        "--out", c_out,
    ]


# Hard cap (seconds) for docker GPU post call. On timeout return 124
# so acquire_gpu releases the lock instead of hanging forever.
POST_TIMEOUT_S = 600


async def _run(cmd: List[str], timeout: int = POST_TIMEOUT_S) -> int:
    def _do() -> int:
        try:
            return subprocess.run(cmd, check=False, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"⏱  post worker timeout after {timeout}s", flush=True)
            return 124  # conventional timeout exit code → treated as failure
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

    def _resolve_prompt(self, item: Any) -> str:
        p = stage_get(item, "video", "prompt_override") or stage_get(item, "concept", "output")
        return str(p or "cinematic continuity, high detail, sharp")

    async def _run_comfy(self, in_path: str, out_path: str, prompt: str) -> int:
        if ltx_comfy is None:
            return 2
        if _svc is not None:
            ens = await asyncio.to_thread(_svc.ensure_for_stage, "upscale", "ltx-spatial")
            if not ens.get("ok") and not ens.get("skipped"):
                return 3
        frames = int(os.environ.get("SLOPFINITY_LTX_UPSCALE_FRAMES", "25"))
        return await asyncio.to_thread(
            ltx_comfy.upscale_video,
            in_path,
            out_path,
            prompt=prompt,
            frames=frames,
        )

    async def run_stage(self, item: Any) -> Dict[str, Any]:
        in_path = self._resolve_input(item)
        out_path = self._resolve_out_path(item)
        prompt = self._resolve_prompt(item)

        if not in_path:
            return {"ok": False, "output": None, "asset": None,
                    "error": "post worker: no input video (stages.video.asset missing)"}
        if not os.path.isfile(in_path):
            return {"ok": False, "output": None, "asset": None,
                    "error": f"post worker: input missing: {in_path}"}

        mode = (os.environ.get("SLOPFINITY_UPSCALE_MODE") or "http").strip().lower()
        use_comfy = mode != "docker" and ltx_comfy is not None

        async def _body() -> int:
            if use_comfy:
                return await self._run_comfy(in_path, out_path, prompt)
            return await _run(_docker_cmd(in_path, out_path))

        if acquire_gpu is None:
            rc = await _body()
        else:
            async with acquire_gpu("upscale", self.model):
                rc = await _body()

        ok = rc == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
        return {
            "ok": ok,
            "output": out_path if ok else None,
            "asset": out_path if ok else None,
            "rc": rc,
            "model": self.model,
            "backend": "comfy" if use_comfy else "docker",
            "error": None if ok else f"upscale failed rc={rc}",
        }
