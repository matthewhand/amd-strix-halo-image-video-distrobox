"""VideoWorker — i2v / t2v video generation (LTX-2.3 via ComfyUI, Wan via docker).

LTX uses slopfinity.ltx_comfy (ComfyUI HTTP). Ensure comfyui via service_registry
before generate. Wan remains docker run --rm until CLI is fixed.
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


def _is_ltx(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("ltx") or m in ("ltx-2.3", "ltx2.3", "ltx")


def _launcher_for(model: str) -> str:
    if model.startswith("wan"):
        return "/opt/wan_launcher.py"
    return "/opt/ltx_launcher.py"


def _docker_cmd(model: str, prompt: str, in_img: str, out_path: str) -> List[str]:
    """Docker path (wan with correct generate.py flags, or ltx launcher)."""
    ws = _workspace()
    c_out = out_path
    if out_path.startswith(ws):
        c_out = "/workspace" + out_path[len(ws):]
    c_in = in_img
    if in_img and in_img.startswith(ws):
        c_in = "/workspace" + in_img[len(ws):]
    base = [
        "docker", "run", "--rm",
        "-v", f"{ws}:/workspace",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface",
        "-w", "/workspace",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        "-e", "WAN_ATTENTION_BACKEND=sdpa",
    ]
    if model.startswith("wan"):
        from ..wan_cli import wan_launcher_argv, wan_paths
        cfg = wan_paths(model)
        ckpt, lora = cfg["ckpt"], cfg["lora"]
        base += [
            "-v", f"{ckpt}:/models/{os.path.basename(ckpt.rstrip('/'))}:ro",
        ]
        if lora and os.path.isdir(lora):
            base += [
                "-v", f"{lora}:/models/lightning/{os.path.basename(lora.rstrip('/'))}:ro",
            ]
        base += [IMAGE] + wan_launcher_argv(prompt, in_img, out_path, model)
        return base
    base += [
        IMAGE,
        "python3", _launcher_for(model),
        "--mode", "video",
        "--prompt", prompt,
        "--image", c_in,
        "--out", c_out,
    ]
    return base


async def _run(cmd: List[str]) -> int:
    def _do() -> int:
        return subprocess.run(cmd, check=False).returncode
    return await asyncio.to_thread(_do)


class VideoWorker(StageWorker):
    """Stage worker for the `video` role."""

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

    async def _run_ltx_comfy(self, prompt: str, in_img: str, out_path: str) -> int:
        if ltx_comfy is None:
            return 2
        if _svc is not None:
            ens = await asyncio.to_thread(_svc.ensure_for_stage, "video", "ltx-2.3")
            if not ens.get("ok") and not ens.get("skipped"):
                return 3
        frames = int(os.environ.get("SLOPFINITY_LTX_FRAMES", "49"))
        return await asyncio.to_thread(
            ltx_comfy.generate_video,
            prompt,
            out_path,
            image_path=in_img or "",
            frames=frames,
        )

    async def run_stage(self, item: Any) -> Dict[str, Any]:
        model = self._resolve_model(item)
        prompt = self._resolve_prompt(item)
        in_img = self._resolve_input_image(item)
        out_path = self._resolve_out_path(item)

        if not prompt:
            return {"ok": False, "output": None, "asset": None,
                    "error": "video worker: empty prompt"}

        mode = (os.environ.get("SLOPFINITY_VIDEO_MODE") or "http").strip().lower()
        use_comfy = _is_ltx(model) and mode != "docker" and ltx_comfy is not None

        async def _body() -> int:
            if use_comfy:
                return await self._run_ltx_comfy(prompt, in_img, out_path)
            cmd = _docker_cmd(model, prompt, in_img, out_path)
            return await _run(cmd)

        if acquire_gpu is None:
            rc = await _body()
        else:
            async with acquire_gpu("video", model):
                rc = await _body()

        ok = rc == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
        return {
            "ok": ok,
            "output": out_path if ok else None,
            "asset": out_path if ok else None,
            "rc": rc,
            "model": model,
            "backend": "comfy" if use_comfy else "docker",
            "error": None if ok else f"video failed rc={rc}",
        }
