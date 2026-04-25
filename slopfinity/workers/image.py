"""ImageWorker — base PNG generation via Qwen-Image (or configured base model).

Phase 3 of the queueing refactor. Mirrors the docker run shape used by the
existing fleet runner / `slopfinity.workers.run_image_qwen`, but operates as
a `StageWorker(role="image")` so the Phase 4 coordinator can dispatch queue
items to it.

Tier → step mapping (default low):
    low     →  8 steps (default)
    medium  → 16 steps
    high    → 28 steps

Output asset path is written to `item.stages.image.asset`.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any, Dict, List, Optional

from ._compat import StageWorker, config_snapshot_get, item_v_idx, stage_get

try:  # acquire_gpu lives in the existing scheduler; should always be present
    from ..scheduler import acquire_gpu
except Exception:  # pragma: no cover — defensive
    acquire_gpu = None  # type: ignore[assignment]


IMAGE = "amd-strix-halo-image-video-toolbox:latest"

TIER_STEPS = {
    "low": 8,
    "medium": 16,
    "high": 28,
    "auto": 8,
}


def _hf_cache() -> str:
    return os.path.expanduser("~/.cache/huggingface")


def _workspace() -> str:
    return os.environ.get("SLOPFINITY_WORKSPACE") or os.getcwd()


def _docker_cmd(prompt: str, steps: int, out_path: str) -> List[str]:
    """Build the `docker run --rm ... qwen_launcher.py generate` argv."""
    return [
        "docker", "run", "--rm",
        "-e", "PYTHONPATH=/opt/qwen-image-studio/src",
        "-v", f"{_workspace()}:/workspace",
        "-v", f"{_hf_cache()}:/root/.cache/huggingface",
        "-w", "/workspace",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        IMAGE,
        "python3", "/opt/qwen_launcher.py", "generate",
        "--prompt", prompt,
        "--steps", str(steps),
        "--out", out_path,
    ]


async def _run(cmd: List[str]) -> int:
    def _do() -> int:
        return subprocess.run(cmd, check=False).returncode
    return await asyncio.to_thread(_do)


class ImageWorker(StageWorker):
    """Stage worker for the `image` role — base PNG via Qwen-Image launcher."""

    role = "image"

    def __init__(self, role: str = "image", model: str = "qwen") -> None:
        super().__init__(role=role)
        self.model = model

    def _resolve_steps(self, item: Any) -> int:
        tier = config_snapshot_get(item, "tier", "low") or "low"
        return TIER_STEPS.get(str(tier).lower(), 8)

    def _resolve_prompt(self, item: Any) -> str:
        # Prefer image-stage prompt_override, else fall back to concept output.
        prompt = stage_get(item, "image", "prompt_override")
        if not prompt:
            prompt = stage_get(item, "concept", "output")
        if not prompt:
            prompt = config_snapshot_get(item, "prompt", "") or ""
        return str(prompt or "")

    def _resolve_out_path(self, item: Any) -> str:
        v_idx = item_v_idx(item)
        out_dir = config_snapshot_get(item, "out_dir", _workspace()) or _workspace()
        return os.path.join(str(out_dir), f"v{v_idx}_base.png")

    async def run_stage(self, item: Any) -> Dict[str, Any]:
        prompt = self._resolve_prompt(item)
        steps = self._resolve_steps(item)
        out_path = self._resolve_out_path(item)

        if not prompt:
            return {"ok": False, "output": None, "asset": None,
                    "error": "image worker: empty prompt (concept stage missing?)"}

        cmd = _docker_cmd(prompt, steps, out_path)

        if acquire_gpu is None:
            rc = await _run(cmd)
        else:
            async with acquire_gpu("Base Image", self.model):
                rc = await _run(cmd)

        ok = rc == 0 and os.path.exists(out_path)
        return {
            "ok": ok,
            "output": out_path if ok else None,
            "asset": out_path if ok else None,
            "rc": rc,
            "steps": steps,
            "model": self.model,
        }
