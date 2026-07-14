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
import json
import os
import subprocess
import urllib.error
import urllib.request
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


# Hard cap (seconds) for docker GPU image call. On timeout return 124
# so acquire_gpu releases the lock instead of hanging forever.
IMAGE_TIMEOUT_S = 600


async def _run(cmd: List[str], timeout: int = IMAGE_TIMEOUT_S) -> int:
    def _do() -> int:
        try:
            return subprocess.run(cmd, check=False, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"⏱  image worker timeout after {timeout}s", flush=True)
            return 124  # conventional timeout exit code → treated as failure
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

        mode = (os.environ.get("SLOPFINITY_IMAGE_MODE") or "http").lower()
        if mode != "docker":
            try:
                from .. import service_registry as _svc
                ens = await asyncio.to_thread(_svc.ensure_for_stage, "image", self.model or "qwen")
                if not ens.get("ok") and not ens.get("skipped"):
                    return {"ok": False, "output": None, "asset": None,
                            "error": f"image ensure failed: {ens}", "steps": steps, "model": self.model}

                base = _svc.base_url_for("qwen-image") or os.environ.get(
                    "IMAGE_API_URL", "http://127.0.0.1:8180"
                )
                base = base.rstrip("/")
                # Qwen Image Studio style: POST /api/generate (best-effort contract)
                url = base if base.endswith("/api/generate") else f"{base}/api/generate"
                payload = json.dumps({
                    "prompt": prompt,
                    "steps": steps,
                    "num_images": 1,
                }).encode("utf-8")

                def _post():
                    req = urllib.request.Request(
                        url, data=payload,
                        headers={"Content-Type": "application/json"}, method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=600) as r:
                        return r.read(), getattr(r, "status", 200)

                if acquire_gpu is None:
                    raw, status = await asyncio.to_thread(_post)
                else:
                    async with acquire_gpu("image", self.model):
                        raw, status = await asyncio.to_thread(_post)

                # If API returns JSON job envelope, surface it; if raw image, write out_path.
                try:
                    data = json.loads(raw.decode("utf-8"))
                    return {
                        "ok": True,
                        "output": data.get("path") or data.get("url") or out_path,
                        "asset": data.get("path") or out_path,
                        "http": True,
                        "job": data,
                        "steps": steps,
                        "model": self.model,
                    }
                except Exception:
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(raw)
                    ok = os.path.getsize(out_path) > 0
                    return {
                        "ok": ok,
                        "output": out_path if ok else None,
                        "asset": out_path if ok else None,
                        "http": True,
                        "steps": steps,
                        "model": self.model,
                    }
            except Exception as exc:
                if mode == "http":
                    return {"ok": False, "output": None, "asset": None,
                            "error": f"image http error: {exc}", "steps": steps, "model": self.model}

        cmd = _docker_cmd(prompt, steps, out_path)

        if acquire_gpu is None:
            rc = await _run(cmd)
        else:
            async with acquire_gpu("image", self.model):
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
