#!/usr/bin/env python3
"""HOMIE subject-to-video launcher for Strix Halo (ROCm-oriented).

Wraps upstream YIYANGCAI/HOMIE generate.py single-sample mode:

  python homie_launcher.py \\
      --prompt "A man tips his hat" \\
      --ref-image /path/to/subject.png \\
      --out /workspace/comfy-outputs/homie_proof.mp4

Weights (env or flags):
  HOMIE_CKPT_DIR   — Wan2.1-T2V-14B Diffusers (vae/text_encoder/tokenizer)
  HOMIE_WEIGHTS    — trained Homie_Wan_14B*.safetensors dir
  HOMIE_MLLM_CKPT  — optional Qwen3-VL-2B-Thinking for MLLM features

Upstream is CUDA-first; we force SDPA-friendly env and optional flash-attn shim.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


DEFAULT_SIZE = "832*480"
DEFAULT_FRAMES = 49
DEFAULT_STEPS = 50
DEFAULT_CFG = 5.0
DEFAULT_FPS = 16


def _install_flash_attn_shim() -> None:
    """Stub flash_attn so transformers probes succeed on ROCm (no CUDA FA2)."""
    if os.environ.get("HOMIE_FA_SHIM", "1") != "1":
        return
    import importlib.machinery

    for mod_name, is_pkg in (
        ("flash_attn", True),
        ("flash_attn.cute", False),
        ("flash_attn_2_cuda", False),
    ):
        if mod_name in sys.modules:
            continue
        try:
            __import__(mod_name)
            continue
        except Exception:
            pass
        m = types.ModuleType(mod_name)
        m.__spec__ = importlib.machinery.ModuleSpec(
            mod_name, loader=None, is_package=is_pkg
        )
        m.__file__ = f"<homie_fa_shim:{mod_name}>"
        if is_pkg:
            m.__path__ = []  # type: ignore[attr-defined]
        for name in ("fwd", "bwd", "varlen_fwd", "varlen_bwd", "flash_attn_varlen_func"):
            setattr(m, name, MagicMock(name=f"{mod_name}.{name}"))
        sys.modules[mod_name] = m
    # package attribute link
    if "flash_attn" in sys.modules and "flash_attn.cute" in sys.modules:
        sys.modules["flash_attn"].cute = sys.modules["flash_attn.cute"]  # type: ignore[attr-defined]


def _ensure_homie_on_path() -> Path:
    candidates = [
        os.environ.get("HOMIE_ROOT"),
        "/opt/HOMIE",
        str(Path(__file__).resolve().parents[1] / "HOMIE"),
        str(Path("/workspace/HOMIE")),
    ]
    for c in candidates:
        if not c:
            continue
        root = Path(c)
        if (root / "generate.py").is_file() and (root / "homie_wan").is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise SystemExit(
        "HOMIE source not found. Set HOMIE_ROOT or clone "
        "https://github.com/YIYANGCAI/HOMIE into ./HOMIE"
    )


def parse_args():
    p = argparse.ArgumentParser(description="HOMIE subject-to-video (ROCm wrapper)")
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--ref-image",
        required=True,
        help="Comma-separated reference image paths (subjects a,b,c…)",
    )
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument(
        "--ckpt-dir",
        default=os.environ.get(
            "HOMIE_CKPT_DIR",
            os.environ.get("WAN_CKPT_DIR", "./weights/Wan2.1-T2V-14B-Diffusers"),
        ),
    )
    p.add_argument(
        "--homie-ckpt",
        default=os.environ.get("HOMIE_WEIGHTS", "./weights/HOMIE-Wan-Model"),
    )
    p.add_argument("--size", default=os.environ.get("HOMIE_SIZE", DEFAULT_SIZE))
    p.add_argument(
        "--frame-num",
        type=int,
        default=int(os.environ.get("HOMIE_FRAMES", DEFAULT_FRAMES)),
    )
    p.add_argument(
        "--sample-steps",
        type=int,
        default=int(os.environ.get("HOMIE_STEPS", DEFAULT_STEPS)),
    )
    p.add_argument(
        "--sample-guide-scale",
        type=float,
        default=float(os.environ.get("HOMIE_CFG", DEFAULT_CFG)),
    )
    p.add_argument(
        "--sample-fps",
        type=int,
        default=int(os.environ.get("HOMIE_FPS", DEFAULT_FPS)),
    )
    p.add_argument("--seed", type=int, default=int(os.environ.get("HOMIE_SEED", "42")))
    p.add_argument("--qwen-feature", default=None, help="Optional precomputed .pt")
    p.add_argument(
        "--task",
        default=os.environ.get("HOMIE_TASK", "s2v-14B"),
    )
    p.add_argument(
        "--offload-model",
        action="store_true",
        default=os.environ.get("HOMIE_OFFLOAD", "1") == "1",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _install_flash_attn_shim()
    root = _ensure_homie_on_path()
    os.chdir(root)

    # Prefer SDPA / avoid CUDA-only flash kernels on ROCm
    os.environ.setdefault("WAN_ATTENTION_BACKEND", "sdpa")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:1024")

    from generate import generate  # type: ignore  # noqa: E402
    from generate import _parse_args as _upstream_parse  # type: ignore  # noqa: E402

    # Build argv for upstream argparse
    argv = [
        "generate.py",
        "--task",
        args.task,
        "--size",
        args.size,
        "--frame_num",
        str(args.frame_num),
        "--sample_fps",
        str(args.sample_fps),
        "--sample_steps",
        str(args.sample_steps),
        "--sample_guide_scale",
        str(args.sample_guide_scale),
        "--ckpt_dir",
        args.ckpt_dir,
        "--homie_ckpt",
        args.homie_ckpt,
        "--prompt",
        args.prompt,
        "--ref_image",
        args.ref_image,
        "--save_file",
        args.out,
        "--save_path",
        str(Path(args.out).parent or "."),
        "--base_seed",
        str(args.seed),
    ]
    if args.qwen_feature:
        argv += ["--qwen_feature", args.qwen_feature]
    if args.offload_model:
        argv += ["--offload_model", "true"]

    old = sys.argv
    try:
        sys.argv = argv
        uargs = _upstream_parse()
        print(
            f"[HOMIE] generate size={uargs.size} frames={uargs.frame_num} "
            f"steps={uargs.sample_steps} out={args.out}",
            flush=True,
        )
        generate(uargs)
    finally:
        sys.argv = old

    if not os.path.isfile(args.out) or os.path.getsize(args.out) == 0:
        print(f"[HOMIE] ERROR: missing output {args.out}", file=sys.stderr)
        return 1
    print(f"[HOMIE] ok {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
