#!/usr/bin/env python3
"""
ERNIE-Image launcher for Strix Halo (ROCm).

Usage:
  python ernie_launcher.py --prompt "a cat reading a book" \
      --model baidu/ERNIE-Image-Turbo --steps 8 --out /tmp/out.png

Defaults target the Turbo variant (8 steps). Pass --model baidu/ERNIE-Image
with --steps 50 for the SFT variant.
"""

import argparse
import os
import sys
import types
from unittest.mock import MagicMock


def install_flash_attn_shim():
    """Stub flash_attn_2_cuda for ROCm — same shim used by qwen_launcher."""
    if os.environ.get("ERNIE_FA_SHIM", "1") != "1":
        return
    try:
        import flash_attn_2_cuda  # noqa: F401
    except ImportError:
        print("[ERNIE] Injecting Flash Attention shim for ROCm.", file=sys.stderr)
        fa = types.ModuleType("flash_attn_2_cuda")
        for name in ("fwd", "bwd", "varlen_fwd", "varlen_bwd"):
            setattr(fa, name, MagicMock())
        sys.modules["flash_attn_2_cuda"] = fa


def parse_args():
    p = argparse.ArgumentParser(description="ERNIE-Image generator")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--model", default="baidu/ERNIE-Image-Turbo",
                   help="HF repo id (Turbo or SFT)")
    p.add_argument("--steps", type=int, default=8,
                   help="8 for Turbo, 50 for SFT")
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-prompt-enhancer", action="store_true",
                   help="Disable the built-in prompt enhancer")
    p.add_argument("--out", default="ernie_output.png")
    return p.parse_args()


def main():
    args = parse_args()
    install_flash_attn_shim()

    import torch
    from diffusers import ErnieImagePipeline

    if not torch.cuda.is_available():
        print("❌ No GPU detected (torch.cuda.is_available() == False).",
              file=sys.stderr)
        sys.exit(1)

    print(f"[ERNIE] Loading {args.model} (bf16) ...", file=sys.stderr)
    pipe = ErnieImagePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"[ERNIE] Generating {args.width}x{args.height}, "
          f"{args.steps} steps, guidance={args.guidance}", file=sys.stderr)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        use_pe=not args.no_prompt_enhancer,
        generator=generator,
    ).images[0]

    image.save(args.out)
    print(f"✅ Saved {args.out}")


if __name__ == "__main__":
    main()
