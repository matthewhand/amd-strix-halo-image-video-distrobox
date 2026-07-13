#!/usr/bin/env python3
"""LTX-2.3 launcher — ComfyUI HTTP backend (image / video / upscale).

Shipped as /opt/ltx_launcher.py in the toolbox image and used by
slopfinity.workers.* docker paths. Prefer calling slopfinity.ltx_comfy
directly from host workers (no docker bounce for Comfy).

Modes:
  image   — text → single-frame PNG via LTX latent length=1
  video   — optional seed image + prompt → MP4 (i2v / t2v)
  upscale — seed frame from input MP4 → LTX spatial upsampler → MP4
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow import of slopfinity when launched from repo or /opt
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="LTX-2.3 ComfyUI launcher")
    p.add_argument("--mode", choices=("image", "video", "upscale"), required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--image", default="", help="Seed image (video mode)")
    p.add_argument("--input", default="", help="Input video (upscale mode)")
    p.add_argument("--out", required=True, help="Output path")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--timeout", type=float, default=1800)
    args = p.parse_args(argv)

    try:
        from slopfinity import ltx_comfy as ltx
    except ImportError:
        # Inside toolbox container: mount or copy slopfinity may be absent —
        # fall back to loading module by path if present on /workspace.
        import importlib.util
        cand = [
            os.path.join(_ROOT, "slopfinity", "ltx_comfy.py"),
            "/workspace/slopfinity/ltx_comfy.py",
        ]
        mod = None
        for path in cand:
            if os.path.isfile(path):
                spec = importlib.util.spec_from_file_location("ltx_comfy", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                break
        if mod is None:
            print("ERROR: cannot import slopfinity.ltx_comfy", file=sys.stderr)
            return 2
        ltx = mod

    if args.mode == "image":
        if not args.prompt:
            print("ERROR: --prompt required for image mode", file=sys.stderr)
            return 2
        return ltx.generate_image(
            args.prompt, args.out,
            width=args.width, height=args.height, timeout_s=args.timeout,
        )
    if args.mode == "video":
        if not args.prompt:
            print("ERROR: --prompt required for video mode", file=sys.stderr)
            return 2
        return ltx.generate_video(
            args.prompt, args.out,
            image_path=args.image or "",
            width=args.width, height=args.height,
            frames=args.frames, timeout_s=args.timeout,
        )
    # upscale
    inp = args.input or args.image
    if not inp:
        print("ERROR: --input (or --image) required for upscale mode", file=sys.stderr)
        return 2
    return ltx.upscale_video(
        inp, args.out,
        prompt=args.prompt or "cinematic continuity, high detail, sharp",
        frames=min(args.frames, 49),
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
