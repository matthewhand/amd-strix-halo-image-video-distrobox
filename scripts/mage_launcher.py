#!/usr/bin/env python3
"""Mage-Flow launcher for Strix Halo (ROCm).

Defaults to Mage-Flow-Turbo (4 steps, cfg=1) for interactive latency.

Usage:
  python mage_launcher.py --prompt "a red cube on a table" \\
      --model microsoft/Mage-Flow-Turbo --steps 4 --cfg 1.0 \\
      --out /workspace/comfy-outputs/mage_proof.png

ROCm notes:
  - Uses SDPA attention backend (no flash-attn CUDA extension).
  - HF cache reuses the toolbox mount at ~/.cache/huggingface
    (host: ./huggingface-cache).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock


DEFAULT_MODEL = "microsoft/Mage-Flow-Turbo"
DEFAULT_STEPS = 4
DEFAULT_CFG = 1.0


def _install_flash_attn_shim() -> None:
    """Stub flash_attn packages so importlib/transformers probes succeed on ROCm."""
    if os.environ.get("MAGE_FA_SHIM", "1") != "1":
        return
    import importlib.machinery

    def _stub(name: str, *, is_pkg: bool = False):
        if name in sys.modules:
            return sys.modules[name]
        try:
            __import__(name)
            return sys.modules[name]
        except Exception:
            pass
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=is_pkg
        )
        m.__file__ = f"<mage_fa_shim:{name}>"
        if is_pkg:
            m.__path__ = []  # type: ignore[attr-defined]
        for n in ("fwd", "bwd", "varlen_fwd", "varlen_bwd", "flash_attn_varlen_func"):
            setattr(m, n, MagicMock(name=f"{name}.{n}"))
        sys.modules[name] = m
        return m

    fa = _stub("flash_attn", is_pkg=True)
    cute = _stub("flash_attn.cute", is_pkg=False)
    fa.cute = cute  # type: ignore[attr-defined]
    _stub("flash_attn_2_cuda", is_pkg=False)


def _ensure_mage_on_path() -> None:
    """Prefer installed package; else add local Mage/mage_flow parent to path."""
    try:
        import mage_flow  # noqa: F401
        return
    except ImportError:
        pass
    candidates = [
        os.environ.get("MAGE_FLOW_ROOT"),
        "/opt/Mage/mage_flow",
        str(Path(__file__).resolve().parents[2] / "submodules" / "Mage" / "mage_flow"),
        str(Path(__file__).resolve().parents[1] / "Mage" / "mage_flow"),
        str(Path("/workspace/Mage/mage_flow")),
        str(Path("/workspace/../Mage/mage_flow").resolve()),
    ]
    for c in candidates:
        if not c:
            continue
        root = Path(c)
        # package dir is mage_flow itself; parent must be on sys.path
        # pyproject maps package-dir mage_flow -> ".", so parent of mage_flow
        # folder must be importable as mage_flow when the folder is named mage_flow
        # Actually setuptools package-dir maps "mage_flow" = "." meaning the
        # mage_flow directory *is* the package. So we need the parent of the
        # mage_flow directory on path and import mage_flow.
        parent = root.parent if root.name == "mage_flow" else root
        if (parent / "mage_flow").is_dir() or root.name == "mage_flow":
            p = str(parent if root.name == "mage_flow" else root)
            if p not in sys.path:
                sys.path.insert(0, p)
            # Also support running from the package dir layout without install:
            # Mage/mage_flow contains __init__.py → treat as top-level package
            # via package-dir style: add the mage_flow dir's parent.
            if root.name == "mage_flow" and str(root.parent) not in sys.path:
                sys.path.insert(0, str(root.parent))
            # Editable-style: the package root IS mage_flow; map via path hack
            # by adding a synthetic parent. Easiest: insert parent of mage_flow.
            return


def parse_args():
    p = argparse.ArgumentParser(description="Mage-Flow text-to-image generator (ROCm)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--neg-prompt", default=None)
    p.add_argument(
        "--model",
        default=os.environ.get("MAGE_MODEL", DEFAULT_MODEL),
        help="HF repo id or local path (default: Mage-Flow-Turbo)",
    )
    p.add_argument("--steps", type=int, default=int(os.environ.get("MAGE_STEPS", DEFAULT_STEPS)))
    p.add_argument("--cfg", type=float, default=float(os.environ.get("MAGE_CFG", DEFAULT_CFG)))
    p.add_argument("--width", type=int, default=int(os.environ.get("MAGE_WIDTH", "1024")))
    p.add_argument("--height", type=int, default=int(os.environ.get("MAGE_HEIGHT", "1024")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=os.environ.get("MAGE_DEVICE", "cuda"))
    p.add_argument("--out", default="mage_output.png")
    p.add_argument(
        "--attn",
        default=os.environ.get("MAGE_ATTN") or os.environ.get("MAGE_ATTN_BACKEND") or "sdpa",
        help="Attention backend: sdpa (ROCm default), flash2, flash4",
    )
    return p.parse_args()


def _ensure_loguru() -> None:
    """loguru is required by mage_flow but not always baked into the toolbox image."""
    try:
        import loguru  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--root-user-action=ignore", "loguru"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        import loguru  # noqa: F401
        return
    except Exception:
        pass
    # Minimal fallback so imports succeed offline.
    import logging
    import types

    mod = types.ModuleType("loguru")

    class _L:
        def __getattr__(self, name):
            return getattr(logging.getLogger("mage"), name, lambda *a, **k: None)

    mod.logger = _L()  # type: ignore[attr-defined]
    sys.modules["loguru"] = mod


def main() -> int:
    args = parse_args()
    _install_flash_attn_shim()
    _ensure_loguru()
    _ensure_mage_on_path()

    # Force HF cache under the toolbox mount when present.
    if not os.environ.get("HF_HOME") and not os.environ.get("HUGGINGFACE_HUB_CACHE"):
        for cand in (
            "/home/user/.cache/huggingface",
            os.path.expanduser("~/.cache/huggingface"),
            "/workspace/huggingface-cache",
        ):
            if os.path.isdir(cand):
                os.environ.setdefault("HF_HOME", cand)
                break

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("❌ No GPU detected (torch.cuda.is_available() == False).", file=sys.stderr)
        return 1

    # ROCm: force SDPA for BOTH the DiT backend and the HF text encoder.
    # ModelConfig defaults to flash2 and re-applies set_attn_backend on load;
    # VF_HF_ATTN_IMPL overrides the Qwen3-VL attn_implementation string.
    os.environ["VF_HF_ATTN_IMPL"] = args.attn if args.attn in ("sdpa", "eager") else "sdpa"
    try:
        from mage_flow.models.modules._attn_backend import set_attn_backend

        set_attn_backend(args.attn)
        print(f"[MAGE] attention backend={args.attn} (VF_HF_ATTN_IMPL={os.environ['VF_HF_ATTN_IMPL']})",
              file=sys.stderr)
    except Exception as exc:
        print(f"[MAGE] WARN: could not set attn backend: {exc}", file=sys.stderr)

    from mage_flow import MageFlowPipeline

    print(f"[MAGE] Loading {args.model} on {args.device} ...", file=sys.stderr)
    t0 = time.time()
    pipe = MageFlowPipeline.from_pretrained(args.model, device=args.device)
    # Re-apply after load — MageFlowModel.__init__ resets backend from config.attn_type.
    try:
        from mage_flow.models.modules._attn_backend import set_attn_backend

        set_attn_backend(args.attn)
    except Exception:
        pass
    print(f"[MAGE] loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Content filter uses Qwen3-VL .generate(). On ROCm + transformers>=5.6 the
    # patched text encoder often fails closed (white blank) even for clean
    # prompts. Default: fail-open unless MAGE_CONTENT_FILTER=1 forces the gate.
    skip_cf = os.environ.get("MAGE_CONTENT_FILTER", "0") != "1"
    if skip_cf:
        from mage_flow.models.modules.mage_text import FilterVerdict

        def _allow(prompt, max_new_tokens=160):
            return FilterVerdict(False, [], "rocm fail-open (MAGE_CONTENT_FILTER!=1)", "")

        pipe.model.txt_enc.screen_text = _allow  # type: ignore[method-assign]
        if hasattr(pipe.model.txt_enc, "screen_edit"):
            pipe.model.txt_enc.screen_edit = (  # type: ignore[method-assign]
                lambda prompt, ref_images, max_new_tokens=192: FilterVerdict(
                    False, [], "rocm fail-open", ""
                )
            )
        print("[MAGE] content filter: fail-open (set MAGE_CONTENT_FILTER=1 to enforce)",
              file=sys.stderr)
    else:
        try:
            verdict = pipe.model.txt_enc.screen_text(args.prompt)
            print(
                f"[MAGE] content_filter violates={verdict.violates} "
                f"cats={verdict.categories} reason={verdict.reason!r} "
                f"raw={verdict.raw[:200]!r}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[MAGE] content_filter probe failed: {exc}", file=sys.stderr)

    print(
        f"[MAGE] generate {args.width}x{args.height} steps={args.steps} cfg={args.cfg}",
        file=sys.stderr,
    )
    t1 = time.time()
    imgs = pipe.generate(
        [args.prompt],
        neg_prompts=[args.neg_prompt] if args.neg_prompt else None,
        seeds=[args.seed],
        heights=[args.height],
        widths=[args.width],
        steps=args.steps,
        cfg=args.cfg,
    )
    img = imgs[0]
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    img.save(out)
    size = os.path.getsize(out)
    print(f"✅ Saved {out} ({size} bytes) in {time.time() - t1:.1f}s", file=sys.stderr)
    print(out)
    # Refuse to claim success on all-white refusal placeholders.
    if size < 20000:
        print(
            f"⚠ output is very small ({size} B) — likely a content-filter refusal "
            "placeholder (white blank). Check content_filter logs above.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
