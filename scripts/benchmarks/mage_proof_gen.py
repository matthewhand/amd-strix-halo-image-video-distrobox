#!/usr/bin/env python3
"""Generate a few Mage-Flow proof images and classify noise vs scene."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MAGE_FA_SHIM", "1")
os.environ.setdefault("VF_HF_ATTN_IMPL", "sdpa")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("MAGE_CONTENT_FILTER", "0")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("MAGE_ROOT", "/opt/Mage"))

from mage_serve import _install_flash_attn_shim, _ensure_loguru, _ensure_mage_on_path  # noqa: E402

_install_flash_attn_shim()
_ensure_loguru()
_ensure_mage_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mage_flow import MageFlowPipeline  # noqa: E402
from mage_flow.models.modules._attn_backend import set_attn_backend  # noqa: E402
from mage_flow.models.modules.mage_text import FilterVerdict  # noqa: E402

OUT = Path(os.environ.get("MAGE_OUT_DIR", "/out"))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("MAGE_MODEL", "/models/Mage-Flow-Turbo")

PROMPTS = [
    ("red_apple", "a single ripe red apple on a white table, soft studio lighting, photoreal"),
    ("coffee_mug", "a blue ceramic mug of coffee with rising steam, morning window light"),
    ("cyberpunk", "cyberpunk street at midnight, dense neon signs, wet asphalt reflections, rain haze"),
]


def is_noise(img) -> bool:
    a = np.asarray(img, dtype=np.float32)
    mean, std = float(a.mean()), float(a.std())
    # Mid-gray low-structure static from the corrupt-weight era
    return 90.0 < mean < 140.0 and std < 40.0


def main() -> int:
    set_attn_backend("sdpa")
    print(f"Loading {MODEL} ...", flush=True)
    t0 = time.time()
    pipe = MageFlowPipeline.from_pretrained(MODEL, device="cuda")
    set_attn_backend("sdpa")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    # weight sanity
    txt_std = float(pipe.model.transformer.txt_in.weight.float().std())
    te_w = next(pipe.model.txt_enc.hf_module.parameters())
    te_std = float(te_w.float().std())
    print(f"txt_in.weight std={txt_std:.5f} te_param0 std={te_std:.5f}", flush=True)
    if txt_std < 1e-4:
        print("FAIL: txt_in still near-zero — corrupt DiT weights", flush=True)
        return 2

    pipe.model.txt_enc.screen_text = lambda *a, **k: FilterVerdict(
        False, [], "open", ""
    )

    results = []
    ok = 0
    for slug, prompt in PROMPTS:
        t1 = time.time()
        imgs = pipe.generate(
            [prompt],
            seeds=[42],
            heights=[768],
            widths=[768],
            steps=4,
            cfg=1.0,
        )
        img = imgs[0]
        path = OUT / f"proof_fresh_{slug}.png"
        img.save(path)
        a = np.asarray(img, dtype=np.float32)
        noise = is_noise(img)
        if not noise:
            ok += 1
        row = {
            "slug": slug,
            "path": str(path),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "noise": noise,
            "elapsed_s": round(time.time() - t1, 1),
            "bytes": path.stat().st_size,
        }
        results.append(row)
        print(
            f"{'NOISE' if noise else 'OK   '} {slug} mean={row['mean']:.1f} "
            f"std={row['std']:.1f} {row['elapsed_s']}s {path}",
            flush=True,
        )

    summary = {"ok": ok, "total": len(PROMPTS), "results": results}
    (OUT / "proof_fresh_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if ok == len(PROMPTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
