#!/usr/bin/env python3
"""Diagnose Mage-Flow noise on ROCm: weight load, TE, denoise steps, VAE, math SDPA."""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MAGE_FA_SHIM", "1")
os.environ.setdefault("VF_HF_ATTN_IMPL", "sdpa")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("MAGE_ROOT", "/opt/Mage"))

from mage_serve import _install_flash_attn_shim, _ensure_loguru, _ensure_mage_on_path  # noqa: E402

_install_flash_attn_shim()
_ensure_loguru()
_ensure_mage_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from mage_flow import MageFlowPipeline  # noqa: E402
from mage_flow.models.modules._attn_backend import set_attn_backend  # noqa: E402
from mage_flow.models.modules.mage_text import FilterVerdict  # noqa: E402
from mage_flow.pipeline import _encode_texts_packed, _template_info  # noqa: E402

OUT = Path(os.environ.get("MAGE_OUT_DIR", "/out"))
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("MAGE_MODEL", "/models/Mage-Flow-Turbo")
ATTN = os.environ.get("MAGE_ATTN", "sdpa")


def _img_stats(img: Image.Image) -> str:
    a = np.asarray(img, dtype=np.float32)
    return f"size={img.size} mean={a.mean():.1f} std={a.std():.1f} min={a.min()} max={a.max()}"


def _is_noise(img: Image.Image) -> bool:
    a = np.asarray(img, dtype=np.float32)
    return 90 < a.mean() < 140 and a.std() < 40


def main() -> int:
    print("cuda", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), flush=True)
        print("bf16", torch.cuda.is_bf16_supported(), flush=True)

    set_attn_backend(ATTN)
    print(f"Loading {MODEL} attn={ATTN} ...", flush=True)
    t0 = time.time()
    pipe = MageFlowPipeline.from_pretrained(MODEL, device="cuda")
    set_attn_backend(ATTN)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    pipe.model.txt_enc.screen_text = lambda *a, **k: FilterVerdict(
        False, [], "open", ""
    )
    model = pipe.model

    # Weight integrity
    print("\n=== weight norms ===", flush=True)
    for name, mod in [
        ("transformer", model.transformer),
        ("txt_enc", model.txt_enc.hf_module if hasattr(model.txt_enc, "hf_module") else model.txt_enc),
        ("vae", model.vae),
    ]:
        params = list(mod.parameters())
        w = torch.cat([p.detach().float().reshape(-1)[:50000] for p in params[:20]])
        print(
            f"  {name}: nparams_sample={w.numel()} mean={w.mean():.5f} std={w.std():.5f} "
            f"absmax={w.abs().max():.5f} dtype0={params[0].dtype}",
            flush=True,
        )
    # First few transformer param names + norms (detect random init ~0.02)
    n_show = 0
    for n, p in model.transformer.named_parameters():
        if p.ndim >= 2:
            print(f"  dit[{n}]: shape={tuple(p.shape)} std={p.float().std():.5f}", flush=True)
            n_show += 1
            if n_show >= 5:
                break

    # TE via packed API
    print("\n=== TE packed encode ===", flush=True)
    info = _template_info("mage-flow")
    template = info["template"]
    drop_idx = int(info["start_idx"])
    for p in ["a red apple on a white table", "zzzz meaningless xxxx"]:
        try:
            with torch.no_grad():
                txt, vec, lens = _encode_texts_packed(
                    model, [p], template, drop_idx, torch.device("cuda")
                )
            print(
                f"  {p!r}: txt={tuple(txt.shape)} mean={txt.float().mean():.4f} "
                f"std={txt.float().std():.4f} absmax={txt.float().abs().max():.4f} "
                f"vec_std={vec.float().std():.4f} lens={lens}",
                flush=True,
            )
            if torch.isnan(txt).any() or torch.isinf(txt).any():
                print("  !! NaN/Inf in TE output", flush=True)
        except Exception as e:
            print(f"  TE FAIL {p!r}: {e}", flush=True)
            traceback.print_exc()

    # Generate with step stats
    print("\n=== generate 512 seed=42 ===", flush=True)
    step_stats = []
    orig = model.transformer.forward

    def wrap(*a, **k):
        out = orig(*a, **k)
        x = out[0] if isinstance(out, tuple) else out
        with torch.no_grad():
            xf = x.float()
            step_stats.append(
                (
                    float(xf.mean()),
                    float(xf.std()),
                    float(xf.abs().max()),
                    bool(torch.isnan(xf).any()),
                    bool(torch.isinf(xf).any()),
                )
            )
        return out

    model.transformer.forward = wrap
    t1 = time.time()
    imgs = pipe.generate(
        ["a single ripe red apple on a white table, soft studio lighting"],
        seeds=[42],
        heights=[512],
        widths=[512],
        steps=4,
        cfg=1.0,
    )
    model.transformer.forward = orig
    print(f"gen {time.time() - t1:.1f}s steps={len(step_stats)}", flush=True)
    for i, s in enumerate(step_stats):
        print(
            f"  step{i}: mean={s[0]:.5f} std={s[1]:.5f} absmax={s[2]:.5f} nan={s[3]} inf={s[4]}",
            flush=True,
        )
    img = imgs[0]
    print("out", _img_stats(img), "NOISE" if _is_noise(img) else "MAYBE_OK", flush=True)
    img.save(OUT / "diag_512_sdpa_aot.png")

    # Math-only SDPA
    print("\n=== math-only SDPA gen 256 ===", flush=True)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        ctx = sdpa_kernel([SDPBackend.MATH])
    except Exception:
        ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )
    with ctx:
        t2 = time.time()
        imgs = pipe.generate(
            ["a bright yellow banana on solid blue background"],
            seeds=[7],
            heights=[256],
            widths=[256],
            steps=4,
            cfg=1.0,
        )
        img = imgs[0]
        print(
            f"math gen {time.time()-t2:.1f}s {_img_stats(img)} "
            f"{'NOISE' if _is_noise(img) else 'MAYBE_OK'}",
            flush=True,
        )
        img.save(OUT / "diag_256_math_sdpa.png")

    # More steps + cfg
    print("\n=== 20 steps cfg=3.5 gen 512 ===", flush=True)
    t3 = time.time()
    imgs = pipe.generate(
        ["a single ripe red apple on a white table, soft studio lighting, photoreal"],
        seeds=[42],
        heights=[512],
        widths=[512],
        steps=20,
        cfg=3.5,
    )
    img = imgs[0]
    print(
        f"long gen {time.time()-t3:.1f}s {_img_stats(img)} "
        f"{'NOISE' if _is_noise(img) else 'MAYBE_OK'}",
        flush=True,
    )
    img.save(OUT / "diag_512_20step_cfg35.png")

    # VAE roundtrip
    print("\n=== VAE roundtrip solid red ===", flush=True)
    try:
        red = torch.zeros(1, 3, 256, 256, device="cuda")
        red[:, 0] = 1.0
        red = red * 2 - 1
        vd = next(model.vae.parameters()).dtype
        with torch.no_grad():
            enc = model.vae.encode(red.to(dtype=vd))
            z = enc if torch.is_tensor(enc) else getattr(enc, "sample", None)
            if z is None:
                z = enc[0] if isinstance(enc, (tuple, list)) else enc
            print(
                f"  lat shape={tuple(z.shape)} mean={z.float().mean():.4f} "
                f"std={z.float().std():.4f}",
                flush=True,
            )
            rec = model.vae.decode(z.float() if z.dtype != torch.float32 else z)
            if not torch.is_tensor(rec):
                rec = getattr(rec, "sample", rec[0])
            rec = rec.clamp(-1, 1)
            print(f"  rec ch_means={rec.float().mean([0,2,3]).tolist()}", flush=True)
            out = ((rec + 1) * 127.5).byte().cpu().numpy()[0].transpose(1, 2, 0)
            Image.fromarray(out).save(OUT / "diag_vae_red.png")
            print(f"  saved mean={out.mean():.1f} std={out.std():.1f}", flush=True)
    except Exception as e:
        print(f"  VAE FAIL: {e}", flush=True)
        traceback.print_exc()

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
