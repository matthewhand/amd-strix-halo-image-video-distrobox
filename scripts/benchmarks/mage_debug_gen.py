#!/usr/bin/env python3
"""Debug Mage-Flow generation quality on ROCm."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MAGE_FA_SHIM", "1")
os.environ.setdefault("VF_HF_ATTN_IMPL", "sdpa")

sys.path.insert(0, os.environ.get("MAGE_ROOT", "/opt/Mage"))
sys.path.insert(0, "/opt/Mage")

import numpy as np
import torch
from PIL import Image


def main():
    print("cuda", torch.cuda.is_available(), flush=True)
    from mage_flow.models.modules._attn_backend import set_attn_backend

    set_attn_backend("sdpa")
    from mage_flow import MageFlowPipeline

    model_path = os.environ.get("MAGE_MODEL", "/models/Mage-Flow-Turbo")
    print("loading", model_path, flush=True)
    pipe = MageFlowPipeline.from_pretrained(model_path, device="cuda")
    model = pipe.model

    # Fail-open content filter
    try:
        from mage_flow.models.modules.mage_text import FilterVerdict

        model.txt_enc.screen_text = lambda prompt, max_new_tokens=160: FilterVerdict(
            False, [], "debug fail-open", ""
        )
        print("content filter fail-open", flush=True)
    except Exception as e:
        print("filter patch failed", e, flush=True)

    # Parameter dtypes
    dtypes = {}
    for n, p in model.named_parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
        if len(dtypes) > 5:
            break
    print("param dtypes sample counts", dtypes, flush=True)

    prompt = "a bright red apple on a pure white table, studio light"
    print("generate", prompt, flush=True)
    imgs = pipe.generate(
        [prompt],
        steps=4,
        cfg=1.0,
        heights=[256],
        widths=[256],
        seeds=[123],
    )
    im = imgs[0]
    a = np.asarray(im)
    print(
        "result mean",
        a.mean(axis=(0, 1)),
        "std",
        a.std(),
        "min",
        a.min(),
        "max",
        a.max(),
        flush=True,
    )
    out = "/opt/ComfyUI/output/mage/debug_apple256.png"
    im.save(out)
    print("saved", out, flush=True)

    # Probe text embeddings magnitude
    try:
        from mage_flow.pipeline import _encode_texts_packed, _template_info

        info = _template_info("mage-flow")
        template = info.get("template", "{}")
        drop_idx = int(info.get("start_idx", 0))
        txt, vec, lens = _encode_texts_packed(
            model, [prompt], template, drop_idx, torch.device("cuda")
        )
        print(
            "txt shape",
            tuple(txt.shape),
            "vec",
            tuple(vec.shape),
            "txt abs mean",
            float(txt.float().abs().mean()),
            "vec abs mean",
            float(vec.float().abs().mean()),
            "lens",
            lens,
            flush=True,
        )
        if float(txt.float().abs().mean()) < 1e-6:
            print("WARNING: text embeddings near zero — conditioning broken", flush=True)
    except Exception as e:
        import traceback

        print("text probe failed", e, flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
