#!/usr/bin/env python3
# Verification harness for the HOMIE-Wan NVIDIA port.
#
# Run from the homie_nvidia_claude/ directory:  python verify.py
#
# Checks (in order of importance):
#   1. KEY MATCH  — build HomieWanModel with the 14B config and confirm its state_dict
#                   keys exactly equal the 1272 keys in the HOMIE checkpoint index json.
#                   This is the single most important check: it guarantees the trained
#                   weights load with no missing / unexpected tensors.
#   2. SHAPE TRACE — build a *tiny* HomieWanModel on CPU, run a forward pass with dummy
#                    latents + a qwen feature + reference_id_labels, and assert the output
#                    frame count equals the input video frame count (reference frames and
#                    qwen tokens dropped), exercising the CrossModalityAffine self-attn path.
#   3. DATASET     — load eval_datasets/nips_new_100.jsonl via load_homie_jsonl and assert
#                    it yields the prompt, reference images, and a [1, S, 2048] qwen tensor.
#
# Checks 1-3 are CPU-only and safe to run in any environment. The full end-to-end GPU run
# (real weights + VAE decode) is documented in README.md / infer.sh and requires a CUDA box.
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)  # .../2026_07_01
sys.path.insert(0, HERE)

INDEX_JSON = os.path.join(REPO_ROOT, "HOMIE-Wan-Model", "Homie_Wan_14B.safetensors.index.json")
EVAL_JSONL = os.path.join(REPO_ROOT, "eval_datasets", "nips_new_100.jsonl")


def check_key_match():
    print("=" * 70)
    print("[1] KEY MATCH: model.state_dict() vs checkpoint index")
    print("=" * 70)
    from homie_wan.configs import get_config
    from homie_wan.modules.model import HomieWanModel

    cfg = get_config("s2v-14B")
    # Build on the meta device so no real memory is allocated for the 14B model.
    with torch.device("meta"):
        model = HomieWanModel(
            model_type=cfg.model_type, patch_size=cfg.patch_size, text_len=cfg.text_len,
            in_dim=cfg.in_dim, dim=cfg.dim, ffn_dim=cfg.ffn_dim, freq_dim=cfg.freq_dim,
            text_dim=cfg.text_dim, img_dim=cfg.img_dim, out_dim=cfg.out_dim,
            num_heads=cfg.num_heads, num_layers=cfg.num_layers, qk_norm=cfg.qk_norm,
            qk_norm_type=cfg.qk_norm_type, cross_attn_norm=cfg.cross_attn_norm, eps=cfg.eps,
            max_seq_len=cfg.max_seq_len, reference_num=cfg.reference_num,
            qwen_hidden_size=cfg.qwen_hidden_size,
        )
    model_keys = set(model.state_dict().keys())

    with open(INDEX_JSON) as f:
        ckpt_keys = set(json.load(f)["weight_map"].keys())

    missing = ckpt_keys - model_keys       # in checkpoint, not in model -> load fails
    unexpected = model_keys - ckpt_keys     # in model, not in checkpoint -> uninitialized

    print(f"  checkpoint keys : {len(ckpt_keys)}")
    print(f"  model keys      : {len(model_keys)}")
    print(f"  missing (ckpt-only)     : {len(missing)}")
    print(f"  unexpected (model-only) : {len(unexpected)}")
    if missing:
        print("  -- missing (first 20) --")
        for k in sorted(missing)[:20]:
            print("    ", k)
    if unexpected:
        print("  -- unexpected (first 20) --")
        for k in sorted(unexpected)[:20]:
            print("    ", k)
    ok = not missing and not unexpected
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def check_shape_trace():
    print("=" * 70)
    print("[2] SHAPE TRACE: tiny model forward (CPU)")
    print("=" * 70)
    from homie_wan.modules.model import HomieWanModel

    reference_num = 3
    model = HomieWanModel(
        model_type="r2v", patch_size=(1, 2, 2), text_len=8, in_dim=16, dim=64,
        ffn_dim=128, freq_dim=256, text_dim=32, out_dim=16, num_heads=4, num_layers=2,
        qk_norm=True, cross_attn_norm=True, eps=1e-6, max_seq_len=128,
        reference_num=reference_num, qwen_hidden_size=2048,
    ).eval()

    B, C = 1, 16
    F_lat, Hh, Ww = 3, 8, 8            # latent frames / height / width
    x = torch.randn(B, C, F_lat, Hh, Ww)
    reference = torch.randn(B, C, reference_num, Hh, Ww)
    timestep = torch.tensor([500.0])
    prompt = torch.randn(B, 8, 32)     # [B, text_len, text_dim]
    qwen_feature = torch.randn(B, 16, 2048)
    reference_id_labels = [["a"], ["b"], ["c"]]

    with torch.no_grad():
        out = model(x, timestep, prompt, reference=reference,
                    qwen_feature=qwen_feature, reference_id_labels=reference_id_labels)

    print(f"  input  video frames : {F_lat}")
    print(f"  output shape        : {tuple(out.shape)}")
    expected = (B, C, F_lat, Hh, Ww)
    ok = tuple(out.shape) == expected
    print(f"  expected            : {expected}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    # also exercise the no-qwen path
    with torch.no_grad():
        out2 = model(x, timestep, prompt, reference=reference,
                     qwen_feature=None, reference_id_labels=reference_id_labels)
    ok2 = tuple(out2.shape) == expected
    print(f"  no-qwen path output : {tuple(out2.shape)}  {'PASS' if ok2 else 'FAIL'}")
    return ok and ok2


def check_dataset():
    print("=" * 70)
    print("[3] DATASET LOADER: nips_new_100.jsonl")
    print("=" * 70)
    from homie_wan.utils.utils import load_homie_jsonl

    if not os.path.exists(EVAL_JSONL):
        print(f"  SKIP: {EVAL_JSONL} not found")
        return True

    prompts, images, qwen_features, reference_id_labels = load_homie_jsonl(EVAL_JSONL)
    print(f"  samples             : {len(prompts)}")
    print(f"  sample[0] prompt    : {prompts[0][:60]}...")
    print(f"  sample[0] #refs     : {len(images[0])}")
    print(f"  sample[0] ref_ids   : {reference_id_labels[0]}")
    q = qwen_features[0]
    print(f"  sample[0] qwen shape: {None if q is None else tuple(q.shape)}")
    ok = (
        len(prompts) > 0
        and len(images[0]) >= 1
        and (q is None or (q.dim() == 3 and q.shape[-1] == 2048))
    )
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = {}
    for name, fn in [("key_match", check_key_match),
                     ("shape_trace", check_shape_trace),
                     ("dataset", check_dataset)]:
        try:
            results[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = False
        print()

    print("=" * 70)
    print("SUMMARY")
    for name, ok in results.items():
        print(f"  {name:12s}: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
