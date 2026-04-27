#!/usr/bin/env python3
"""
HeartMuLa music-generation launcher for Strix Halo (ROCm).

Invoked from inside the strix-halo-comfyui container by the orchestrator
(`run_fleet.py::run_heartmula_gen`):

    python3 /workspace/scripts/heartmula_launcher.py \
        --prompt "<freeform tag text>" \
        --duration <float seconds> \
        --out <output wav path> \
        --real

Without --real this is a no-op smoke test that exits 0 immediately.

heartlib expects two textual inputs (lyrics, tags). The orchestrator only
gives us a single freeform prompt — we treat it as tags and leave lyrics
empty (instrumental).

Model layout (parent directory contains both submodels + tokenizer +
gen_config). Override with HEARTMULA_MODEL_ROOT, or override the
individual subdirs with HEARTMULA_MULA_PATH / HEARTMULA_CODEC_PATH (if
either of those is set, its parent dir is used as the root).
"""

import argparse
import os
import sys
import traceback


DEFAULT_MODEL_ROOT = "/mnt/downloads/comfy-models/HeartMuLa"
DEFAULT_VERSION = "3B"


def parse_args():
    p = argparse.ArgumentParser(description="HeartMuLa music generator")
    p.add_argument("--prompt", required=True,
                   help="Freeform tag text (genre/mood/instruments). "
                        "Used as the `tags` input; lyrics will be empty.")
    p.add_argument("--duration", type=float, required=True,
                   help="Target audio length in seconds.")
    p.add_argument("--out", required=True,
                   help="Output WAV path. torchaudio infers format from "
                        "extension; .wav writes 48 kHz PCM WAV.")
    p.add_argument("--real", action="store_true",
                   help="Without this flag the script exits 0 immediately "
                        "(smoke test).")
    p.add_argument("--version", default=DEFAULT_VERSION,
                   help="HeartMuLa version suffix (default: 3B).")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    return p.parse_args()


def resolve_model_root() -> str:
    """Honor env-var overrides; fall back to the on-disk default.

    heartlib's from_pretrained takes a *parent* directory containing
    HeartMuLa-oss-{version}/, HeartCodec-oss/, tokenizer.json, gen_config.json.
    If the user sets HEARTMULA_MULA_PATH or HEARTMULA_CODEC_PATH (which point
    at the submodels), we derive the parent.
    """
    root = os.environ.get("HEARTMULA_MODEL_ROOT")
    if root:
        return root
    for var in ("HEARTMULA_MULA_PATH", "HEARTMULA_CODEC_PATH"):
        sub = os.environ.get(var)
        if sub:
            return os.path.dirname(os.path.normpath(sub))
    return DEFAULT_MODEL_ROOT


def main() -> int:
    args = parse_args()

    if not args.real:
        print("[heartmula] --real not set; smoke-test exit 0.", flush=True)
        return 0

    # ROCm gfx1151 override for Strix Halo. Set before torch import.
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")

    try:
        import torch
        from heartlib import HeartMuLaGenPipeline
        # Upstream heartlib calls HeartCodec.from_pretrained without
        # ignore_mismatched_sizes=True. The shipped checkpoint has
        # `flow_matching.vq_embed.layers.{0..7}._codebook.initted` as a
        # 1-element tensor [1] while the in-code model defines it as a
        # 0-d scalar []. The mismatch is BENIGN — it's a codebook init
        # tracker, not a learned weight — but a recent transformers
        # version raises on ANY mismatch by default. Force-default
        # ignore_mismatched_sizes=True on the codec load so loading
        # succeeds; the codebook will just re-init from the ckpt.
        from heartlib.heartcodec.modeling_heartcodec import HeartCodec
        _orig_codec_from_pretrained = HeartCodec.from_pretrained.__func__ \
            if hasattr(HeartCodec.from_pretrained, '__func__') \
            else HeartCodec.from_pretrained
        def _patched_codec_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault("ignore_mismatched_sizes", True)
            return _orig_codec_from_pretrained(cls, *args, **kwargs)
        HeartCodec.from_pretrained = classmethod(_patched_codec_from_pretrained)
        # heartlib calls torchaudio.save() in postprocess. Recent
        # torchaudio routes save through TorchCodec (not installed
        # here). Replace with a soundfile-based writer that handles
        # the (channels, samples) tensor heartlib hands it. The
        # signature matches torchaudio.save's positional args
        # (uri, src_tensor, sample_rate, ...).
        import torchaudio
        import soundfile as _sf
        def _patched_save(uri, src, sample_rate, *args, **kwargs):
            # heartlib passes shape (C, N) on CPU as float32. soundfile
            # wants (N,) for mono or (N, C) for multi-channel.
            t = src.detach().cpu().to(torch.float32).numpy()
            if t.ndim == 2:
                # (C, N) → (N, C); if C=1 squeeze to mono.
                t = t.T
                if t.shape[1] == 1:
                    t = t[:, 0]
            _sf.write(str(uri), t, int(sample_rate))
        torchaudio.save = _patched_save
    except Exception as e:
        print(f"[heartmula] FATAL: import failed: {e}", file=sys.stderr,
              flush=True)
        traceback.print_exc()
        return 2

    model_root = resolve_model_root()
    if not os.path.isdir(model_root):
        print(f"[heartmula] FATAL: model root not found: {model_root}",
              file=sys.stderr, flush=True)
        return 3

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not torch.cuda.is_available():
        print("[heartmula] WARNING: torch.cuda.is_available() is False; "
              "ROCm/CUDA device will likely fail to initialise.",
              file=sys.stderr, flush=True)

    duration_ms = max(1000, int(args.duration * 1000))
    print(f"[heartmula] root={model_root} version={args.version} "
          f"duration_ms={duration_ms} out={out_path}", flush=True)
    print(f"[heartmula] tags={args.prompt!r}", flush=True)
    print("[heartmula] loading pipeline (lazy_load=False — first load "
          "pulls ~21 GB of weights, expect ~2-3 min before generation "
          "starts)...", flush=True)

    try:
        device = torch.device("cuda")
        pipe = HeartMuLaGenPipeline.from_pretrained(
            model_root,
            device={"mula": device, "codec": device},
            dtype={"mula": torch.bfloat16, "codec": torch.float32},
            version=args.version,
            lazy_load=False,
        )
    except Exception as e:
        print(f"[heartmula] FATAL: pipeline load failed: {e}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        return 4

    # torchtune's Llama3ScaledRoPE.__init__ calls rope_init() — but if
    # freqs.is_meta (memory-efficient meta-device load), the function
    # bails BEFORE setting is_cache_built=True. After weights land,
    # the cache is still unbuilt and the first forward() raises
    # 'RoPE cache is not built'. Walk the loaded modules + call
    # rope_init() now that the tensors are real.
    rope_count = 0
    for m in pipe.mula.modules():
        if hasattr(m, 'rope_init') and callable(m.rope_init):
            try:
                m.rope_init()
                # rope_init() builds the cache on the DEFAULT device
                # (cpu); the surrounding attention module's tensors
                # are on cuda. Move the cache buffer over so the
                # forward-pass index op doesn't cross devices.
                if hasattr(m, 'cache') and isinstance(m.cache, torch.Tensor):
                    m.cache = m.cache.to(device)
                rope_count += 1
            except Exception as e:
                print(f"[heartmula] WARN: rope_init failed on {type(m).__name__}: {e}",
                      file=sys.stderr, flush=True)
    if rope_count:
        print(f"[heartmula] rope_init: built cache on {rope_count} modules (moved to {device})",
              flush=True)

    # tags/lyrics: heartlib's preprocess() accepts either a string literal or
    # a path (it tests os.path.isfile and reads the file if so). Passing the
    # prompt as a literal string is simplest. lyrics="" -> instrumental.
    inputs = {"lyrics": "", "tags": args.prompt}

    try:
        with torch.no_grad():
            pipe(
                inputs,
                max_audio_length_ms=duration_ms,
                save_path=out_path,
                topk=args.topk,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
            )
    except Exception as e:
        print(f"[heartmula] FATAL: generation failed: {e}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        return 5

    if not os.path.isfile(out_path) or os.path.getsize(out_path) < 1024:
        print(f"[heartmula] FATAL: output missing or too small: {out_path}",
              file=sys.stderr, flush=True)
        return 6

    print(f"[heartmula] OK wrote {out_path} "
          f"({os.path.getsize(out_path)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
