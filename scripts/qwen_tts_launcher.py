#!/usr/bin/env python3
"""Qwen3-TTS launcher — CLI wrapper that loads Qwen3-TTS and emits a WAV.

Usage:
    python3 qwen_tts_launcher.py --text "hello world" --voice ryan --out /path/out.wav \
        [--model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice]

Forces HSA_OVERRIDE_GFX_VERSION=11.0.0 (required for this model on gfx1151 —
NOT 11.5.1 like the rest of the stack) and attn_implementation="sdpa"
(flash-attn-2 does not build on gfx1151 for Qwen3-TTS).

Weights are resumably downloaded via huggingface_hub.snapshot_download to
${HF_HOME:-~/.cache/huggingface}. A disk-space pre-flight guard aborts with
exit code 2 if the target filesystem has <15 GB free.

Fallback: Kokoro-82M could be dropped in here (same CLI surface) if Qwen3-TTS
breaks on a future ROCm bump. Not implemented — Qwen is the primary path.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback

# Force gfx1151-compatible ROCm env BEFORE torch imports.
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

MIN_FREE_GB = 15
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def _disk_guard(path: str) -> None:
    """Exit(2) if target mount has <MIN_FREE_GB free."""
    target = path
    while target and not os.path.exists(target):
        target = os.path.dirname(target) or "/"
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        msg = (
            f"insufficient disk: {free_gb:.1f} GB free on {target}, "
            f"needs {MIN_FREE_GB} GB"
        )
        print(f"[qwen-tts] ERROR: {msg}", file=sys.stderr)
        sys.exit(2)


def _snapshot(model_id: str, cache_dir: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
    )


def synthesize(text: str, voice: str, out_path: str, model_id: str) -> None:
    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    _disk_guard(cache_dir)

    print(f"[qwen-tts] model={model_id} voice={voice} out={out_path}", file=sys.stderr)
    print(f"[qwen-tts] cache={cache_dir}", file=sys.stderr)
    local_dir = _snapshot(model_id, cache_dir)
    print(f"[qwen-tts] weights at {local_dir}", file=sys.stderr)

    # Qwen3-TTS is loaded via the official `qwen-tts` package (NOT
    # transformers.AutoModel) — `qwen3_tts` model_type isn't registered
    # in transformers' CONFIG_MAPPING upstream as of 5.8.0.dev0, and
    # the snapshot ships no remote modeling*.py for trust_remote_code
    # to use. The qwen_tts package provides Qwen3TTSModel.from_pretrained
    # which registers the model class internally before delegating
    # to AutoModel under the hood. See:
    # https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    # ("During model loading in the qwen-tts package or vLLM, model
    # weights will be automatically downloaded...")
    import torch  # type: ignore
    from qwen_tts import Qwen3TTSModel  # type: ignore

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # `eager` routes around an SDPA kernel-image mismatch on Strix Halo
    # gfx1151 even with HSA_OVERRIDE_GFX_VERSION=11.0.0 set: the
    # qwen-tts package's precompiled SDPA path raises
    # `hipErrorInvalidImage` on first generation. eager attention
    # uses the manual PyTorch path that compiles per-arch on first
    # forward — slower but functional.
    attn = os.environ.get("QWEN_TTS_ATTN", "eager")
    model = Qwen3TTSModel.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        attn_implementation=attn,
    )
    if hasattr(model, "model"):
        model.model = model.model.to(device).eval()

    # CustomVoice models accept `speaker` (one of 9 premium timbres) +
    # optional `language` hint. The legacy `voice` arg in this launcher's
    # CLI maps to `speaker`. generate_custom_voice returns
    # (List[np.ndarray], sample_rate).
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=voice,
    )
    if not wavs:
        raise RuntimeError("qwen_tts returned empty wav list")
    wav = wavs[0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    import wave
    import numpy as np  # type: ignore

    pcm = np.clip(wav, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())

    print(f"[qwen-tts] wrote {out_path} ({len(pcm)} samples @ {sr}Hz)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen3-TTS WAV synthesis launcher")
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default="ryan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)

    try:
        synthesize(args.text, args.voice, args.out, args.model)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - real-runtime errors
        print(f"[qwen-tts] FATAL: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
