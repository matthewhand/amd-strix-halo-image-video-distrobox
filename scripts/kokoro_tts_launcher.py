#!/usr/bin/env python3
"""Kokoro-82M TTS launcher — CLI wrapper around kokoro-onnx for Strix Halo.

Usage:
    python3 kokoro_tts_launcher.py --text "hello world" --voice af_heart \\
        --out /path/out.wav [--lang en-us] [--speed 1.0]

Why this exists alongside qwen_tts_launcher.py:
    Qwen3-TTS's precompiled HIP kernels don't have a gfx1151 variant (even
    with HSA_OVERRIDE_GFX_VERSION=11.0.0), so /tts requests hit
    `hipErrorInvalidImage` on this hardware. Kokoro-82M ships as ONNX, runs
    on the CPU via onnxruntime, and works out of the box. ~8x realtime on
    a Strix Halo CPU — generates 1 minute of audio in ~7 seconds.

54 voices ship with kokoro-v1.0:
    af_*, am_*  — American female / male
    bf_*, bm_*  — British
    ef_*, em_*  — Spanish
    ff_*        — French
    hf_*, hm_*  — Hindi
    if_*, im_*  — Italian
    jf_*, jm_*  — Japanese
    pf_*, pm_*  — Portuguese
    zf_*, zm_*  — Chinese (Mandarin)

Model files (~325 MB ONNX + ~28 MB voices) are downloaded on first run to
KOKORO_MODEL_DIR (default /workspace/.kokoro inside the container or
~/.cache/kokoro on the host). Resumable; second run is instant.

Outputs mono 24kHz PCM WAV.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path


KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

DEFAULT_DIR = "/workspace/.kokoro"
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG = "en-us"
MIN_FREE_GB = 2  # ~370 MB needed; 2 GB safety floor


def _first_writable(candidates):
    for cand in candidates:
        if not cand:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".kokoro-write-probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.unlink(probe)
            return cand
        except (OSError, PermissionError):
            continue
    raise RuntimeError("no writable model dir among candidates")


def _disk_guard(path: str) -> None:
    """Exit(2) if target mount has < MIN_FREE_GB free."""
    import shutil
    target = path
    while target and not os.path.exists(target):
        target = os.path.dirname(target) or "/"
    free_gb = shutil.disk_usage(target).free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        print(f"[kokoro-tts] ERROR: insufficient disk: {free_gb:.1f} GB free on {target} "
              f"(need {MIN_FREE_GB})", file=sys.stderr)
        sys.exit(2)


def _download(url: str, dest: Path) -> None:
    """Resumable single-file download with progress hint."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[kokoro-tts] {dest.name} cached ({dest.stat().st_size} bytes)", file=sys.stderr)
        return
    print(f"[kokoro-tts] downloading {url} → {dest}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)
    print(f"[kokoro-tts] wrote {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)


def _resolve_model_files(model_dir: str) -> tuple[str, str]:
    """Ensure model.onnx + voices.bin exist locally, downloading on miss."""
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    model = d / "kokoro-v1.0.onnx"
    voices = d / "voices-v1.0.bin"
    _download(KOKORO_MODEL_URL, model)
    _download(KOKORO_VOICES_URL, voices)
    return str(model), str(voices)


# Per-voice-prefix → kokoro language code. The `voice` ids encode their
# language by prefix: af_/am_ → American English, bf_/bm_ → British,
# etc. When the caller doesn't supply --lang explicitly we auto-pick
# the right G2P/phonemizer language; otherwise English IPA conversion
# of e.g. Japanese text gives wrong pronunciation. en-us is the safe
# fallback for unrecognized prefixes.
LANG_BY_VOICE_PREFIX = {
    "af_": "en-us", "am_": "en-us",
    "bf_": "en-gb", "bm_": "en-gb",
    "ef_": "es",    "em_": "es",
    "ff_": "fr-fr",
    "hf_": "hi",    "hm_": "hi",
    "if_": "it",    "im_": "it",
    "jf_": "ja",    "jm_": "ja",
    "pf_": "pt-br", "pm_": "pt-br",
    "zf_": "cmn",   "zm_": "cmn",
}


def _auto_lang(voice: str) -> str:
    for prefix, lang in LANG_BY_VOICE_PREFIX.items():
        if voice.startswith(prefix):
            return lang
    return "en-us"


def synthesize(text: str, voice: str, out_path: str, lang: str, speed: float, model_dir: str) -> None:
    _disk_guard(model_dir)
    model_path, voices_path = _resolve_model_files(model_dir)

    from kokoro_onnx import Kokoro  # type: ignore
    import soundfile as sf  # type: ignore

    if lang == "auto":
        lang = _auto_lang(voice)
        print(f"[kokoro-tts] auto-lang from voice prefix '{voice[:3]}' → {lang}", file=sys.stderr)
    print(f"[kokoro-tts] voice={voice} lang={lang} speed={speed} out={out_path}", file=sys.stderr)
    t0 = time.time()
    k = Kokoro(model_path, voices_path)
    available = list(k.get_voices())
    if voice not in available:
        print(f"[kokoro-tts] ERROR: unknown voice '{voice}'. "
              f"Available ({len(available)}): {', '.join(sorted(available))}",
              file=sys.stderr)
        sys.exit(3)
    samples, sr = k.create(text, voice=voice, speed=speed, lang=lang)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, samples, sr)
    dt = time.time() - t0
    dur = len(samples) / sr
    print(f"[kokoro-tts] wrote {out_path} ({len(samples)} samples @ {sr}Hz = {dur:.2f}s, "
          f"gen {dt:.1f}s = {dur/dt:.1f}x realtime)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kokoro-82M WAV synthesis launcher")
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default=DEFAULT_VOICE,
                    help="Voice id (af_heart, am_eric, bf_emma, etc.). "
                         "Set to 'list' to print all 54 voices and exit.")
    ap.add_argument("--out", required=True)
    # Default lang is "auto" → derived from voice prefix at synthesize-time.
    # Pass an explicit code (en-us / en-gb / ja / cmn / ...) to override.
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--model-dir", default=os.environ.get("KOKORO_MODEL_DIR", DEFAULT_DIR))
    args = ap.parse_args(argv)

    if args.voice == "list":
        # Cheap path: load just enough to enumerate voices
        from kokoro_onnx import Kokoro  # type: ignore
        m, v = _resolve_model_files(args.model_dir)
        k = Kokoro(m, v)
        for voice in sorted(k.get_voices()):
            print(voice)
        return 0

    try:
        synthesize(args.text, args.voice, args.out, args.lang, args.speed, args.model_dir)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[kokoro-tts] FATAL: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
