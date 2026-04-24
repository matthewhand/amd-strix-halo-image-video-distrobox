#!/usr/bin/env python3
"""
HeartMuLa music generator launcher for Strix Halo (ROCm / gfx1151).

HeartMuLa is an LLM-based text+lyrics -> music foundation model (arXiv 2601.10547,
https://github.com/HeartMuLa/heartlib). The 3B backbone autoregressively emits
audio codec tokens that HeartCodec decodes to 48 kHz waveform.

Weights are expected on the host at /mnt/downloads/comfy-models/HeartMuLa/ and
mounted into the container at /workspace/comfy-models/HeartMuLa/ (symlink
tree) or /models/HeartMuLa/. The canonical layout the heartlib quickstart
expects is:

    <ckpt_root>/
        gen_config.json
        tokenizer.json
        HeartCodec-oss/
        HeartMuLa-oss-3B/

This launcher is designed to run *inside* amd-strix-halo-image-video-toolbox:latest.

Usage:
    python3 heartmula_launcher.py --prompt "cinematic orchestral rise" \
                                  --duration 6 \
                                  --out /workspace/music.wav

Smoke-test mode (NO GPU, no inference — just imports & weight check):
    python3 heartmula_launcher.py --prompt test --duration 3 \
                                  --out /tmp/test.wav --no-gpu-test

IMPORTANT: Actual inference is gated behind --real. Without --real the script
performs only the smoke-test path, so callers who accidentally run it while the
fleet is using the GPU will NOT contend for VRAM.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# Config — the standard mount points the fleet uses.
# ----------------------------------------------------------------------------
DEFAULT_CKPT_ROOTS = [
    "/workspace/comfy-models/HeartMuLa",   # via fleet docker -v $PWD:/workspace
    "/models/HeartMuLa",                   # alt mount
    "/mnt/downloads/comfy-models/HeartMuLa",  # host-path fallback
]
HEARTLIB_REPO = "https://github.com/HeartMuLa/heartlib.git"
HEARTLIB_CACHE = "/workspace/.heartlib"   # cloned lazily, persisted via host volume
REQUIRED_TOP_FILES = ("gen_config.json", "tokenizer.json")
REQUIRED_SUBDIRS = ("HeartCodec-oss", "HeartMuLa-oss-3B")


def log(msg):
    print(f"[heartmula] {msg}", flush=True)


def err(msg):
    print(f"[heartmula][ERROR] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# Weight discovery
# ----------------------------------------------------------------------------
def find_ckpt_root(explicit=None):
    roots = [explicit] if explicit else []
    roots += DEFAULT_CKPT_ROOTS
    for r in roots:
        if not r:
            continue
        p = Path(r)
        if not p.is_dir():
            continue
        ok = all((p / f).exists() for f in REQUIRED_TOP_FILES)
        ok = ok and all((p / d).is_dir() for d in REQUIRED_SUBDIRS)
        if ok:
            return p
    return None


def assert_weights(ckpt_root):
    if ckpt_root is None:
        err("No HeartMuLa checkpoint root found. Checked: "
            + ", ".join(DEFAULT_CKPT_ROOTS))
        err("Ensure /mnt/downloads/comfy-models/HeartMuLa is mounted into the container.")
        sys.exit(2)

    missing = []
    for f in REQUIRED_TOP_FILES:
        if not (ckpt_root / f).exists():
            missing.append(f)
    for d in REQUIRED_SUBDIRS:
        if not (ckpt_root / d).is_dir():
            missing.append(d + "/")
    if missing:
        err(f"Checkpoint root {ckpt_root} missing: {missing}")
        sys.exit(2)

    log(f"Checkpoint root OK: {ckpt_root}")
    # Sanity: print codec sample rate from config so callers know the output rate.
    try:
        codec_cfg = json.loads((ckpt_root / "HeartCodec-oss" / "config.json").read_text())
        sr = codec_cfg.get("sample_rate", "unknown")
        log(f"HeartCodec sample_rate = {sr} Hz")
    except Exception as e:
        log(f"(could not read codec config: {e})")


# ----------------------------------------------------------------------------
# Library import / lazy install
# ----------------------------------------------------------------------------
def ensure_heartlib(install_if_missing=True):
    """
    Make `heartlib` importable. Prefer an already-installed package; otherwise
    clone the official repo into HEARTLIB_CACHE and add it to sys.path.

    Returns (ok, detail) where ok is True iff `import heartlib` succeeds,
    and detail is a short human string describing the outcome / blocker.
    """
    try:
        import heartlib  # noqa: F401
        log(f"heartlib already importable from {heartlib.__file__}")
        return True, "preinstalled"
    except ImportError:
        pass

    cache = Path(HEARTLIB_CACHE)
    if cache.exists() and (cache / "pyproject.toml").exists():
        log(f"Found existing heartlib clone at {cache}")
    elif install_if_missing:
        log(f"heartlib not installed — cloning to {cache}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call(
                ["git", "clone", "--depth", "1", HEARTLIB_REPO, str(cache)]
            )
        except Exception as e:
            err(f"git clone failed: {e}")
            return False, f"clone_failed: {e}"
    else:
        return False, "clone_disabled"

    # Add to sys.path rather than pip-installing to avoid mutating the
    # read-only-ish container site-packages on every run. heartlib uses a
    # src-layout so we prepend both the repo root and src/.
    for p in (cache / "src", cache):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import heartlib  # noqa: F401
        log(f"heartlib importable via sys.path: {heartlib.__file__}")
        return True, f"sys.path:{cache}"
    except ImportError as e:
        # Classic case: heartlib source is present but a deep dep
        # (torchtune, encodec, etc.) isn't installed in the toolbox image.
        msg = str(e)
        err(f"heartlib source found at {cache} but deps missing: {msg}")
        err("Install the missing package(s) into the toolbox image before "
            "running with --real. See heartlib/pyproject.toml for the full list.")
        return False, f"deps_missing: {msg}"


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
def smoke_test(ckpt_root, allow_clone):
    log("=== SMOKE TEST (no GPU inference) ===")

    # 1. Core deps.
    core_ok = True
    for mod in ("torch", "torchaudio", "transformers", "soundfile", "numpy"):
        try:
            m = __import__(mod)
            log(f"  import {mod:12s} OK  ({getattr(m, '__version__', '?')})")
        except Exception as e:
            err(f"  import {mod} FAILED: {e}")
            core_ok = False

    # 2. Weight layout.
    assert_weights(ckpt_root)

    # 3. torch / ROCm sanity (NO allocation, NO kernel launch).
    try:
        import torch
        log(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")
            log(f"  HSA_OVERRIDE_GFX_VERSION = {os.environ.get('HSA_OVERRIDE_GFX_VERSION', '<unset>')}")
    except Exception as e:
        err(f"  torch probe failed: {e}")
        core_ok = False

    # 4. heartlib. Missing deep deps are treated as a WARNING in smoke mode:
    # the source + weights are on disk, but the toolbox image needs extra
    # pip installs before --real will work. We still return 0 so the harness
    # can proceed and the PR smoke log clearly lists what's left to install.
    hl_ok, hl_detail = ensure_heartlib(install_if_missing=allow_clone)
    log(f"  heartlib status: {'OK' if hl_ok else 'WARN'} ({hl_detail})")

    if core_ok:
        if hl_ok:
            log("Smoke OK — environment is ready for --real inference.")
        else:
            log("Smoke OK (with warning): heartlib deps must be installed "
                "before --real. Core libs + weights are in place.")
        return 0
    log("Smoke FAILED — see errors above.")
    return 1


# ----------------------------------------------------------------------------
# Real inference
# ----------------------------------------------------------------------------
def run_inference(args, ckpt_root):
    """
    Delegate to heartlib's example script. heartlib exposes
    `examples/run_music_generation.py` which takes --model_path/--version and
    writes to --save_path. It natively produces mp3; we convert to WAV after.
    """
    ok, detail = ensure_heartlib(install_if_missing=True)
    if not ok:
        err(f"heartlib unavailable ({detail}) — cannot run real inference.")
        return 3

    # Write a tiny lyrics and tags file derived from the prompt. The prompt acts
    # as the "tags" (style description); lyrics are left empty / hummed, which
    # heartlib tolerates for instrumental output.
    scratch = Path("/tmp/heartmula_run")
    scratch.mkdir(parents=True, exist_ok=True)
    lyrics_path = scratch / "lyrics.txt"
    tags_path = scratch / "tags.txt"
    lyrics_path.write_text(args.lyrics or "[instrumental]\n")
    tags_path.write_text(args.prompt)

    # heartlib writes mp3; we route to an intermediate path then transcode.
    mp3_out = scratch / "output.mp3"

    # Locate the run_music_generation example script.
    hl_root = Path(HEARTLIB_CACHE)
    example = hl_root / "examples" / "run_music_generation.py"
    if not example.exists():
        err(f"heartlib example not found at {example}")
        return 4

    cmd = [
        sys.executable, str(example),
        "--model_path", str(ckpt_root),
        "--version", "3B",
        "--lyrics", str(lyrics_path),
        "--tags", str(tags_path),
        "--save_path", str(mp3_out),
        "--max_audio_length_ms", str(int(args.duration * 1000)),
    ]
    if args.lazy_load:
        cmd += ["--lazy_load", "true"]

    log("Invoking heartlib: " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    log(f"heartlib exited rc={rc} in {dt:.1f}s")
    if rc != 0 or not mp3_out.exists():
        err("inference failed or no output file produced")
        return rc or 5

    # Transcode to WAV at the --out path.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Transcoding {mp3_out} -> {out_path}")
    rc = subprocess.call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mp3_out),
        "-ac", "2", "-ar", "48000",
        str(out_path),
    ])
    if rc != 0:
        err("ffmpeg transcode failed")
        return 6

    log(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="HeartMuLa music generator")
    p.add_argument("--prompt", required=True,
                   help="Style/tags prompt (e.g. 'cinematic orchestral rise').")
    p.add_argument("--lyrics", default="",
                   help="Optional lyrics text; empty -> instrumental.")
    p.add_argument("--duration", type=float, default=6.0,
                   help="Target audio duration in seconds.")
    p.add_argument("--out", required=True,
                   help="Output WAV path (absolute, inside container).")
    p.add_argument("--ckpt-root", default=None,
                   help="Override checkpoint root (auto-detected otherwise).")
    p.add_argument("--lazy-load", action="store_true",
                   help="Enable heartlib --lazy_load for tight VRAM budgets.")
    p.add_argument("--real", action="store_true",
                   help="Actually run GPU inference. Without this the script "
                        "only performs the smoke test (safe while fleet runs).")
    p.add_argument("--no-gpu-test", action="store_true",
                   help="Force smoke-test mode even if --real is passed.")
    p.add_argument("--no-clone", action="store_true",
                   help="Fail instead of cloning heartlib if missing.")
    return p.parse_args()


def main():
    args = parse_args()
    log(f"prompt={args.prompt!r} duration={args.duration}s out={args.out}")

    ckpt_root = find_ckpt_root(args.ckpt_root)

    if args.no_gpu_test or not args.real:
        rc = smoke_test(ckpt_root, allow_clone=not args.no_clone)
        sys.exit(rc)

    if ckpt_root is None:
        assert_weights(ckpt_root)  # exits 2

    sys.exit(run_inference(args, ckpt_root))


if __name__ == "__main__":
    main()
