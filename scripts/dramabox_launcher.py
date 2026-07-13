#!/usr/bin/env python3
"""DramaBox (ResembleAI) expressive TTS CLI launcher.

Wraps DramaBox's warm TTSServer.generate_to_file for the multi-engine
qwen_tts_serve dispatcher (engine=dramabox).

Usage:
  python3 dramabox_launcher.py --text 'Hello.' --voice narrator-female --out /path/out.wav

Env:
  DRAMABOX_ROOT   — path to DramaBox checkout (default: /opt/DramaBox or
                    /mnt/downloads/dramabox-work/DramaBox)
  HF_HOME         — HF cache containing ResembleAI/Dramabox + gemma (default
                    /mnt/downloads/dramabox-hf)
  DRAMABOX_VOICES — dir of bundled ref WAVs (narrator-female.wav, …)
  HSA_OVERRIDE_GFX_VERSION — forced to 11.5.1 for gfx1151
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ["HSA_OVERRIDE_GFX_VERSION"] = os.environ.get(
    "QWEN_TTS_GFX_OVERRIDE", os.environ.get("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
)
# Avoid hf_transfer hard-fail when the package is absent.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

BUNDLED_VOICES = {
    "narrator-female",
    "narrator-male",
    "kid",
    "none",
}


def _resolve_root() -> str:
    for cand in (
        os.environ.get("DRAMABOX_ROOT"),
        "/opt/DramaBox",
        "/mnt/downloads/dramabox-work/DramaBox",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DramaBox"),
    ):
        if cand and os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "src")):
            return os.path.abspath(cand)
    raise FileNotFoundError(
        "DramaBox root not found. Set DRAMABOX_ROOT to the checkout "
        "(needs src/ + ltx2/)."
    )


def _resolve_weight_paths() -> dict:
    """Prefer local HF snapshots already on /mnt/downloads; fall back to download."""
    hf = (
        os.environ.get("DRAMABOX_HF_HOME")
        or "/mnt/downloads/dramabox-hf"
        or os.environ.get("HF_HOME")
    )
    # Explicit env overrides
    dit = os.environ.get("DRAMABOX_DIT")
    audio = os.environ.get("DRAMABOX_AUDIO_COMPONENTS")
    gemma = os.environ.get("GEMMA_DIR") or os.environ.get("DRAMABOX_GEMMA")

    snap_root = os.path.join(hf, "models--ResembleAI--Dramabox", "snapshots")
    if not dit or not audio:
        if os.path.isdir(snap_root):
            snaps = sorted(
                [
                    os.path.join(snap_root, d)
                    for d in os.listdir(snap_root)
                    if os.path.isdir(os.path.join(snap_root, d))
                ]
            )
            for s in reversed(snaps):
                d_cand = os.path.join(s, "dramabox-dit-v1.safetensors")
                a_cand = os.path.join(s, "dramabox-audio-components.safetensors")
                if os.path.isfile(d_cand) and os.path.isfile(a_cand):
                    dit = dit or d_cand
                    audio = audio or a_cand
                    break

    if not gemma:
        for rel in (
            "models--unsloth--gemma-3-12b-it-bnb-4bit/snapshots",
            "models--unsloth--gemma-3-12b-it/snapshots",
        ):
            base = os.path.join(hf, rel)
            if not os.path.isdir(base):
                continue
            for d in sorted(os.listdir(base)):
                p = os.path.join(base, d)
                if os.path.isfile(os.path.join(p, "config.json")):
                    gemma = p
                    break
            if gemma:
                break

    if dit and audio and gemma:
        return {
            "transformer": dit,
            "audio_components": audio,
            "gemma_root": gemma,
        }

    # Last resort: official downloader (needs network + disk).
    from model_downloader import get_all_paths  # type: ignore

    return get_all_paths(cache_dir=os.environ.get("DRAMABOX_CACHE") or hf)


def _resolve_voice_ref(voice: str, voices_dir: str) -> str | None:
    v = (voice or "none").strip()
    if not v or v.lower() in ("none", "default", "-"):
        return None
    if os.path.isfile(v):
        return os.path.abspath(v)
    # basename without extension
    base = os.path.splitext(os.path.basename(v))[0]
    for ext in (".wav", ".mp3", ".flac"):
        p = os.path.join(voices_dir, base + ext)
        if os.path.isfile(p):
            return p
    # also search DramaBox shipped assets/voices
    return None


def _wrap_prompt(text: str) -> str:
    """If the caller already used quote/stage grammar, pass through."""
    t = (text or "").strip()
    if not t:
        return t
    if '"' in t and any(k in t.lower() for k in ("speaks", "says", "whispers", "laugh")):
        return t
    # Plain text → mild expressive framing (works without voice clone).
    return f'A clear narrator speaks, "{t}"'


def synthesize(
    text: str,
    voice: str,
    out_path: str,
    *,
    cfg_scale: float = 2.5,
    stg_scale: float = 1.5,
    watermark: bool = False,
) -> None:
    root = _resolve_root()
    voices_dir = os.environ.get("DRAMABOX_VOICES") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dramabox_voices"
    )
    # Prefer host HF cache for ResembleAI + gemma
    if not os.environ.get("HF_HOME"):
        for cand in (
            "/mnt/downloads/dramabox-hf",
            "/mnt/downloads/cache/huggingface",
        ):
            if os.path.isdir(cand):
                os.environ["HF_HOME"] = cand
                break

    sys.path.insert(0, os.path.join(root, "ltx2"))
    sys.path.insert(0, os.path.join(root, "src"))

    print(f"[dramabox] root={root}", file=sys.stderr)
    print(f"[dramabox] HF_HOME={os.environ.get('HF_HOME')}", file=sys.stderr)
    print(f"[dramabox] voice={voice} out={out_path}", file=sys.stderr)

    from inference_server import TTSServer  # type: ignore

    paths = _resolve_weight_paths()
    print(
        f"[dramabox] paths={{ {', '.join(f'{k}={v}' for k,v in paths.items())} }}",
        file=sys.stderr,
    )

    # bnb_4bit=False: unsloth full gemma snapshot on this host is not pre-4bit,
    # and bitsandbytes is often missing/broken on ROCm.
    bnb = os.environ.get("DRAMABOX_BNB_4BIT", "0").strip() in ("1", "true", "yes")
    server = TTSServer(
        checkpoint=paths["transformer"],
        full_checkpoint=paths["audio_components"],
        gemma_root=paths["gemma_root"],
        device="cuda",
        dtype=os.environ.get("LTX_DTYPE", "bf16"),
        compile_model=False,
        bnb_4bit=bnb,
    )

    voice_ref = _resolve_voice_ref(voice, voices_dir)
    # Fall back to DramaBox shipped demo voices by alias
    if voice_ref is None and voice and voice.lower() not in ("none", "default", "-"):
        shipped = os.path.join(root, "assets", "voices")
        voice_ref = _resolve_voice_ref(voice, shipped)

    prompt = _wrap_prompt(text)
    print(f"[dramabox] voice_ref={voice_ref}", file=sys.stderr)
    print(f"[dramabox] prompt={prompt[:120]!r}", file=sys.stderr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Prefer generate() + soundfile/wave save — newer torchaudio routes
    # save() through torchcodec which is often missing in the ROCm image.
    waveform, sr = server.generate(
        prompt=prompt,
        voice_ref=voice_ref,
        cfg_scale=cfg_scale,
        stg_scale=stg_scale,
    )
    _save_wav(out_path, waveform, int(sr))
    if not os.path.isfile(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError(f"dramabox wrote empty/missing output: {out_path}")
    print(
        f"[dramabox] wrote {out_path} ({os.path.getsize(out_path)} bytes)",
        file=sys.stderr,
    )


def _save_wav(out_path: str, waveform, sr: int) -> None:
    """Save tensor/ndarray mono/stereo PCM16 without torchcodec."""
    import numpy as np

    try:
        import torch

        if isinstance(waveform, torch.Tensor):
            wav = waveform.detach().float().cpu().numpy()
        else:
            wav = np.asarray(waveform, dtype=np.float32)
    except Exception:
        wav = np.asarray(waveform, dtype=np.float32)

    if wav.ndim == 2:
        # (channels, samples) or (samples, channels)
        if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1]:
            wav = wav.mean(axis=0)
        else:
            wav = wav.mean(axis=-1)
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype("<i2")
    try:
        import soundfile as sf

        sf.write(out_path, pcm.astype(np.float32) / 32767.0, sr, subtype="PCM_16")
        return
    except Exception:
        pass
    import wave

    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DramaBox TTS launcher")
    ap.add_argument("--text", required=True)
    ap.add_argument("--voice", default="narrator-female")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg-scale", type=float, default=2.5)
    ap.add_argument("--stg-scale", type=float, default=1.5)
    ap.add_argument("--watermark", action="store_true")
    args = ap.parse_args(argv)
    try:
        synthesize(
            args.text,
            args.voice,
            args.out,
            cfg_scale=args.cfg_scale,
            stg_scale=args.stg_scale,
            watermark=args.watermark,
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[dramabox] FATAL: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
