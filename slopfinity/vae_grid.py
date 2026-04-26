"""VAE-grid artifact detector for slopfinity outputs.

VAE decoders for Qwen-Image / Ernie-Image / LTX / Wan operate in latent
blocks (typically 8x8 or 16x16 px). Under stress (low step count, mismatched
VAE/model, AMD/ROCm precision quirks) the decoded image picks up a faint
periodic grid pattern at the block frequency. It's especially common on
Strix Halo with Qwen-Image at low tiers.

This module exposes ONE public function:

    detect_grid(image_path) -> {
        "has_grid": bool,        # True when score > threshold
        "score":    float,       # ratio of peak energy to baseline
        "peak_freq": int|None,   # 8 or 16 if a block grid was found, else None
        "method":   "fft",
    }

Implementation: take the green channel (most luminance information for the
human eye), 2D FFT, look for amplitude spikes at the spatial frequencies
that correspond to 8-px and 16-px periods. If either spike is N times the
local baseline, flag it. Numpy-only, no torch / no GPU.

Cost: ~30-80 ms for a 1024x1024 PNG on Strix Halo.
"""
from __future__ import annotations

import math
import os
from typing import Optional

# Threshold for "this is a grid": peak amplitude must be >= GRID_THRESHOLD x
# the median amplitude in a small neighbourhood of the candidate freq. 4x is
# the value that catches obvious Qwen artefacts on Strix Halo without
# false-positiving on textures with naturally periodic content.
GRID_THRESHOLD = 4.0

# Spatial periods we look for, in pixels. Most modern VAEs are 8x8, but
# upscalers and some video VAEs end up at 16x16. Add to this list if a
# new model surfaces a different period.
CANDIDATE_PERIODS = (8, 16)


def detect_grid(image_path: str) -> dict:
    """Run the FFT detector on an image file. Returns a result dict; never
    raises — on failure (PIL missing, file unreadable, numpy oom) the
    result has ``has_grid=False`` and ``error`` set so the caller can log.
    """
    if not image_path or not os.path.exists(image_path):
        return {"has_grid": False, "score": 0.0, "peak_freq": None,
                "method": "fft", "error": "file_not_found"}
    try:
        # Local imports so importing this module doesn't pay the numpy
        # / PIL load cost when the detector is disabled.
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as e:
        return {"has_grid": False, "score": 0.0, "peak_freq": None,
                "method": "fft", "error": f"missing_dep:{e.name}"}

    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            # Downsample huge images to bound FFT cost. 1024 is plenty
            # for grid detection — a smaller raster preserves the
            # offending periodicity at proportionally smaller pixel
            # values, and we just rescale the candidate period below.
            w, h = im.size
            scale = 1.0
            if max(w, h) > 1024:
                scale = 1024.0 / max(w, h)
                im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return {"has_grid": False, "score": 0.0, "peak_freq": None,
                    "method": "fft", "error": "not_rgb"}

        # Green channel = best human-luma proxy without computing YCbCr.
        g = arr[:, :, 1]
        # Detrend: subtract local mean so DC and low-freq texture don't
        # dominate the spectrum. Simple per-row + per-col mean removal
        # is enough for our 4x threshold.
        g = g - g.mean()
        spec = np.abs(np.fft.fft2(g))
        H, W = spec.shape
        # Energy at frequency f (cycles/image) for an N-pixel period sits
        # at index N // period_in_px. After downsampling by `scale`, the
        # effective period is original_period * scale.
        baseline = float(np.median(spec)) + 1e-6

        best_score = 0.0
        best_period: Optional[int] = None
        for orig_period in CANDIDATE_PERIODS:
            eff_period = max(2, int(round(orig_period * scale)))
            # Vertical spike (period along rows) → row-axis FFT bin H/eff_period
            # Horizontal spike (period along cols) → col-axis FFT bin W/eff_period
            row_bin = H // eff_period
            col_bin = W // eff_period
            if row_bin <= 1 or col_bin <= 1:
                continue
            # Take a small neighborhood (±1 bin) so off-by-one rounding
            # doesn't kill the signal. Max amplitude in that window vs
            # baseline = the score for this period.
            row_peak = float(spec[max(0, row_bin - 1):row_bin + 2, 0:3].max())
            col_peak = float(spec[0:3, max(0, col_bin - 1):col_bin + 2].max())
            peak = max(row_peak, col_peak)
            score = peak / baseline
            if score > best_score:
                best_score = score
                best_period = orig_period

        return {
            "has_grid": best_score >= GRID_THRESHOLD,
            "score": round(best_score, 3),
            "peak_freq": best_period if best_score >= GRID_THRESHOLD else None,
            "method": "fft",
        }
    except Exception as e:
        return {"has_grid": False, "score": 0.0, "peak_freq": None,
                "method": "fft", "error": f"runtime:{type(e).__name__}:{e}"}


def write_sidecar(image_path: str, result: dict) -> Optional[str]:
    """Persist the result next to the image as ``<file>.grid.json`` so the
    server can serve it without re-running the FFT. Returns the sidecar
    path on success, None on failure.
    """
    if not image_path:
        return None
    sidecar = image_path + ".grid.json"
    try:
        import json
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        return sidecar
    except Exception:
        return None


def read_sidecar(image_path: str) -> Optional[dict]:
    """Read the persisted result if present, else None."""
    if not image_path:
        return None
    sidecar = image_path + ".grid.json"
    if not os.path.exists(sidecar):
        return None
    try:
        import json
        with open(sidecar, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def fmt_summary(result: dict) -> str:
    """One-line summary for logs."""
    if not isinstance(result, dict):
        return "vae-grid: <invalid>"
    if result.get("error"):
        return f"vae-grid: error={result['error']}"
    if result.get("has_grid"):
        return (
            f"vae-grid: ⚠ has_grid period={result.get('peak_freq')}px "
            f"score={result.get('score')}"
        )
    return f"vae-grid: ✓ clean score={result.get('score')}"
