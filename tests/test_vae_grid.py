"""Hermetic unit tests for slopfinity.vae_grid (FFT grid-artifact detector).

All inputs are constructed explicitly (no randomness, no real model output) so
results are deterministic. These assertions describe the CORRECT detector
behaviour (local-baseline scoring), not the earlier over-triggering one:

  - a smooth gradient image            -> has_grid False (clean)
  - a single smooth sinusoid           -> has_grid False (clean)
  - a flat image                       -> has_grid False, score 0
  - a smooth image + injected 8-px grid  -> has_grid True, peak_freq 8
  - a smooth image + injected 16-px grid -> has_grid True, peak_freq 16
  - a gridded image scores strictly higher than the same image without a grid
  - error paths: missing file, non-RGB, corrupt file
  - write_sidecar / read_sidecar round-trip + fmt_summary formatting

A "clean" image here is a smooth, low-frequency-dominated one (a gradient or
sinusoid) — exactly the case the old global-median baseline false-positived on.
The fixed detector compares each candidate frequency to its LOCAL spectral
neighbourhood, so a smoothly-decaying spectrum scores ~1.0 (no spike) while a
sharp block grid towers over its neighbours and is flagged.

Requires numpy + PIL (both are project deps); skipped cleanly if absent.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import vae_grid  # noqa: E402

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


def _save_rgb(arr, path):
    """Save an HxWx3 float/int array as an 8-bit RGB PNG."""
    a = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(str(path))


def _smooth_gradient(h=256, w=256):
    """A deterministic, smooth left-to-right luminance gradient (clean).

    This is the canonical false-positive case for the OLD global-median
    baseline: nearly all of its spectral energy sits near DC, dragging the
    global median to ~0 so any faint periodic energy scored 4x+. The fixed
    local-baseline detector compares each candidate frequency to its own
    neighbouring bins — a smoothly-decaying ramp has neighbours nearly equal
    to the "peak", so the score is ~1.0 and has_grid is correctly False.

    Fully deterministic (no randomness): every assertion is reproducible.
    """
    xx = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    grad = xx / (w - 1) * 255.0
    return np.stack([grad, grad, grad], axis=-1)


def _smooth_sinusoid(h=256, w=256, period=120.0, amp=60.0):
    """A deterministic single smooth sinusoid (clean). Its energy sits at one
    low frequency well away from the 8/16-px block bins, with a smooth
    spectrum elsewhere — so the local-baseline detector reads it as clean."""
    yy, xx = np.mgrid[0:h, 0:w]
    s = 128.0 + amp * np.sin(2.0 * np.pi * xx / period)
    return np.stack([s, s, s], axis=-1).astype(np.float32)


def _inject_grid(base, period=8, amp=60.0):
    """Add a sharp square-wave block grid of the given pixel period to green."""
    h, w = base.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = (((xx % period) < period // 2).astype(np.float32) +
               ((yy % period) < period // 2).astype(np.float32)) * amp
    out = base.copy()
    out[:, :, 1] = np.clip(out[:, :, 1] + pattern, 0, 255)
    return out


# ---------- clean image ------------------------------------------------------

def test_smooth_gradient_has_no_grid(tmp_path):
    # A plain smooth gradient is the classic false-positive for the old
    # global-median baseline; the fixed detector must read it as CLEAN.
    p = tmp_path / "clean.png"
    _save_rgb(_smooth_gradient(), p)
    r = vae_grid.detect_grid(str(p))
    assert r["method"] == "fft"
    assert "error" not in r
    assert r["has_grid"] is False
    assert r["peak_freq"] is None
    assert r["score"] < vae_grid.GRID_THRESHOLD


def test_smooth_sinusoid_has_no_grid(tmp_path):
    # A single smooth sinusoid (no block grid) must also read as CLEAN.
    p = tmp_path / "sinusoid.png"
    _save_rgb(_smooth_sinusoid(), p)
    r = vae_grid.detect_grid(str(p))
    assert "error" not in r
    assert r["has_grid"] is False
    assert r["peak_freq"] is None
    assert r["score"] < vae_grid.GRID_THRESHOLD


def test_flat_image_scores_zero(tmp_path):
    p = tmp_path / "flat.png"
    _save_rgb(np.full((128, 128, 3), 128.0), p)
    r = vae_grid.detect_grid(str(p))
    assert r["has_grid"] is False
    assert r["score"] == 0.0
    assert r["peak_freq"] is None


# ---------- injected grid ----------------------------------------------------

def test_injected_8px_grid_is_detected(tmp_path):
    # Inject a strong 8-px-period checker on top of a smooth gradient —
    # exactly the VAE block artifact the detector targets.
    base = _inject_grid(_smooth_gradient(), period=8, amp=60.0)
    p = tmp_path / "grid8.png"
    _save_rgb(base, p)

    r = vae_grid.detect_grid(str(p))
    assert "error" not in r
    assert r["has_grid"] is True
    assert r["peak_freq"] == 8
    assert r["score"] >= vae_grid.GRID_THRESHOLD


def test_injected_16px_grid_is_detected(tmp_path):
    # A 16-px block grid (upscaler / some video VAEs) must also be caught.
    base = _inject_grid(_smooth_gradient(), period=16, amp=60.0)
    p = tmp_path / "grid16.png"
    _save_rgb(base, p)

    r = vae_grid.detect_grid(str(p))
    assert "error" not in r
    assert r["has_grid"] is True
    assert r["peak_freq"] == 16
    assert r["score"] >= vae_grid.GRID_THRESHOLD


def test_grid_score_strictly_higher_than_clean(tmp_path):
    clean_arr = _smooth_gradient()
    clean = tmp_path / "c.png"
    _save_rgb(clean_arr, clean)
    gridded = tmp_path / "g.png"
    _save_rgb(_inject_grid(clean_arr, period=8, amp=50.0), gridded)

    c_score = vae_grid.detect_grid(str(clean))["score"]
    g_score = vae_grid.detect_grid(str(gridded))["score"]
    assert g_score > c_score


# ---------- error paths ------------------------------------------------------

def test_missing_file():
    r = vae_grid.detect_grid("/no/such/file.png")
    assert r["has_grid"] is False
    assert r["error"] == "file_not_found"
    assert r["score"] == 0.0


def test_empty_path():
    r = vae_grid.detect_grid("")
    assert r["error"] == "file_not_found"


def test_grayscale_image_still_handled(tmp_path):
    # A single-channel ("L") image is converted to RGB internally, so it must
    # not error out. Use a smooth gradient so it reads as clean (no grid).
    p = tmp_path / "gray.png"
    arr = np.clip(_smooth_gradient(128, 128)[:, :, 0], 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(p))
    r = vae_grid.detect_grid(str(p))
    assert "error" not in r
    assert r["has_grid"] is False


def test_unreadable_file_returns_runtime_error(tmp_path):
    # A .png that isn't actually an image -> PIL raises -> runtime error caught.
    p = tmp_path / "bogus.png"
    p.write_text("this is not a png")
    r = vae_grid.detect_grid(str(p))
    assert r["has_grid"] is False
    assert r["error"].startswith("runtime:")


# ---------- sidecar round-trip ----------------------------------------------

def test_write_and_read_sidecar(tmp_path):
    img = tmp_path / "img.png"
    img.write_text("x")
    result = {"has_grid": True, "score": 5.5, "peak_freq": 8, "method": "fft"}
    sidecar = vae_grid.write_sidecar(str(img), result)
    assert sidecar == str(img) + ".grid.json"
    assert os.path.exists(sidecar)
    with open(sidecar) as fh:
        assert json.load(fh) == result
    # read_sidecar returns the same dict.
    assert vae_grid.read_sidecar(str(img)) == result


def test_write_sidecar_empty_path():
    assert vae_grid.write_sidecar("", {}) is None


def test_read_sidecar_missing(tmp_path):
    assert vae_grid.read_sidecar(str(tmp_path / "nope.png")) is None


def test_read_sidecar_empty_path():
    assert vae_grid.read_sidecar("") is None


def test_read_sidecar_corrupt_json(tmp_path):
    img = tmp_path / "img.png"
    (tmp_path / "img.png.grid.json").write_text("{ not valid json")
    img.write_text("x")
    assert vae_grid.read_sidecar(str(img)) is None


# ---------- fmt_summary ------------------------------------------------------

def test_fmt_summary_clean():
    s = vae_grid.fmt_summary({"has_grid": False, "score": 1.2})
    assert "clean" in s
    assert "1.2" in s


def test_fmt_summary_grid():
    s = vae_grid.fmt_summary({"has_grid": True, "peak_freq": 8, "score": 9.0})
    assert "has_grid" in s
    assert "8" in s


def test_fmt_summary_error():
    s = vae_grid.fmt_summary({"error": "file_not_found"})
    assert "error=file_not_found" in s


def test_fmt_summary_invalid_type():
    assert vae_grid.fmt_summary(None) == "vae-grid: <invalid>"
    assert vae_grid.fmt_summary("nope") == "vae-grid: <invalid>"
