"""Hermetic unit tests for slopfinity.vae_grid (FFT grid-artifact detector).

All inputs are constructed explicitly (no randomness, no real model output) so
results are deterministic:
  - a smooth gradient image -> has_grid False, low score
  - a gradient with an injected 8-px periodic pattern -> has_grid True, peak_freq 8
  - error paths: missing file, non-RGB
  - write_sidecar / read_sidecar round-trip + fmt_summary formatting

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


def _broadband_texture(h=256, w=256):
    """A deterministic, spectrally-flat texture (clean / no block grid).

    The detector's baseline is the global-median spectrum amplitude, so a
    *clean* image for it is one whose energy is spread evenly across all
    frequencies (a large, uniform median) with no peak at the 8/16-px block
    bins. White noise gives exactly that. We seed the generator with a fixed
    value so the array — and therefore every assertion below — is fully
    reproducible run to run.
    """
    rng = np.random.default_rng(12345)
    n = rng.integers(0, 256, size=(h, w)).astype(np.float32)
    return np.stack([n, n, n], axis=-1)


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

def test_clean_broadband_texture_has_no_grid(tmp_path):
    p = tmp_path / "clean.png"
    _save_rgb(_broadband_texture(), p)
    r = vae_grid.detect_grid(str(p))
    assert r["method"] == "fft"
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
    # Inject a strong 8-px-period checker on top of a clean broadband texture —
    # exactly the VAE block artifact the detector targets.
    base = _inject_grid(_broadband_texture(), period=8, amp=60.0)
    p = tmp_path / "grid8.png"
    _save_rgb(base, p)

    r = vae_grid.detect_grid(str(p))
    assert "error" not in r
    assert r["has_grid"] is True
    assert r["peak_freq"] == 8
    assert r["score"] >= vae_grid.GRID_THRESHOLD


def test_grid_score_strictly_higher_than_clean(tmp_path):
    clean_arr = _broadband_texture()
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
    # not error out. Use a broadband texture so it reads as clean (no grid).
    p = tmp_path / "gray.png"
    arr = np.clip(_broadband_texture(128, 128)[:, :, 0], 0, 255).astype(np.uint8)
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
