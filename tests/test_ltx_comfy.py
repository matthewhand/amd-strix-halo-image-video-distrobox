"""Unit tests for slopfinity.ltx_comfy (no GPU / no live Comfy required)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slopfinity import ltx_comfy as ltx  # noqa: E402


def test_workflow_video_i2v_has_load_image_and_save():
    wf = ltx.workflow_video_i2v(
        "seed.png", "a cat walks",
        width=640, height=384, frames=17, prefix="unit_vid",
        include_audio=False,
    )
    assert isinstance(wf, dict)
    types = {n["class_type"] for n in wf.values()}
    assert "LoadImage" in types
    assert "LTXVImgToVideoConditionOnly" in types
    assert "CFGGuider" in types  # not MultimodalGuider (needs AV pack)
    assert "SaveVideo" in types or "SaveImage" in types
    load = [n for n in wf.values() if n["class_type"] == "LoadImage"][0]
    assert load["inputs"]["image"] == "seed.png"


def test_workflow_image_single_frame():
    wf = ltx.workflow_image("tiny red cube", width=512, height=512, prefix="unit_img")
    types = {n["class_type"] for n in wf.values()}
    assert "EmptyLTXVLatentVideo" in types
    empty = [n for n in wf.values() if n["class_type"] == "EmptyLTXVLatentVideo"][0]
    assert empty["inputs"]["length"] == 1
    assert "SaveImage" in types


def test_workflow_upscale_uses_spatial_upsampler():
    wf = ltx.workflow_upscale_i2v_spatial(
        "seed.png", "detail", frames=17, prefix="unit_up",
    )
    types = {n["class_type"] for n in wf.values()}
    assert "LowVRAMLatentUpscaleModelLoader" in types or "LatentUpscaleModelLoader" in types
    assert "LTXVLatentUpsampler" in types
    assert "CFGGuider" in types
    assert "SaveVideo" in types
    loader = [
        n for n in wf.values()
        if n["class_type"] in ("LowVRAMLatentUpscaleModelLoader", "LatentUpscaleModelLoader")
    ][0]
    assert "spatial-upscaler" in loader["inputs"]["model_name"]


def test_generate_video_submits_and_copies(tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"
    seed = tmp_path / "seed.png"
    seed.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    fake_entry = {
        "status": {"status_str": "success"},
        "outputs": {
            "24": {"images": [{"filename": "fake_vid.mp4", "subfolder": "", "type": "output"}]},
        },
    }
    # Place a real file where _find_output_file will look
    odir = tmp_path / "comfy-outputs"
    odir.mkdir()
    (odir / "fake_vid.mp4").write_bytes(b"mp4data")

    monkeypatch.setattr(ltx, "comfy_output_dir", lambda: odir)
    monkeypatch.setattr(ltx, "comfy_input_dir", lambda: tmp_path)
    monkeypatch.setattr(ltx, "submit_prompt", lambda wf, **k: "pid-1")
    monkeypatch.setattr(ltx, "wait_history", lambda pid, **k: fake_entry)

    rc = ltx.generate_video("prompt", str(out), image_path=str(seed), frames=9)
    assert rc == 0
    assert out.is_file() and out.read_bytes() == b"mp4data"


def test_upscale_video_uses_seed_frame(tmp_path, monkeypatch):
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"fake-mp4")
    out = tmp_path / "out.mp4"
    odir = tmp_path / "comfy-outputs"
    odir.mkdir()
    (odir / "up.mp4").write_bytes(b"UP2")

    monkeypatch.setattr(ltx, "comfy_output_dir", lambda: odir)
    monkeypatch.setattr(ltx, "comfy_input_dir", lambda: tmp_path)
    monkeypatch.setattr(ltx, "_extract_seed_frame", lambda v, d: d.write_bytes(b"png") or True)
    monkeypatch.setattr(ltx, "submit_prompt", lambda wf, **k: "pid-2")
    monkeypatch.setattr(
        ltx, "wait_history",
        lambda pid, **k: {
            "status": {"status_str": "success"},
            "outputs": {"42": {"images": [{"filename": "up.mp4"}]}},
        },
    )

    rc = ltx.upscale_video(str(inp), str(out), frames=9)
    assert rc == 0
    assert out.read_bytes() == b"UP2"


def test_generate_image_failure_returns_nonzero(monkeypatch):
    monkeypatch.setattr(ltx, "submit_prompt", mock.Mock(side_effect=RuntimeError("down")))
    rc = ltx.generate_image("x", "/tmp/nope.png")
    assert rc != 0
