"""Tests for slopfinity.workers.video::VideoWorker."""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from slopfinity.workers import video as video_mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"video worker not importable: {exc}", allow_module_level=True)


def _mk_item(model="ltx-2.3", v_idx=2, out_dir=None):
    return {
        "v_idx": v_idx,
        "config_snapshot": {"video_model": model, "out_dir": out_dir or "/tmp"},
        "stages": {
            "concept": {"output": "a sunset", "status": "done"},
            "image": {"status": "done", "asset": "/tmp/v2_base.png"},
            "video": {"status": "pending"},
        },
    }


def test_video_worker_role():
    w = video_mod.VideoWorker()
    assert w.role == "video"


def test_video_worker_picks_launcher_per_model():
    assert video_mod._launcher_for("wan2.2") == "/opt/wan_launcher.py"
    assert video_mod._launcher_for("wan2.5") == "/opt/wan_launcher.py"
    assert video_mod._launcher_for("ltx-2.3") == "/opt/ltx_launcher.py"


def test_is_ltx():
    assert video_mod._is_ltx("ltx-2.3")
    assert video_mod._is_ltx("LTX-2.3")
    assert not video_mod._is_ltx("wan2.2")


def test_video_worker_ltx_comfy_path(tmp_path, monkeypatch):
    item = _mk_item(out_dir=str(tmp_path), v_idx=4)
    # Seed image must exist for stage_input only if comfy path copies it —
    # we mock generate_video entirely.
    seed = tmp_path / "seed.png"
    seed.write_bytes(b"png")
    item["stages"]["image"]["asset"] = str(seed)

    async def fake_comfy(self, prompt, in_img, out_path):
        with open(out_path, "wb") as f:
            f.write(b"MP4")
        return 0

    monkeypatch.setattr(video_mod, "ltx_comfy", object())  # truthy
    monkeypatch.setattr(video_mod.VideoWorker, "_run_ltx_comfy", fake_comfy)
    monkeypatch.setattr(video_mod, "acquire_gpu", None)
    monkeypatch.setenv("SLOPFINITY_VIDEO_MODE", "http")

    w = video_mod.VideoWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["backend"] == "comfy"
    assert result["asset"].endswith("v4_video.mp4")


def test_video_worker_docker_fallback_for_wan(tmp_path, monkeypatch):
    item = _mk_item(model="wan2.2", out_dir=str(tmp_path), v_idx=7)

    async def fake_run(cmd):
        # --out may be absolute host path
        if "--out" in cmd:
            out_path = cmd[cmd.index("--out") + 1]
        else:
            out_path = str(tmp_path / "v7_video.mp4")
        with open(out_path, "wb") as f:
            f.write(b"WAN")
        return 0

    monkeypatch.setattr(video_mod, "_run", fake_run)
    monkeypatch.setattr(video_mod, "acquire_gpu", None)

    w = video_mod.VideoWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["backend"] == "docker"


def test_video_worker_empty_prompt():
    item = _mk_item()
    item["stages"]["concept"]["output"] = ""
    w = video_mod.VideoWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is False
