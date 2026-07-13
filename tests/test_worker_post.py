"""Tests for slopfinity.workers.post::PostWorker (ltx-spatial upscaler)."""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from slopfinity.workers import post as post_mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"post worker not importable: {exc}", allow_module_level=True)


def _mk_item(in_path="/tmp/v1_video.mp4", v_idx=1, out_dir=None):
    return {
        "v_idx": v_idx,
        "config_snapshot": {"out_dir": out_dir or "/tmp"},
        "stages": {
            "video": {"status": "done", "asset": in_path},
            "post": {"status": "pending"},
        },
    }


def test_post_worker_role():
    w = post_mod.PostWorker()
    assert w.role == "post"


def test_post_worker_comfy_path(tmp_path, monkeypatch):
    in_path = tmp_path / "v5_video.mp4"
    in_path.write_bytes(b"MP4_IN")
    item = _mk_item(in_path=str(in_path), out_dir=str(tmp_path), v_idx=5)

    async def fake_comfy(self, in_p, out_p, prompt):
        with open(out_p, "wb") as f:
            f.write(b"MP4_UP")
        return 0

    monkeypatch.setattr(post_mod, "ltx_comfy", object())
    monkeypatch.setattr(post_mod.PostWorker, "_run_comfy", fake_comfy)
    monkeypatch.setattr(post_mod, "acquire_gpu", None)
    monkeypatch.setenv("SLOPFINITY_UPSCALE_MODE", "http")

    w = post_mod.PostWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["backend"] == "comfy"
    assert result["asset"].endswith("v5_upscaled.mp4")


def test_post_worker_docker_fallback(tmp_path, monkeypatch):
    in_path = tmp_path / "v1_video.mp4"
    in_path.write_bytes(b"MP4_IN")
    item = _mk_item(in_path=str(in_path), out_dir=str(tmp_path), v_idx=1)

    async def fake_run(cmd):
        out_path = cmd[cmd.index("--out") + 1]
        with open(out_path, "wb") as f:
            f.write(b"MP4_UP")
        return 0

    monkeypatch.setattr(post_mod, "_run", fake_run)
    monkeypatch.setattr(post_mod, "acquire_gpu", None)
    monkeypatch.setenv("SLOPFINITY_UPSCALE_MODE", "docker")

    w = post_mod.PostWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["backend"] == "docker"


def test_post_worker_missing_input():
    item = _mk_item()
    item["stages"]["video"]["asset"] = ""
    w = post_mod.PostWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is False


def test_ltx_launcher_script_exists_and_modes():
    path = os.path.join(ROOT, "scripts", "ltx_launcher.py")
    assert os.path.isfile(path)
    text = open(path).read()
    assert "--mode" in text
    assert "image" in text and "video" in text and "upscale" in text
