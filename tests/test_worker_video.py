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


def test_video_worker_role_and_can_claim():
    w = video_mod.VideoWorker()
    assert w.role == "video"
    assert w.can_claim(_mk_item()) is True


def test_video_worker_picks_launcher_per_model():
    assert video_mod._launcher_for("wan2.2") == "/opt/wan_launcher.py"
    assert video_mod._launcher_for("wan2.5") == "/opt/wan_launcher.py"
    assert video_mod._launcher_for("ltx-2.3") == "/opt/ltx_launcher.py"


def test_video_worker_run_stage_success(tmp_path, monkeypatch):
    item = _mk_item(out_dir=str(tmp_path), v_idx=4)

    async def fake_run(cmd):
        out_path = cmd[cmd.index("--out") + 1]
        with open(out_path, "wb") as f:
            f.write(b"MP4")
        return 0

    monkeypatch.setattr(video_mod, "_run", fake_run)
    monkeypatch.setattr(video_mod, "acquire_gpu", None)

    w = video_mod.VideoWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["asset"].endswith("v4_video.mp4")


def test_video_worker_empty_prompt():
    item = _mk_item()
    item["stages"]["concept"]["output"] = ""
    w = video_mod.VideoWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is False
