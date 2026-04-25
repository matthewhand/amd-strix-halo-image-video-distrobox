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


def test_post_worker_role_and_can_claim():
    w = post_mod.PostWorker()
    assert w.role == "post"
    assert w.can_claim(_mk_item()) is True


def test_post_worker_run_stage_success(tmp_path, monkeypatch):
    in_path = tmp_path / "v5_video.mp4"
    in_path.write_bytes(b"MP4_IN")
    item = _mk_item(in_path=str(in_path), out_dir=str(tmp_path), v_idx=5)

    async def fake_run(cmd):
        out_path = cmd[cmd.index("--out") + 1]
        with open(out_path, "wb") as f:
            f.write(b"MP4_UP")
        return 0

    monkeypatch.setattr(post_mod, "_run", fake_run)
    monkeypatch.setattr(post_mod, "acquire_gpu", None)

    w = post_mod.PostWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["asset"].endswith("v5_upscaled.mp4")


def test_post_worker_missing_input():
    item = _mk_item()
    item["stages"]["video"]["asset"] = ""
    w = post_mod.PostWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is False
