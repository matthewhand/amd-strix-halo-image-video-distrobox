"""Tests for slopfinity.workers.image::ImageWorker."""
from __future__ import annotations

import asyncio
import os
import sys
from unittest import mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from slopfinity.workers import image as image_mod
except Exception as exc:  # pragma: no cover
    pytest.skip(f"image worker not importable: {exc}", allow_module_level=True)


def _mk_item(prompt="a cat", tier="low", v_idx=3, out_dir=None):
    return {
        "v_idx": v_idx,
        "config_snapshot": {"tier": tier, "out_dir": out_dir or "/tmp"},
        "stages": {
            "concept": {"output": prompt, "status": "done"},
            "image": {"status": "pending"},
        },
    }


def test_image_worker_role_and_can_claim():
    w = image_mod.ImageWorker()
    assert w.role == "image"
    assert w.can_claim(_mk_item()) is True
    done = _mk_item()
    done["stages"]["image"]["status"] = "done"
    assert w.can_claim(done) is False


def test_image_worker_resolves_steps_by_tier():
    w = image_mod.ImageWorker()
    assert w._resolve_steps(_mk_item(tier="low")) == 8
    assert w._resolve_steps(_mk_item(tier="medium")) == 16
    assert w._resolve_steps(_mk_item(tier="high")) == 28
    assert w._resolve_steps(_mk_item(tier="auto")) == 8


def test_image_worker_run_stage_success(tmp_path, monkeypatch):
    out_dir = str(tmp_path)
    item = _mk_item(out_dir=out_dir, v_idx=7)

    async def fake_run(cmd):
        # Simulate the docker call producing the expected PNG.
        out_path = cmd[-1]
        with open(out_path, "wb") as f:
            f.write(b"PNG_BYTES")
        return 0

    monkeypatch.setattr(image_mod, "_run", fake_run)
    monkeypatch.setattr(image_mod, "acquire_gpu", None)

    w = image_mod.ImageWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is True
    assert result["asset"].endswith("v7_base.png")
    assert os.path.exists(result["asset"])


def test_image_worker_empty_prompt():
    item = _mk_item(prompt="")
    w = image_mod.ImageWorker()
    result = asyncio.run(w.run_stage(item))
    assert result["ok"] is False
    assert "prompt" in (result.get("error") or "")


def test_image_worker_docker_cmd_shape():
    cmd = image_mod._docker_cmd("hello", 16, "/tmp/out.png")
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "/opt/qwen_launcher.py" in cmd
    assert "--steps" in cmd
    assert "16" in cmd
    assert cmd[-1] == "/tmp/out.png"
