"""Unit tests for slopfinity.wan_cli argv mapping (no GPU)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slopfinity.wan_cli import wan_launcher_argv, wan_paths  # noqa: E402


def test_wan_launcher_argv_uses_save_file_not_out():
    argv = wan_launcher_argv(
        "prompt here",
        "/workspace/comfy-outputs/seed.png",
        "/workspace/comfy-outputs/out.mp4",
        model="wan2.2",
    )
    assert argv[0:2] == ["python3", "/opt/wan_launcher.py"]
    assert "--save_file" in argv
    assert "--out" not in argv
    assert "--model" not in argv  # not a generate.py flag
    assert "--task" in argv
    assert argv[argv.index("--task") + 1] in (
        "i2v-A14B", "t2v-A14B", "ti2v-5B",
    )
    assert "--ckpt_dir" in argv
    assert "--prompt" in argv
    assert "prompt here" in argv
    # image only when task is not pure t2v
    task = argv[argv.index("--task") + 1]
    if task != "t2v-A14B":
        assert "--image" in argv


def test_wan_paths_env_override(monkeypatch):
    monkeypatch.setenv("WAN_CKPT_DIR", "/data/Wan2.2-I2V-A14B")
    monkeypatch.setenv("WAN_TASK", "t2v-A14B")
    monkeypatch.setenv("WAN_FRAME_NUM", "9")
    cfg = wan_paths("wan2.2")
    assert cfg["ckpt"] == "/data/Wan2.2-I2V-A14B"
    assert cfg["task"] == "t2v-A14B"
    assert cfg["frame_num"] == "9"


def test_is_complete_ckpt_rejects_comfy_fp8_t5(tmp_path):
    from slopfinity.wan_cli import _is_complete_ckpt

    d = tmp_path / "fake"
    d.mkdir()
    # ~6.7 GB Comfy-style misnamed T5 would fail the size floor; use tiny file
    t5 = d / "models_t5_umt5-xxl-enc-bf16.pth"
    t5.write_bytes(b"not-real")
    assert _is_complete_ckpt(str(d), "ti2v-5B") is False


def test_video_worker_docker_cmd_for_wan_includes_task():
    from slopfinity.workers import video as video_mod
    cmd = video_mod._docker_cmd(
        "wan2.2", "p", "/tmp/seed.png", "/tmp/out.mp4",
    )
    assert "/opt/wan_launcher.py" in cmd
    assert "--task" in cmd
    assert "--save_file" in cmd
    assert "--out" not in cmd
