"""Structural + call-path proof that image/ltx-2.3 uses shipped run_image_ltx."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_workers_fleet():
    """Load slopfinity/workers.py file (not workers/ package)."""
    path = ROOT / "slopfinity" / "workers.py"
    spec = importlib.util.spec_from_file_location("slopfinity.workers_fleet", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "slopfinity"
    mod.__name__ = "slopfinity.workers_fleet"
    sys.modules["slopfinity.workers_fleet"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_image_ltx_exists_and_invokes_ltx_launcher_mode_image():
    mod = _load_workers_fleet()
    assert hasattr(mod, "run_image_ltx")

    captured = {}

    async def fake_run(cmd):
        captured["cmd"] = cmd
        return 2  # launcher missing is fine — we assert argv shape

    # Avoid real GPU lock / docker
    class _NullCM:
        async def __aenter__(self):
            return None
        async def __aexit__(self, *a):
            return False

    with mock.patch.object(mod, "_run", fake_run), \
         mock.patch.object(mod, "acquire_gpu", lambda *a, **k: _NullCM()):
        rc = asyncio.run(mod.run_image_ltx("a prompt", "/workspace/out.png"))
    assert rc == 2
    cmd = captured["cmd"]
    assert "docker" in cmd[0]
    assert "/opt/ltx_launcher.py" in cmd
    assert "--mode" in cmd
    assert "image" in cmd
    assert "--prompt" in cmd
    assert "a prompt" in cmd
    assert "--out" in cmd
    assert "/workspace/out.png" in cmd
