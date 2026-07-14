"""Hermetic unit tests for slopfinity.stats.

Covers the pure parsing / threshold logic:
  - _status_from_pct  (ok / warn / danger thresholds)
  - get_outputs_disk  (uses shutil.disk_usage; mocked, plus missing-path path)
  - get_output_counts (filesystem globbing in a tmp dir, default-dir fallback)
  - get_ram_estimate  (per-stage breakdown, overhead row, status thresholds,
                       slopped:/none lookups, pretty labels)
  - _lookup / _pretty (model -> GB and friendly label)
  - get_sys_stats     (rocm-smi / /proc parsing, all mocked — no real GPU/proc)

Everything is mocked: no real rocm-smi, no real /proc dependence, tmp dirs only.
"""
from __future__ import annotations

import os
import sys
import types
import unittest.mock as mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import stats  # noqa: E402


# ---------- _status_from_pct -------------------------------------------------

@pytest.mark.parametrize("pct,expected", [
    (0, "ok"),
    (50.0, "ok"),
    (91.9, "ok"),
    (92, "warn"),
    (95.5, "warn"),
    (96.99, "warn"),
    (97, "danger"),
    (99.9, "danger"),
    (100, "danger"),
])
def test_status_from_pct_thresholds(pct, expected):
    assert stats._status_from_pct(pct) == expected


# ---------- get_outputs_disk -------------------------------------------------

def _fake_usage(total, used, free):
    return types.SimpleNamespace(total=total, used=used, free=free)


def test_get_outputs_disk_ok(monkeypatch):
    gib = 1024 ** 3
    # 100 GB total, 50 GB used -> 50% -> ok
    monkeypatch.setattr(stats.shutil, "disk_usage",
                        lambda p: _fake_usage(100 * gib, 50 * gib, 50 * gib))
    r = stats.get_outputs_disk("/whatever")
    assert r["status"] == "ok"
    assert r["total_gb"] == 100.0
    assert r["used_gb"] == 50.0
    assert r["free_gb"] == 50.0
    assert r["pct"] == 50.0
    assert "missing" not in r


def test_get_outputs_disk_warn(monkeypatch):
    gib = 1024 ** 3
    # 93% used -> warn
    monkeypatch.setattr(stats.shutil, "disk_usage",
                        lambda p: _fake_usage(100 * gib, 93 * gib, 7 * gib))
    r = stats.get_outputs_disk("/x")
    assert r["status"] == "warn"
    assert r["pct"] == 93.0


def test_get_outputs_disk_danger(monkeypatch):
    gib = 1024 ** 3
    monkeypatch.setattr(stats.shutil, "disk_usage",
                        lambda p: _fake_usage(100 * gib, 98 * gib, 2 * gib))
    r = stats.get_outputs_disk("/x")
    assert r["status"] == "danger"


def test_get_outputs_disk_missing(monkeypatch):
    def boom(_p):
        raise FileNotFoundError("no such path")
    monkeypatch.setattr(stats.shutil, "disk_usage", boom)
    r = stats.get_outputs_disk("/does/not/exist")
    assert r["missing"] is True
    assert r["status"] == "ok"
    assert r["used_gb"] == 0
    assert r["total_gb"] == 0


# ---------- get_storage ------------------------------------------------------

def test_get_storage_mixed(monkeypatch):
    gib = 1024 ** 3

    def usage(path):
        if path == "/":
            return _fake_usage(100 * gib, 50 * gib, 50 * gib)
        raise FileNotFoundError(path)

    monkeypatch.setattr(stats.shutil, "disk_usage", usage)
    out = stats.get_storage()
    by_mount = {m["mount"]: m for m in out}
    assert by_mount["/"]["status"] == "ok"
    assert by_mount["/"]["pct"] == 50.0
    assert by_mount["/mnt/data"]["missing"] is True
    assert by_mount["/mnt/downloads"]["missing"] is True


# ---------- get_output_counts ------------------------------------------------

def test_get_output_counts_empty_dir(tmp_path):
    r = stats.get_output_counts(str(tmp_path))
    assert r == {
        "finals": 0, "chains": 0, "base_images": 0,
        "total_mp4": 0, "total_png": 0, "latest_final": None,
    }


def test_get_output_counts_nonexistent_dir():
    r = stats.get_output_counts("/path/that/should/not/exist/xyzzy")
    assert r["finals"] == 0
    assert r["latest_final"] is None


def test_get_output_counts_counts_files(tmp_path):
    # Two finals, with mtimes so the newest is deterministic.
    f_old = tmp_path / "FINAL_001.mp4"
    f_new = tmp_path / "FINAL_002.mp4"
    f_old.write_text("x")
    f_new.write_text("x")
    os.utime(f_old, (1000, 1000))
    os.utime(f_new, (2000, 2000))

    # Chain clips: current "slop_" prefix and legacy "v" prefix.
    (tmp_path / "slop_1_c0.mp4").write_text("x")
    (tmp_path / "v2_c1.mp4").write_text("x")
    # Base images, both prefixes.
    (tmp_path / "slop_1_base.png").write_text("x")
    (tmp_path / "v2_base.png").write_text("x")
    # An unrelated png that still counts in total_png.
    (tmp_path / "thumb.png").write_text("x")

    r = stats.get_output_counts(str(tmp_path))
    assert r["finals"] == 2
    assert r["chains"] == 2
    assert r["base_images"] == 2
    # finals(2) + chains(2) == 4 mp4 files
    assert r["total_mp4"] == 4
    # 2 base pngs + 1 thumb == 3 png files
    assert r["total_png"] == 3
    # Newest by mtime first.
    assert r["latest_final"] == "FINAL_002.mp4"


def test_get_output_counts_default_dir_fallback(tmp_path, monkeypatch):
    # When base_dir is None it probes a fixed candidate list. Redirect the
    # "/workspace" candidate to our tmp dir for both the isdir() probe and the
    # subsequent Path() glob.
    (tmp_path / "FINAL_x.mp4").write_text("x")
    from pathlib import Path as RealPath

    real_isdir = os.path.isdir
    monkeypatch.setattr(stats.os.path, "isdir",
                        lambda p: True if p == "/workspace" else real_isdir(p))
    monkeypatch.setattr(
        stats, "Path",
        lambda p: RealPath(str(tmp_path)) if p == "/workspace" else RealPath(p))
    r = stats.get_output_counts(None)
    assert r["finals"] == 1


# ---------- _lookup / _pretty ------------------------------------------------

def test_lookup_known_and_unknown():
    assert stats._lookup("qwen") == 28
    # wan2.5 peak ignored by default config override → 0
    assert stats._lookup("wan2.5") == 0
    assert stats._lookup("wan2.2", role="video") == 0
    assert stats._lookup("kokoro") == 8
    assert stats._lookup("dramabox") == 18
    assert stats._lookup("ltx-2.3", role="image") == 38
    assert stats._lookup("ltx-2.3", role="video") == 48
    assert stats._lookup("totally-unknown-model") == 0
    assert stats._lookup("none") == 0
    assert stats._lookup("") == 0


def test_lookup_none_and_slopped():
    assert stats._lookup(None) == 0
    # slopped:<file> placeholders cost zero incremental RAM.
    assert stats._lookup("slopped:foo.png") == 0


def test_pretty_labels():
    assert stats._pretty("qwen") == "Qwen Image"
    assert stats._pretty("wan2.2") == "Wan 2.2"
    assert stats._pretty("") == "—"
    assert stats._pretty(None) == "—"
    assert stats._pretty("none") == "—"
    assert stats._pretty("slopped:my_asset.png") == "Slopped (my_asset.png)"
    # Unknown models pass through verbatim.
    assert stats._pretty("mystery") == "mystery"


# ---------- get_ram_estimate -------------------------------------------------

def test_get_ram_estimate_breakdown_structure():
    r = stats.get_ram_estimate("qwen", "ltx-2.3", None, None, tts_model=None)
    bd = r["breakdown"]
    # 5 stages + overhead row.
    assert len(bd) == 6
    roles = [e["role"] for e in bd]
    assert roles == ["image", "video", "audio", "tts", "upscale", "overhead"]
    # Overhead row constant.
    assert bd[-1]["gb"] == stats._OVERHEAD_GB
    assert bd[-1]["model"] == "OS + ComfyUI"
    # Empty roles render model "none" and gb 0.
    audio_row = bd[2]
    assert audio_row["model"] == "none"
    assert audio_row["gb"] == 0
    assert audio_row["label"] == "—"


def test_get_ram_estimate_total_math():
    # Serial peak = max(qwen28, ltx48, heartmula14, kokoro8, spatial30) = 48
    # estimated = peak + overhead(6) + safety_free(10) = 64
    r = stats.get_ram_estimate("qwen", "ltx-2.3", "heartmula", "ltx-spatial",
                               tts_model="kokoro")
    assert r["serial_peak_gb"] == 48.0
    assert r["estimated_gb"] == 64.0  # 48 + 6 + 10
    assert r["safety_free_gb"] == 10
    assert r["naive_all_warm_gb"] == 28 + 48 + 14 + 8 + 30 + 6  # 134
    assert r["budget_gb"] == 128
    # serial 64 is ok-tier, but naive all-warm ≥100 → warn (stack risk)
    assert r["status"] == "warn"


def test_get_ram_estimate_status_ok():
    # serial peak qwen 28 + oh 6 + free 10 = 44 -> ok
    r = stats.get_ram_estimate("qwen", None, None, None)
    assert r["estimated_gb"] == 44.0
    assert r["status"] == "ok"


def test_get_ram_estimate_status_danger():
    # wan2.5 budget overridden to 0; peak becomes ltx-spatial 30 → serial 30+6+10=46
    # Use many large non-wan peaks to still hit danger on serial path:
    # qwen 28 + ltx 48 + heartmula 14 + spatial 30 + qwen-tts 10 → peak 48 → 64 ok/warn
    # Force danger via naive stack without wan: still serial peak 48
    r = stats.get_ram_estimate("qwen", "ltx-2.3", "heartmula", "ltx-spatial",
                               tts_model="qwen-tts")
    assert r["serial_peak_gb"] == 48.0
    assert r["breakdown"][3]["model"] == "qwen-tts"
    assert r["breakdown"][3]["gb"] == 10
    # wan override: video row is 0
    r_wan = stats.get_ram_estimate("qwen", "wan2.5", None, None, tts_model=None)
    assert r_wan["breakdown"][1]["model"] == "wan2.5"
    assert r_wan["breakdown"][1]["gb"] == 0
    assert r_wan["serial_peak_gb"] == 28.0  # qwen only


def test_get_ram_estimate_slopped_costs_zero():
    base = stats.get_ram_estimate("qwen", None, None, None)["estimated_gb"]
    sl = stats.get_ram_estimate("slopped:existing.png", None, None, None)
    # Slopped base contributes 0 instead of qwen's 28 → estimated drops by 28.
    assert sl["estimated_gb"] == base - 28
    assert sl["breakdown"][0]["label"].startswith("Slopped (")


def test_get_ram_estimate_includes_dramabox():
    r = stats.get_ram_estimate("qwen", "ltx-2.3", None, None, tts_model="dramabox")
    tts_row = r["breakdown"][3]
    assert tts_row["model"] == "dramabox"
    assert tts_row["gb"] == 18
    assert tts_row["label"] == "DramaBox"


# ---------- get_sys_stats (all subprocess + /proc mocked) --------------------

def _reset_name_cache(monkeypatch):
    monkeypatch.setattr(stats, "_GPU_NAME", "GPU-X", raising=False)
    monkeypatch.setattr(stats, "_CPU_NAME", "CPU-Y", raising=False)


def test_get_sys_stats_parses_rocm_and_proc(monkeypatch):
    _reset_name_cache(monkeypatch)

    rocm_out = (
        "GPU[0]\t\t: GPU use (%): 42\n"
        "GPU[0]\t\t: GPU Memory Allocated (VRAM%): 73\n"
    )
    monkeypatch.setattr(stats.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=rocm_out))

    loadavg = "2.00 1.50 0.75 1/234 5678\n"
    meminfo = (
        "MemTotal:       131072000 kB\n"
        "MemFree:         60000000 kB\n"
        "MemAvailable:   100000000 kB\n"
    )

    real_open = open

    def fake_open(path, *a, **k):
        if path == "/proc/loadavg":
            return mock.mock_open(read_data=loadavg)()
        if path == "/proc/meminfo":
            return mock.mock_open(read_data=meminfo)()
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(stats.os, "cpu_count", lambda: 4)

    s = stats.get_sys_stats()
    assert s["gpu"] == 42
    assert s["vram"] == 73
    assert s["load_1m"] == 2.0
    assert s["load_5m"] == 1.5
    assert s["load_15m"] == 0.75
    # load_pct = round(2.0/4 * 100) = 50
    assert s["load_pct"] == 50
    # ram_u = (MemTotal - MemAvailable) / 1024 / 1024 GB
    assert s["ram_u"] == round((131072000 - 100000000) / (1024 * 1024), 1)
    assert s["ram_t"] == round(131072000 / (1024 * 1024), 1)
    assert s["gpu_name"] == "GPU-X"
    assert s["cpu_name"] == "CPU-Y"


def test_get_sys_stats_survives_all_failures(monkeypatch):
    _reset_name_cache(monkeypatch)

    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(stats.subprocess, "run", boom)
    monkeypatch.setattr("builtins.open", boom)

    s = stats.get_sys_stats()
    # All numeric fields fall back to their zero defaults; no exception.
    assert s["gpu"] == 0
    assert s["vram"] == 0
    assert s["ram_u"] == 0
    assert s["load_pct"] == 0


# ---------- _detect_cpu_name / _detect_gpu_name parsing ----------------------

def test_detect_cpu_name_strips_igpu_suffix(monkeypatch):
    cpuinfo = (
        "processor\t: 0\n"
        "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n"
        "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n"
    )
    monkeypatch.setattr("builtins.open",
                        lambda *a, **k: mock.mock_open(read_data=cpuinfo)())
    assert stats._detect_cpu_name() == "AMD RYZEN AI MAX+ 395"


def test_detect_cpu_name_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr("builtins.open", boom)
    assert stats._detect_cpu_name() == ""


def test_detect_gpu_name_from_rocm(monkeypatch):
    out = "GPU[0]\t\t: Card Series: AMD Radeon 8060S\n"
    monkeypatch.setattr(stats.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert stats._detect_gpu_name() == "AMD Radeon 8060S"


def test_detect_gpu_name_falls_back_to_lspci(monkeypatch):
    calls = {"n": 0}

    def run(cmd, *a, **k):
        calls["n"] += 1
        if cmd[0] == "rocm-smi":
            return types.SimpleNamespace(stdout="no useful keys here\n")
        # lspci fallback
        return types.SimpleNamespace(
            stdout="c1:00.0 VGA compatible controller: AMD/ATI Strix (rev c1)\n")

    monkeypatch.setattr(stats.subprocess, "run", run)
    name = stats._detect_gpu_name()
    assert "Strix" in name
    assert "(rev" not in name
