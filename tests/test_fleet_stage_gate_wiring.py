"""Prove run_fleet heavy stages go through stage_gate (anti-OOM wiring)."""
from __future__ import annotations

import fcntl
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_public_stage_entrypoints_are_gated_wrappers():
    import run_fleet as rf

    assert callable(rf.run_image_gen)
    assert callable(rf._run_image_gen_ungated)
    assert rf.run_image_gen is not rf._run_image_gen_ungated

    assert callable(rf.generate_video_ltx)
    assert callable(rf._generate_video_ltx_ungated)
    assert rf.generate_video_ltx is not rf._generate_video_ltx_ungated

    assert callable(rf.generate_video_ltx_flf2v)
    assert callable(rf._generate_video_ltx_flf2v_ungated)

    assert callable(rf.generate_video_ltx_continuation)
    assert callable(rf._generate_video_ltx_continuation_ungated)

    assert callable(rf.tts_wav)
    assert callable(rf._tts_wav_ungated)
    assert rf.tts_wav is not rf._tts_wav_ungated

    assert callable(rf.heartmula_wav)
    assert callable(rf.generate_base_image_ltx23)
    assert callable(rf._generate_base_image_ltx23_ungated)


def test_gated_calls_stage_gate_with_role_model():
    import run_fleet as rf

    cm = mock.MagicMock()
    cm.__enter__ = mock.Mock(return_value={"ok": True})
    cm.__exit__ = mock.Mock(return_value=False)
    mock_gate = mock.Mock(return_value=cm)

    with mock.patch.object(rf, "_stage_gate", mock_gate):
        out = rf._gated("video", "ltx-2.3", lambda: 42, keep_after=False)
    assert out == 42
    mock_gate.assert_called_once()
    args, kwargs = mock_gate.call_args
    assert args[0] == "video"
    assert args[1] == "ltx-2.3"
    assert kwargs.get("keep_after") is False


def test_gated_refuses_on_insufficient_memory():
    import run_fleet as rf
    from slopfinity.stage_gate import InsufficientMemoryError

    def boom(*a, **k):
        raise InsufficientMemoryError("nope")

    with mock.patch.object(rf, "_stage_gate", side_effect=boom):
        with pytest.raises(InsufficientMemoryError):
            rf._gated("video", "ltx-2.3", lambda: True)


def test_run_image_gen_routes_through_gated(monkeypatch):
    import run_fleet as rf

    calls = []

    def fake_gated(role, model, fn, keep_after=False):
        calls.append((role, model, keep_after))
        return fn()

    monkeypatch.setattr(rf, "_gated", fake_gated)
    monkeypatch.setattr(rf, "_run_image_gen_ungated", lambda *a, **k: "IMG_OK")
    assert rf.run_image_gen("qwen", "hi", "/tmp/x.png") == "IMG_OK"
    assert calls == [("image", "qwen", False)]


def test_generate_video_ltx_routes_through_gated(monkeypatch):
    import run_fleet as rf

    calls = []

    def fake_gated(role, model, fn, keep_after=False):
        calls.append((role, model))
        return fn()

    monkeypatch.setattr(rf, "_gated", fake_gated)
    monkeypatch.setattr(rf, "_generate_video_ltx_ungated", lambda *a, **k: "VID_OK")
    assert rf.generate_video_ltx("a.png", "p", "o.mp4", "1280*720", 17) == "VID_OK"
    assert calls == [("video", "ltx-2.3")]


def test_tts_wav_routes_dramabox_through_gated(monkeypatch):
    import run_fleet as rf

    calls = []

    def fake_gated(role, model, fn, keep_after=False):
        calls.append((role, model))
        return fn()

    monkeypatch.setattr(rf, "_gated", fake_gated)
    monkeypatch.setattr(rf, "_tts_wav_ungated", lambda *a, **k: True)
    assert rf.tts_wav("hello", "/tmp/t.wav", engine="dramabox") is True
    assert calls == [("tts", "dramabox")]


def test_pipeline_lock_second_process_exits(tmp_path):
    import run_fleet as rf

    lock = str(tmp_path / "fleet.lock")
    fh = open(lock, "a+")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with mock.patch.object(rf.sys, "exit", side_effect=SystemExit(2)):
            with pytest.raises(SystemExit) as ei:
                rf._acquire_pipeline_lock(lock)
            assert ei.value.code == 2
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
