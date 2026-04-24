"""Tests for the Qwen3-TTS proxy endpoint and the ffmpeg_mux helper.

These tests mock out all subprocess/HTTP calls — no ffmpeg or torch needed.
Run with: pytest tests/test_tts.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
from unittest import mock

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------- ffmpeg_mux command construction ----------

def test_mux_cmd_default_pads_audio():
    from slopfinity.workers import ffmpeg_mux

    cmd = ffmpeg_mux.build_cmd("v.mp4", "a.wav", "o.mp4")
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd
    # Default: pad_to_video=True, loop_audio=False → apad filter, mapped [a].
    assert "-filter_complex" in cmd
    idx = cmd.index("-filter_complex")
    assert cmd[idx + 1] == "[1:a]apad[a]"
    assert "-map" in cmd and "0:v:0" in cmd and "[a]" in cmd
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"
    assert "-shortest" in cmd
    assert cmd[-1] == "o.mp4"


def test_mux_cmd_loop_audio_uses_stream_loop():
    from slopfinity.workers import ffmpeg_mux

    cmd = ffmpeg_mux.build_cmd("v.mp4", "a.wav", "o.mp4", loop_audio=True)
    assert "-stream_loop" in cmd
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    # No apad filter when looping.
    assert "-filter_complex" not in cmd
    assert "1:a:0" in cmd


def test_mux_cmd_no_pad_no_loop_uses_plain_mapping():
    from slopfinity.workers import ffmpeg_mux

    cmd = ffmpeg_mux.build_cmd(
        "v.mp4", "a.wav", "o.mp4", loop_audio=False, pad_to_video=False
    )
    assert "-filter_complex" not in cmd
    assert "-stream_loop" not in cmd
    assert "0:v:0" in cmd and "1:a:0" in cmd


def test_mux_missing_inputs_raises():
    from slopfinity.workers import ffmpeg_mux

    with pytest.raises(FileNotFoundError):
        ffmpeg_mux.mux("/no/such/v.mp4", "/no/such/a.wav", "/tmp/o.mp4")


# ---------- /tts proxy: mock the worker HTTP call ----------

def _fake_urlopen(expected_payload):
    """Return a fake urlopen that asserts request payload."""

    class _Resp:
        def __init__(self, body):
            self._body = body.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    def _opener(req, timeout=None):
        data = json.loads(req.data.decode("utf-8"))
        assert data == expected_payload
        return _Resp(json.dumps({
            "ok": True, "status": "ok",
            "url": "/files/tts/tts_ryan_123.wav",
            "audio_path": "/files/tts/tts_ryan_123.wav",
            "voice": "ryan",
        }))

    return _opener


def test_tts_proxy_forwards_text_and_voice():
    from fastapi.testclient import TestClient
    import slopfinity.server as srv

    client = TestClient(srv.app)
    with mock.patch.object(
        srv.urllib.request, "urlopen",
        side_effect=_fake_urlopen({"text": "hi there", "voice": "ryan"}),
    ):
        r = client.post("/tts", json={"text": "hi there", "voice": "ryan"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["url"] == "/files/tts/tts_ryan_123.wav"
    assert body["audio_path"] == body["url"]
    assert body["voice"] == "ryan"


def test_tts_proxy_worker_unreachable_returns_503_not_sine():
    from fastapi.testclient import TestClient
    import slopfinity.server as srv

    client = TestClient(srv.app)
    with mock.patch.object(
        srv.urllib.request, "urlopen",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        r = client.post("/tts", json={"text": "hi", "voice": "ryan"})
    assert r.status_code == 503
    body = r.json()
    assert body["ok"] is False
    assert "qwen-tts-service" in body["error"]


def test_tts_proxy_empty_text_returns_400():
    from fastapi.testclient import TestClient
    import slopfinity.server as srv

    client = TestClient(srv.app)
    r = client.post("/tts", json={"text": "", "voice": "ryan"})
    assert r.status_code == 400


# ---------- qwen_tts_serve launcher invocation ----------

def test_serve_invokes_launcher_with_correct_args(tmp_path, monkeypatch):
    """qwen_tts_serve.py must shell out to qwen_tts_launcher.py with the
    text/voice/out/model args and HSA_OVERRIDE_GFX_VERSION=11.0.0."""
    monkeypatch.setenv("TTS_OUT_DIR", str(tmp_path))
    for mod in [m for m in list(sys.modules) if m.startswith("qwen_tts_serve")]:
        del sys.modules[mod]

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    import qwen_tts_serve  # type: ignore
    importlib.reload(qwen_tts_serve)

    from fastapi.testclient import TestClient

    client = TestClient(qwen_tts_serve.app)

    captured = {}

    def _fake_run(cmd, env=None, capture_output=False, text=False, timeout=None):
        captured["cmd"] = cmd
        captured["env"] = env
        out_idx = cmd.index("--out") + 1
        with open(cmd[out_idx], "wb") as f:
            f.write(b"RIFF....WAVE")

        class _P:
            returncode = 0
            stderr = ""
            stdout = ""

        return _P()

    with mock.patch.object(qwen_tts_serve.subprocess, "run", side_effect=_fake_run):
        r = client.post("/tts", json={"text": "hello", "voice": "ryan"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["voice"] == "ryan"

    cmd = captured["cmd"]
    assert "--text" in cmd and cmd[cmd.index("--text") + 1] == "hello"
    assert "--voice" in cmd and cmd[cmd.index("--voice") + 1] == "ryan"
    assert "--out" in cmd
    assert "--model" in cmd
    assert captured["env"]["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"
