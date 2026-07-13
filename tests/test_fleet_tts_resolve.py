"""run_fleet.tts_wav must resolve worker WAVs under experiments/tts."""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_tts_wav_finds_file_under_output_dir_tts(tmp_path, monkeypatch):
    import run_fleet as rf

    out_dir = tmp_path / "comfy-outputs" / "experiments"
    tts_dir = out_dir / "tts"
    tts_dir.mkdir(parents=True)
    wav = tts_dir / "tts_kokoro_af_heart_test.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)

    dest = out_dir / "slop_1_test_tts.wav"
    monkeypatch.setattr(rf, "OUTPUT_DIR", str(out_dir))

    envelope = {
        "ok": True,
        "url": "/files/tts/tts_kokoro_af_heart_test.wav",
        "voice": "af_heart",
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(envelope).encode()

    monkeypatch.setattr(
        rf.urllib.request,
        "urlopen",
        lambda *a, **k: _Resp(),
    )
    # Stage gate would docker-start TTS; unit test only checks path resolve.
    monkeypatch.setattr(rf, "_gated", lambda role, model, fn, **k: fn())
    ok = rf.tts_wav("hello world narration", str(dest), voice="af_heart")
    assert ok is True
    assert dest.is_file()
    assert dest.read_bytes()[:4] == b"RIFF"
