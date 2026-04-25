#!/usr/bin/env python3
"""
Mock TTS Worker HTTP server for CI.

Stdlib-only mock that emulates the qwen-tts-service surface used by
slopfinity/server.py /tts proxy:

  GET  /health   -> 200 {"ok":true,"launcher":"mock","out":"/tmp/mock-tts"}
  POST /tts      -> 200 {"ok":true,"status":"ok","url":"/files/tts/...wav",
                          "voice":"<voice>"}
                  Side effect: writes a tiny valid 16 kHz mono 16-bit WAV
                  (1 second of silence) to /tmp/mock-tts/<voice>_<id>.wav.

Bind defaults to 127.0.0.1:8010 (matching the real worker), overridable via
TTS_MOCK_HOST / TTS_MOCK_PORT.

No third-party dependencies. The WAV is hand-built — a 44-byte RIFF header
plus 32000 bytes of zeroed samples (1 s @ 16 kHz mono 16-bit).
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE  # 1 second
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1
BYTE_RATE = SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE // 8
BLOCK_ALIGN = NUM_CHANNELS * BITS_PER_SAMPLE // 8
DATA_BYTES = NUM_SAMPLES * BLOCK_ALIGN  # 32000
RIFF_SIZE = 36 + DATA_BYTES


def _silence_wav() -> bytes:
    """Build a 44-byte RIFF/WAVE header + 32000 bytes of silence samples."""
    header = b"RIFF" + struct.pack("<I", RIFF_SIZE) + b"WAVE"
    fmt_chunk = (
        b"fmt "
        + struct.pack(
            "<IHHIIHH",
            16,                  # PCM fmt chunk size
            1,                   # PCM format
            NUM_CHANNELS,
            SAMPLE_RATE,
            BYTE_RATE,
            BLOCK_ALIGN,
            BITS_PER_SAMPLE,
        )
    )
    data_chunk = b"data" + struct.pack("<I", DATA_BYTES) + bytes(DATA_BYTES)
    return header + fmt_chunk + data_chunk


_OUT_DIR = Path(os.environ.get("TTS_MOCK_OUT", "/tmp/mock-tts"))


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: D401 - quiet logs
        sys.stderr.write("[mock-tts] " + (fmt % args) + "\n")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):  # noqa: N802
        if self.path in ("/", ""):
            self._json(200, {"status": "ok", "service": "mock-tts"})
            return
        if self.path.rstrip("/") == "/health":
            self._json(
                200,
                {"ok": True, "launcher": "mock", "out": str(_OUT_DIR)},
            )
            return
        self._json(404, {"error": "not_found", "path": self.path})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") == "/tts":
            body = self._read_json()
            voice = (body.get("voice") or "ryan").strip() or "ryan"
            uid = uuid.uuid4().hex[:12]
            ts = int(time.time() * 1000)
            fname = f"{voice}_{ts}_{uid}.wav"
            try:
                _OUT_DIR.mkdir(parents=True, exist_ok=True)
                (_OUT_DIR / fname).write_bytes(_silence_wav())
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"ok": False, "error": f"persist_failed: {exc}"})
                return
            self._json(
                200,
                {
                    "ok": True,
                    "status": "ok",
                    "url": f"/files/tts/mock_{uid}.wav",
                    "audio_path": f"/files/tts/mock_{uid}.wav",
                    "voice": voice,
                },
            )
            return
        self._json(404, {"error": "not_found", "path": self.path})


def main() -> int:
    host = os.environ.get("TTS_MOCK_HOST", "127.0.0.1")
    port = int(os.environ.get("TTS_MOCK_PORT", "8010"))
    server = HTTPServer((host, port), MockHandler)
    sys.stderr.write(f"[mock-tts] listening on http://{host}:{port}\n")
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
