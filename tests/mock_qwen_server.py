#!/usr/bin/env python3
"""
Mock Qwen Image Web Server for CI.

Stdlib-only mock that emulates the routes used by tests/e2e_qwen_web_test.py:
  GET  /                   -> 200 {"status":"ok"}
  POST /api/generate       -> 200 {"job_id":"mock-<uuid>"} and writes a tiny
                              valid PNG to ~/.qwen-image-studio/<job_id>.png
  GET  /api/job/<id>       -> 200 {"status":"completed"}

No third-party dependencies. Designed to be spawned as a subprocess by the
E2E test when MOCK_QWEN_WEB=1.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Smallest possible valid PNG: 1x1 transparent pixel.
# Source: well-known canonical 67-byte PNG.
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: D401 - quiet logs
        sys.stderr.write("[mock-qwen] " + (fmt % args) + "\n")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/" or self.path == "":
            self._json(200, {"status": "ok"})
            return
        if self.path.startswith("/api/job/"):
            self._json(200, {"status": "completed"})
            return
        self._json(404, {"error": "not_found", "path": self.path})

    def do_POST(self):  # noqa: N802 - http.server API
        if self.path == "/api/generate":
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length:
                # Drain body; we don't care about contents.
                self.rfile.read(length)
            job_id = f"mock-{uuid.uuid4().hex[:12]}"
            try:
                out_dir = Path.home() / ".qwen-image-studio"
                out_dir.mkdir(parents=True, exist_ok=True)
                # Pad past 100 bytes so the persistence test (>100B in mock
                # mode) is comfortably satisfied. Trailing bytes after IEND
                # are ignored by PNG decoders; we only care about file size.
                (out_dir / f"{job_id}.png").write_bytes(TINY_PNG + b"\x00" * 1024)
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": f"persist_failed: {exc}"})
                return
            self._json(200, {"job_id": job_id})
            return
        self._json(404, {"error": "not_found", "path": self.path})


def main() -> int:
    host = os.environ.get("MOCK_QWEN_HOST", "127.0.0.1")
    port = int(os.environ.get("MOCK_QWEN_PORT", "8000"))
    server = HTTPServer((host, port), MockHandler)
    sys.stderr.write(f"[mock-qwen] listening on http://{host}:{port}\n")
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
