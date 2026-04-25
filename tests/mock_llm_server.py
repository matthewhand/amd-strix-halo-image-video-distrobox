#!/usr/bin/env python3
"""
Mock LLM Provider (OpenAI-compatible) HTTP server for CI.

Stdlib-only mock that emulates the OpenAI-compat surface used by
slopfinity/llm/providers.py and the LM Studio / Ollama probe code:

  GET  /v1/models             -> 200 list with one mock model
  POST /v1/chat/completions   -> 200 deterministic chat-completion shape
  POST /v1/completions        -> 200 deterministic text-completion shape

The completion body is shaped from the request:
  - System prompt mentions "STRICT JSON" or "image, video, music, tts"
    -> returns a JSON dict with those keys, so /enhance?distribute=true
       parses cleanly.
  - User prompt mentions "Suggest" or "subject ideas"
    -> returns a JSON array of N short subject strings.
  - Otherwise -> returns a single-sentence rewrite of the user message.

No third-party dependencies. Spawned as a subprocess by
tests/test_ai_mock_integration.py — port comes from LLM_MOCK_PORT
(default 11434, matching the Ollama default so probes also hit us).
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


_CANNED_SUBJECTS = [
    "lonely robot in a flooded mall",
    "lumpy clay deity inspecting tax forms",
    "neon dragon chewing through a fiber line",
    "philosophical toaster watches a sunset",
    "cyberpunk monk debugging a kettle",
    "rusted satellite dreaming of meadows",
    "dust-bowl android writing haiku",
    "void-fish translating spreadsheets",
]


def _canned_content(messages: list[dict], n_hint: int = 5) -> str:
    sys_p = ""
    user_p = ""
    for m in messages or []:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if role == "system":
            sys_p += "\n" + str(content)
        elif role == "user":
            user_p += "\n" + str(content)

    sys_l = sys_p.lower()
    user_l = user_p.lower()

    # Multi-stage distribute (STRICT JSON image/video/music/tts).
    if "strict json" in sys_l or ("image" in sys_l and "video" in sys_l and "music" in sys_l and "tts" in sys_l):
        payload = {
            "image": f"cinematic still: {user_p.strip() or 'a robot'}",
            "video": f"slow dolly across {user_p.strip() or 'a robot'}, 24fps",
            "music": "ambient synth, melancholy, slow",
            "tts": f"And so {user_p.strip() or 'the robot'} began its quiet shift.",
        }
        return json.dumps(payload)

    # Subject-ideas array (concept-artist prompt).
    if "subject ideas" in sys_l or "concept artist" in sys_l or "suggest" in user_l:
        # Try to parse a number out of the system prompt ("exactly N").
        n = n_hint
        for tok in sys_p.split():
            if tok.isdigit():
                n = max(1, min(20, int(tok)))
                break
        items = (_CANNED_SUBJECTS * ((n // len(_CANNED_SUBJECTS)) + 1))[:n]
        return json.dumps(items)

    # Default: one-sentence rewrite.
    base = user_p.strip().splitlines()[0] if user_p.strip() else "an idea"
    return f"A vivid cinematic rendering of {base}, lit by warm rim light."


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: D401 - quiet logs
        sys.stderr.write("[mock-llm] " + (fmt % args) + "\n")

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
            self._json(200, {"status": "ok", "service": "mock-llm"})
            return
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "mock-llm", "object": "model", "owned_by": "mock"},
                    ],
                },
            )
            return
        self._json(404, {"error": "not_found", "path": self.path})

    def do_POST(self):  # noqa: N802
        body = self._read_json()
        path = self.path.rstrip("/")
        if path in ("/v1/chat/completions", "/chat/completions"):
            messages = body.get("messages") or []
            content = _canned_content(messages)
            self._json(
                200,
                {
                    "id": "mock-cmpl-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model") or "mock-llm",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
            return
        if path in ("/v1/completions", "/completions"):
            prompt = body.get("prompt") or ""
            content = _canned_content(
                [{"role": "user", "content": str(prompt)}]
            )
            self._json(
                200,
                {
                    "id": "mock-cmpl-text-1",
                    "object": "text_completion",
                    "created": 0,
                    "model": body.get("model") or "mock-llm",
                    "choices": [
                        {"index": 0, "text": content, "finish_reason": "stop"},
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
            return
        self._json(404, {"error": "not_found", "path": self.path})


def main() -> int:
    host = os.environ.get("LLM_MOCK_HOST", "127.0.0.1")
    port = int(os.environ.get("LLM_MOCK_PORT", "11434"))
    server = HTTPServer((host, port), MockHandler)
    sys.stderr.write(f"[mock-llm] listening on http://{host}:{port}\n")
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
