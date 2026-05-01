#!/usr/bin/env python3
"""
Integration test: slopfinity FastAPI app talking to stdlib mocks for every
AI surface (LLM provider + TTS worker).

CI rule (see tests/README.md): no real AI calls, ever. This test enforces
that contract for /enhance, /enhance?distribute, /subjects/suggest, and
/tts.

Spawns:
  1. tests/mock_llm_server.py   on 127.0.0.1:<LLM_MOCK_PORT> (default 11434)
  2. tests/mock_tts_server.py   on 127.0.0.1:<TTS_MOCK_PORT> (default 8010)
  3. uvicorn slopfinity.server:app on 127.0.0.1:<ephemeral>

Then exercises each endpoint and asserts on the response shape.

Run via pytest or directly:
    python -m pytest tests/test_ai_mock_integration.py -v
    python tests/test_ai_mock_integration.py
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
        time.sleep(0.2)
    raise RuntimeError(f"Service at {url} did not come up in {timeout}s: {last_err}")


def _post_json(url: str, payload: dict, timeout: float = 5.0) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def _get_json(url: str, timeout: float = 5.0) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


# ---------------------------------------------------------------------------
# Fixture: spawn mocks + app, tear down on exit
# ---------------------------------------------------------------------------

class _Stack:
    procs: list[subprocess.Popen] = []
    state_dir: Path | None = None
    app_url: str = ""
    setup_error: Exception | None = None


_STACK = _Stack()


def _spawn_all() -> _Stack:
    if _STACK.setup_error is not None:
        raise _STACK.setup_error
    if _STACK.procs:
        return _STACK

    llm_port = int(os.environ.get("LLM_MOCK_PORT") or _free_port())
    tts_port = int(os.environ.get("TTS_MOCK_PORT") or _free_port())
    app_port = _free_port()

    state_dir = Path(tempfile.mkdtemp(prefix="slopfinity-mock-"))
    (state_dir / "tts").mkdir(exist_ok=True)
    # Pre-seed config.json so lmstudio_call() targets our mock and skips
    # the auto-pick (we hard-code mock-llm so no extra round trip is needed).
    config = {
        "llm": {
            "provider": "lmstudio",
            "base_url": f"http://127.0.0.1:{llm_port}/v1",
            "model_id": "mock-llm",
            "api_key": "",
            "temperature": 0.0,
            "max_retries": 0,
            "timeout_s": 5,
        },
    }
    (state_dir / "config.json").write_text(json.dumps(config))

    env_common = os.environ.copy()
    env_common["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env_common.get("PYTHONPATH", "")

    # 1. LLM mock
    llm_env = env_common.copy()
    llm_env["LLM_MOCK_PORT"] = str(llm_port)
    p_llm = subprocess.Popen(
        [sys.executable, str(TESTS_DIR / "mock_llm_server.py")],
        env=llm_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _STACK.procs.append(p_llm)

    # 2. TTS mock
    tts_env = env_common.copy()
    tts_env["TTS_MOCK_PORT"] = str(tts_port)
    p_tts = subprocess.Popen(
        [sys.executable, str(TESTS_DIR / "mock_tts_server.py")],
        env=tts_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _STACK.procs.append(p_tts)

    _wait_http(f"http://127.0.0.1:{llm_port}/v1/models", timeout=10)
    _wait_http(f"http://127.0.0.1:{tts_port}/health", timeout=10)

    # 3. slopfinity app via uvicorn
    app_env = env_common.copy()
    app_env["SLOPFINITY_STATE_DIR"] = str(state_dir)
    app_env["TTS_WORKER_URL"] = f"http://127.0.0.1:{tts_port}/tts"
    # Belt-and-braces: also export LLM_PROVIDER_BASE_URL so any future code
    # path that reads it directly resolves to our mock.
    app_env["LLM_PROVIDER_BASE_URL"] = f"http://127.0.0.1:{llm_port}/v1"
    p_app = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "slopfinity.server:app",
            "--host", "127.0.0.1",
            "--port", str(app_port),
            "--log-level", "warning",
        ],
        cwd=str(REPO_ROOT),
        env=app_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _STACK.procs.append(p_app)

    app_url = f"http://127.0.0.1:{app_port}"
    try:
        _wait_http(app_url + "/", timeout=20)
    except Exception as wait_err:
        # Surface uvicorn stderr so the failure is debuggable.
        try:
            p_app.terminate()
            out, err = p_app.communicate(timeout=2)
            sys.stderr.write("[uvicorn-stdout]\n" + (out or b"").decode(errors="replace") + "\n")
            sys.stderr.write("[uvicorn-stderr]\n" + (err or b"").decode(errors="replace") + "\n")
        except Exception:
            pass
        _STACK.setup_error = RuntimeError(f"uvicorn failed to start: {wait_err}")
        raise _STACK.setup_error

    _STACK.state_dir = state_dir
    _STACK.app_url = app_url
    return _STACK


def _teardown() -> None:
    for p in _STACK.procs:
        try:
            p.terminate()
        except Exception:
            pass
    deadline = time.time() + 5
    for p in _STACK.procs:
        remaining = max(0.1, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    _STACK.procs.clear()


atexit.register(_teardown)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _setup_module_once() -> str:
    return _spawn_all().app_url


def test_enhance_simple():
    base = _setup_module_once()
    status, body = _post_json(base + "/enhance", {"prompt": "a robot"}, timeout=5)
    assert status == 200, body
    assert "suggestion" in body, body
    assert isinstance(body["suggestion"], str) and body["suggestion"], body


def test_enhance_distribute():
    base = _setup_module_once()
    status, body = _post_json(
        base + "/enhance",
        {"prompt": "a robot", "distribute": True},
        timeout=5,
    )
    assert status == 200, body
    assert body.get("distribute") is True, body
    stages = body.get("stages") or {}
    for k in ("image", "video", "music", "tts"):
        assert k in stages, (k, body)
        assert isinstance(stages[k], str), (k, body)
    # The mock returns a non-empty image/video string for distribute.
    assert stages["image"], body
    assert stages["video"], body


def test_subjects_suggest():
    base = _setup_module_once()
    # /subjects/suggest is a GET in the current server. Response shape is a
    # per-mode dict ({"story": [...], "simple": [...], "chat": [...]}); the
    # legacy flat-list shape was retired alongside the per-mode budgets.
    status, body = _get_json(base + "/subjects/suggest?n=3", timeout=5)
    assert status == 200, body
    assert "suggestions" in body, body
    sug = body["suggestions"]
    assert isinstance(sug, dict), body
    for mode in ("story", "simple", "chat"):
        assert mode in sug, (mode, body)
        assert isinstance(sug[mode], list), (mode, body)
        assert all(isinstance(s, str) for s in sug[mode]), (mode, body)


def test_tts_proxy():
    base = _setup_module_once()
    status, body = _post_json(
        base + "/tts",
        {"text": "hello", "voice": "ryan"},
        timeout=5,
    )
    assert status == 200, body
    assert body.get("ok") is True, body
    assert body.get("url"), body
    assert body.get("voice") == "ryan", body


# ---------------------------------------------------------------------------
# Direct-run entry point — `python tests/test_ai_mock_integration.py`
# ---------------------------------------------------------------------------

def _main() -> int:
    failures: list[str] = []
    tests = [
        test_enhance_simple,
        test_enhance_distribute,
        test_subjects_suggest,
        test_tts_proxy,
    ]
    try:
        _setup_module_once()
        for t in tests:
            try:
                t()
                sys.stdout.write(f"PASS {t.__name__}\n")
            except AssertionError as e:
                failures.append(f"{t.__name__}: {e}")
                sys.stdout.write(f"FAIL {t.__name__}: {e}\n")
            except Exception as e:  # noqa: BLE001
                failures.append(f"{t.__name__}: {e!r}")
                sys.stdout.write(f"ERROR {t.__name__}: {e!r}\n")
    finally:
        _teardown()
    if failures:
        sys.stdout.write(f"\n{len(failures)} failure(s)\n")
        return 1
    sys.stdout.write("\nAll AI-mock integration tests passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
