import os
import sys

# Suppress emoji dir prints from serve modules during pytest runs
os.environ.setdefault("SLOPFINITY_QUIET", "1")

# scripts/ on path so serve modules and fixtures resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Ignore heavy modules that require torch etc not present in harness python
collect_ignore = [
    "test_enhancer.py",
    "test_llm_enhancer.py",
    "e2e_qwen_web_test.py",
    "legacy",
    "render_variety.py",
    "run_all.sh",
]

import pytest


@pytest.fixture(autouse=True)
def _fast_voices(monkeypatch):
    """Avoid slow kokoro_onnx import in tts voices contract tests."""
    def _fake_voices_impl():
        return {
            "ok": True,
            "default_engine": "kokoro",
            "engines": {
                "kokoro": {"default_voice": "af_heart", "voices": ["af_heart"]},
            },
        }
    try:
        import qwen_tts_serve as ts
    except ImportError:
        yield
        return
    monkeypatch.setattr(ts, "_voices_impl", _fake_voices_impl)
    yield
