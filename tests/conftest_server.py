"""
Shared pytest fixtures for FastAPI route tests.

Uses httpx.AsyncClient with ASGITransport so tests run against the real app
without binding a port. No external servers needed.

The module-level FastAPI() call in server.py reads branding from disk, so we
patch the minimum globals before the module loads.
"""
import os
import sys
import pytest
import pytest_asyncio
import unittest.mock as mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_branding_stub = {"app": {"name": "Test"}, "theme": {}}
_default_config = {
    "base_model": "qwen",
    "video_model": "ltx",
    "audio_model": None,
    "tts_model": None,
    "size": "1280x720",
    "frames": 97,
    "chains": 3,
    "tier": "med",
    "enhancer_prompt": "Rewrite concisely.",
    "suggest_use_subjects": True,
    "suggest_custom_prompt": "",
    "branding": {"active": "slopfinity"},
}

# Patch branding + config before the server module is imported so the
# FastAPI() title call doesn't touch the filesystem.
with mock.patch("slopfinity.branding.load", return_value=_branding_stub), \
     mock.patch("slopfinity.config.load_config", return_value=_default_config):
    from slopfinity.server import app


import httpx
from httpx import AsyncClient, ASGITransport


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    """Async HTTPX client wired to the ASGI app.

    Sets SLOPFINITY_DISABLE_CSRF=1 so route tests don't have to add
    Origin headers on every POST.  Tests that explicitly check CSRF
    behaviour should monkeypatch SLOPFINITY_DISABLE_CSRF back to ''
    themselves.
    """
    monkeypatch.setenv("SLOPFINITY_DISABLE_CSRF", "1")
    # Point EXP_DIR at a tmp dir so asset-listing tests don't read
    # the real /workspace or the developer's comfy-outputs folder.
    monkeypatch.setattr("slopfinity.server.EXP_DIR", str(tmp_path))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


@pytest.fixture
def default_config():
    return dict(_default_config)
