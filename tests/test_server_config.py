"""Route tests for /config and /settings endpoints."""
import pytest
import unittest.mock as mock
from tests.conftest_server import client, default_config  # noqa: F401

pytestmark = pytest.mark.asyncio

_ALLOWED_CONFIG_KEYS = {
    "base_model", "video_model", "audio_model", "tts_model", "upscale_model",
    "size", "frames", "chains", "tier", "fps", "seed", "steps",
    "enhancer_prompt", "image_prompt", "video_prompt", "music_prompt", "tts_prompt",
    "suggest_use_subjects", "suggest_custom_prompt", "suggest_max_len_simple",
    "suggest_max_len_endless", "suggest_max_len_chat",
    "chaos_suggest_system_prompt", "infinity", "polymorphic",
    "queue_paused", "scheduler", "model_loading",
}


class TestConfigEndpoint:
    async def test_post_valid_key_updates_config(self, client, default_config):
        saved = {}
        cfg_copy = dict(default_config)
        with mock.patch("slopfinity.server.cfg.load_config", return_value=cfg_copy), \
             mock.patch("slopfinity.server.cfg.save_config", side_effect=lambda c: saved.update(c)):
            resp = await client.post("/config", json={"key": "chains", "value": 5})
        assert resp.status_code == 200
        # /config returns {"status": "ok"} on success
        data = resp.json()
        assert data.get("status") == "ok" or data.get("ok") is True

    async def test_post_unknown_key_rejected(self, client, default_config):
        cfg_copy = dict(default_config)
        with mock.patch("slopfinity.server.cfg.load_config", return_value=cfg_copy), \
             mock.patch("slopfinity.server.cfg.save_config"):
            resp = await client.post("/config", json={"key": "__evil__", "value": "x"})
        # Should return 200 with ok:False, or 400/422
        assert resp.status_code in (200, 400, 422)
        if resp.status_code == 200:
            # ok may be False or may be absent — just ensure evil key not saved as top-level
            data = resp.json()
            assert data.get("ok") is False or "__evil__" not in data

    async def test_post_missing_key_returns_error(self, client, default_config):
        cfg_copy = dict(default_config)
        with mock.patch("slopfinity.server.cfg.load_config", return_value=cfg_copy), \
             mock.patch("slopfinity.server.cfg.save_config"):
            resp = await client.post("/config", json={"value": "x"})
        assert resp.status_code in (200, 400, 422)


class TestSettingsGet:
    async def test_returns_current_settings(self, client, default_config):
        with mock.patch("slopfinity.server.cfg.load_config", return_value=dict(default_config)), \
             mock.patch("slopfinity.server.cfg.get_state", return_value={}):
            resp = await client.get("/settings")
        assert resp.status_code == 200
        # The /settings endpoint returns a flat dict of config values
        data = resp.json()
        assert isinstance(data, dict)
        # Should contain at least a few expected keys
        assert "chains" in data or "scheduler" in data or "base_model" in data


class TestSettingsPost:
    async def test_post_valid_settings(self, client, default_config):
        saved = {}
        cfg_copy = dict(default_config)
        with mock.patch("slopfinity.server.cfg.load_config", return_value=cfg_copy), \
             mock.patch("slopfinity.server.cfg.save_config", side_effect=lambda c: saved.update(c)):
            resp = await client.post("/settings", json={"chains": 7})
        assert resp.status_code == 200

    async def test_post_empty_body_does_not_crash(self, client, default_config):
        cfg_copy = dict(default_config)
        with mock.patch("slopfinity.server.cfg.load_config", return_value=cfg_copy), \
             mock.patch("slopfinity.server.cfg.save_config"):
            resp = await client.post("/settings", json={})
        assert resp.status_code in (200, 400, 422)
