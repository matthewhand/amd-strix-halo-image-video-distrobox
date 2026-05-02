"""Unit tests for the pure chat-tool helper functions in slopfinity/server.py.

These functions are all side-effect-free modulo cfg.get_queue / cfg.save_queue,
which we monkey-patch below so no real files are touched.
"""
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We import the helpers directly from the server module. Because server.py
# imports a FastAPI app at module level (which in turn reads disk for branding)
# we patch the minimum necessary before importing.
import unittest.mock as mock

# Minimal stubs so the module-level FastAPI() call doesn't fail in a bare
# test environment (no /workspace, no branding JSON, etc.).
_branding_stub = {"app": {"name": "Test"}, "theme": {}}
_config_stub = {}

with mock.patch("slopfinity.branding.load", return_value=_branding_stub), \
     mock.patch("slopfinity.config.load_config", return_value=_config_stub):
    from slopfinity.routers.assets import _kind_of
    from slopfinity.routers.chat import (
        _chat_tool_queue_clip,
        _chat_tool_list_queue,
        _chat_tool_get_status,
        _chat_tool_cancel_item,
        _chat_tool_recent_finals,
        _chat_tool_describe_config,
    )
    from slopfinity.server import _trusted_origin_set


# ---------------------------------------------------------------------------
# _kind_of
# ---------------------------------------------------------------------------

class TestKindOf:
    def test_mp4_is_video(self):
        assert _kind_of("clip.mp4") == "video"

    def test_webm_is_video(self):
        assert _kind_of("clip.webm") == "video"

    def test_mov_is_video(self):
        assert _kind_of("clip.mov") == "video"

    def test_wav_is_audio(self):
        assert _kind_of("voice.wav") == "audio"

    def test_mp3_is_audio(self):
        assert _kind_of("track.mp3") == "audio"

    def test_ogg_is_audio(self):
        assert _kind_of("fx.ogg") == "audio"

    def test_flac_is_audio(self):
        assert _kind_of("hi.flac") == "audio"

    def test_png_is_image(self):
        assert _kind_of("frame.png") == "image"

    def test_unknown_ext_is_image(self):
        assert _kind_of("file.xyz") == "image"

    def test_uppercase_extension(self):
        # _kind_of uses .lower() — uppercase must not break classification.
        assert _kind_of("CLIP.MP4") == "video"


# ---------------------------------------------------------------------------
# _trusted_origin_set
# ---------------------------------------------------------------------------

class TestTrustedOriginSet:
    def test_includes_http_and_https_for_host(self):
        result = _trusted_origin_set("localhost:9099")
        assert "http://localhost:9099" in result
        assert "https://localhost:9099" in result

    def test_empty_host_returns_empty_or_only_env(self):
        result = _trusted_origin_set("")
        # Should not crash; env additions may be present but localhost:9099 won't be.
        assert isinstance(result, set)

    def test_env_additions_included(self, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_TRUSTED_ORIGINS", "https://example.com,https://other.net")
        result = _trusted_origin_set("localhost:9099")
        assert "https://example.com" in result
        assert "https://other.net" in result

    def test_env_additions_stripped_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_TRUSTED_ORIGINS", "https://example.com/")
        result = _trusted_origin_set("localhost:9099")
        assert "https://example.com" in result
        assert "https://example.com/" not in result


# ---------------------------------------------------------------------------
# _chat_tool_queue_clip
# ---------------------------------------------------------------------------

class TestChatToolQueueClip:
    def _make_queue(self, items=None):
        return items or []

    def test_empty_prompt_returns_error(self):
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]), \
             mock.patch("slopfinity.server.cfg.save_queue"):
            result = _chat_tool_queue_clip({"prompt": ""})
        assert result["ok"] is False
        assert "error" in result

    def test_valid_prompt_enqueues_and_returns_ok(self):
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda q: saved.extend(q)):
            result = _chat_tool_queue_clip({"prompt": "neon dragons"})
        assert result["ok"] is True
        assert result["prompt"] == "neon dragons"
        assert len(saved) == 1
        assert saved[0]["prompt"] == "neon dragons"
        assert saved[0]["status"] == "pending"
        assert saved[0]["priority"] == "next"

    def test_chains_clamped_to_valid_range(self):
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda q: saved.extend(q)):
            result = _chat_tool_queue_clip({"prompt": "x", "chains": 999})
        # max is 30
        assert saved[0]["config_snapshot"]["chains"] == 30

    def test_frames_clamped_to_valid_range(self):
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda q: saved.extend(q)):
            result = _chat_tool_queue_clip({"prompt": "x", "frames": 1})
        # min is 9
        assert saved[0]["config_snapshot"]["frames"] == 9

    def test_invalid_tier_not_included(self):
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda q: saved.extend(q)):
            _chat_tool_queue_clip({"prompt": "x", "tier": "ultra"})
        assert "tier" not in saved[0].get("config_snapshot", {})

    def test_valid_tier_low_included(self):
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda q: saved.extend(q)):
            _chat_tool_queue_clip({"prompt": "x", "tier": "low"})
        assert saved[0]["config_snapshot"]["tier"] == "low"

    def test_new_item_inserted_at_front_of_pending(self):
        existing = [
            {"status": "pending", "prompt": "old", "ts": 1.0},
        ]
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=existing), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda q: saved.__iadd__(q)):
            _chat_tool_queue_clip({"prompt": "new"})
        pending = [x for x in saved if x["status"] == "pending"]
        assert pending[0]["prompt"] == "new"
        assert pending[1]["prompt"] == "old"


# ---------------------------------------------------------------------------
# _chat_tool_list_queue
# ---------------------------------------------------------------------------

class TestChatToolListQueue:
    def _queue(self):
        return [
            {"status": "pending", "prompt": "a", "ts": 1.0},
            {"status": "pending", "prompt": "b", "ts": 2.0},
            {"status": "working", "prompt": "c", "ts": 3.0},
            {"status": "done",    "prompt": "d", "ts": 4.0},
            {"status": "cancelled", "prompt": "e", "ts": 5.0},
        ]

    def test_counts(self):
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=self._queue()):
            result = _chat_tool_list_queue({})
        assert result["pending"] == 2
        assert result["running"] == 1
        assert result["done"] == 1
        assert result["cancelled"] == 1

    def test_next_5_length(self):
        q = [{"status": "pending", "prompt": f"p{i}", "ts": float(i)} for i in range(10)]
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q):
            result = _chat_tool_list_queue({})
        assert len(result["next_5"]) == 5

    def test_prompt_truncated_to_80(self):
        long_prompt = "x" * 200
        q = [{"status": "pending", "prompt": long_prompt, "ts": 1.0}]
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q):
            result = _chat_tool_list_queue({})
        assert len(result["next_5"][0]["prompt"]) <= 80


# ---------------------------------------------------------------------------
# _chat_tool_get_status
# ---------------------------------------------------------------------------

class TestChatToolGetStatus:
    def test_returns_state_keys(self):
        state = {"mode": "Running", "step": "Image", "current_prompt": "cats", "chain_index": 2, "total_chains": 5, "video": 3, "started_at": 1000.0}
        with mock.patch("slopfinity.server.cfg.get_state", return_value=state):
            result = _chat_tool_get_status({})
        assert result["mode"] == "Running"
        assert result["chain"] == 2
        assert result["total_chains"] == 5

    def test_defaults_when_empty_state(self):
        with mock.patch("slopfinity.server.cfg.get_state", return_value={}):
            result = _chat_tool_get_status({})
        assert result["mode"] == "Idle"
        assert result["step"] == "Waiting"
        assert result["current_prompt"] == ""


# ---------------------------------------------------------------------------
# _chat_tool_cancel_item
# ---------------------------------------------------------------------------

class TestChatToolCancelItem:
    def test_invalid_ts_returns_error(self):
        result = _chat_tool_cancel_item({"ts": "not-a-number"})
        assert result["ok"] is False

    def test_cancels_matching_pending_item(self):
        ts = time.time()
        q = [{"status": "pending", "prompt": "neon", "ts": ts}]
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda x: saved.__iadd__(x)):
            result = _chat_tool_cancel_item({"ts": ts})
        assert result["ok"] is True
        assert saved[0]["status"] == "cancelled"

    def test_does_not_cancel_done_item(self):
        ts = time.time()
        q = [{"status": "done", "prompt": "done-item", "ts": ts}]
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q), \
             mock.patch("slopfinity.server.cfg.save_queue"):
            result = _chat_tool_cancel_item({"ts": ts})
        assert result["ok"] is False

    def test_no_matching_item_returns_error(self):
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]), \
             mock.patch("slopfinity.server.cfg.save_queue"):
            result = _chat_tool_cancel_item({"ts": 999.0})
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# _chat_tool_describe_config
# ---------------------------------------------------------------------------

class TestChatToolDescribeConfig:
    def test_returns_expected_keys(self):
        config = {"base_model": "qwen", "video_model": "ltx", "audio_model": None,
                  "tts_model": "kokoro", "size": "1280x720", "frames": 97,
                  "chains": 3, "tier": "med"}
        with mock.patch("slopfinity.server.cfg.load_config", return_value=config):
            result = _chat_tool_describe_config({})
        assert result["base_model"] == "qwen"
        assert result["frames"] == 97
        assert result["tier"] == "med"

    def test_missing_config_keys_return_none(self):
        with mock.patch("slopfinity.server.cfg.load_config", return_value={}):
            result = _chat_tool_describe_config({})
        assert result["base_model"] is None
        assert result["video_model"] is None
