"""Tests for slopfinity.branding — load, list_profiles, _deep_merge, env overlays."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import branding as br


class TestBuiltinDefaults:
    def test_defaults_has_required_keys(self):
        assert "app" in br.BUILTIN_DEFAULTS
        assert "theme" in br.BUILTIN_DEFAULTS
        assert "colors" in br.BUILTIN_DEFAULTS
        assert "labels" in br.BUILTIN_DEFAULTS
        assert "aesthetics" in br.BUILTIN_DEFAULTS

    def test_defaults_app_values(self):
        assert br.BUILTIN_DEFAULTS["app"]["name"] == "Slopfinity"
        assert br.BUILTIN_DEFAULTS["app"]["logo_emoji"] == "♾️"


class TestEnvAestheticsOverlay:
    def test_env_overlay_skips_empty(self, monkeypatch):
        monkeypatch.delenv("SLOPFINITY_BG_IMAGE", raising=False)
        result = br._env_aesthetics_overlay()
        assert result == {}

    def test_env_overlay_parses_valid_values(self, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BG_IMAGE", "https://example.com/bg.png")
        monkeypatch.setenv("SLOPFINITY_BG_BLUR_PX", "10")
        monkeypatch.setenv("SLOPFINITY_BG_DIM_PCT", "50")
        monkeypatch.setenv("SLOPFINITY_BG_PARALLAX", "false")

        result = br._env_aesthetics_overlay()
        assert result["image_url"] == "https://example.com/bg.png"
        assert result["blur_px"] == 10
        assert result["dim_pct"] == 50
        assert result["parallax"] is False

    def test_env_overlay_parallax_truthy(self, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BG_PARALLAX", "yes")
        result = br._env_aesthetics_overlay()
        assert result["parallax"] is True

    def test_env_overlay_invalid_int_skipped(self, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BG_BLUR_PX", "not-a-number")
        result = br._env_aesthetics_overlay()
        assert "blur_px" not in result

    def test_env_overlay_blanks_skipped(self, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BG_IMAGE", "")
        result = br._env_aesthetics_overlay()
        assert "image_url" not in result


class TestDeepMerge:
    def test_deep_merge_simple(self):
        base = {"a": 1, "b": 2}
        over = {"b": 3, "c": 4}
        result = br._deep_merge(base, over)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        base = {"a": {"x": 1, "y": 2}, "b": 0}
        over = {"a": {"y": 3, "z": 4}}
        result = br._deep_merge(base, over)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}, "b": 0}

    def test_deep_merge_does_not_modify_base(self):
        base = {"a": {"x": 1}}
        over = {"a": {"y": 2}}
        result = br._deep_merge(base, over)
        assert "y" not in base["a"]


class TestListProfiles:
    def test_list_profiles_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        assert br.list_profiles() == ["slopfinity"]

    def test_list_profiles_with_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        (tmp_path / "custom.json").write_text("{}\n")
        (tmp_path / "demo.json.example").write_text("{}\n")
        profiles = br.list_profiles()
        assert "custom" in profiles
        assert "demo" in profiles
        assert profiles == sorted(profiles)


class TestLoadProfileFile:
    def test_load_profile_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        result = br._load_profile_file("nonexistent")
        assert result is None

    def test_load_profile_file_valid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        data = {"app": {"name": "TestApp"}}
        (tmp_path / "test.json").write_text('{\n  "app": {"name": "TestApp"}\n}\n')
        result = br._load_profile_file("test")
        assert result == {"app": {"name": "TestApp"}}

    def test_load_profile_file_strips_description(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        (tmp_path / "test.json").write_text(
            '{\n  "_description": "desc", "app": {"name": "Test"}\n}\n'
        )
        result = br._load_profile_file("test")
        assert "_description" not in result
        assert result["app"]["name"] == "Test"

    def test_load_profile_file_fallback_example(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        (tmp_path / "test.json.example").write_text(
            '{\n  "app": {"name": "Example"}\n}\n'
        )
        result = br._load_profile_file("test")
        assert result["app"]["name"] == "Example"

    def test_load_profile_file_corrupt_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        (tmp_path / "test.json").write_text("not valid json\n")
        result = br._load_profile_file("test")
        assert result is None


class TestLoad:
    def test_load_uses_builtin_defaults(self, monkeypatch):
        monkeypatch.delenv("SLOPFINITY_BRANDING_DIR", raising=False)
        monkeypatch.delenv("SLOPFINITY_BG_IMAGE", raising=False)
        monkeypatch.delenv("SLOPFINITY_BG_SVG", raising=False)
        result = br.load()
        assert result["app"]["name"] == "Slopfinity"

    def test_load_with_active_profile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        (tmp_path / "slopfinity.json").write_text(
            '{\n  "app": {"tagline": "Custom"}\n}\n'
        )
        result = br.load("slopfinity")
        assert result["app"]["tagline"] == "Custom"

    def test_load_merges_env_over_base(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        monkeypatch.setenv("SLOPFINITY_BG_IMAGE", "custom.png")
        monkeypatch.setenv("SLOPFINITY_BG_DIM_PCT", "75")
        result = br.load()
        assert result["aesthetics"]["image_url"] == "custom.png"
        assert result["aesthetics"]["dim_pct"] == 75

    def test_load_never_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
        (tmp_path / "broken.json").write_text("invalid")
        (tmp_path / "slopfinity.json").write_text("invalid")
        result = br.load("broken")
        assert isinstance(result, dict)
        assert "app" in result
