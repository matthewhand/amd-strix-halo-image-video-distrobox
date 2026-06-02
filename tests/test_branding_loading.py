"""Profile-loading / fallback-chain tests for slopfinity.branding.

Complements tests/test_branding.py, which already covers BUILTIN_DEFAULTS,
the SLOPFINITY_BG_* env overlay, _deep_merge in isolation, list_profiles,
and _load_profile_file. This file adds the cases that suite lacks:

  * the full three-tier fallback chain through load():
        active profile  ->  slopfinity profile  ->  BUILTIN_DEFAULTS,
  * deep-merge LAYERING through load() (nested dicts merged, not replaced),
  * env-aesthetics overlay winning over a profile-supplied aesthetics block,
  * the active-profile-missing case (slopfinity fallback still applies),
  * .json taking precedence over .json.example for the same name.

All tests are hermetic — SLOPFINITY_BRANDING_DIR is pointed at a tmp dir and
SLOPFINITY_BG_* are cleared so no host env leaks in.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity import branding as br  # noqa: E402


@pytest.fixture
def branding_env(tmp_path, monkeypatch):
    """Isolate branding to a tmp dir with all SLOPFINITY_BG_* cleared."""
    monkeypatch.setenv("SLOPFINITY_BRANDING_DIR", str(tmp_path))
    for env_key in br._ENV_AESTHETICS_KEYS:
        monkeypatch.delenv(env_key, raising=False)
    return tmp_path


def _write(d, name, payload):
    import json
    (d / name).write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Fallback chain: active -> slopfinity -> BUILTIN_DEFAULTS
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_active_layers_over_slopfinity_over_builtin(self, branding_env):
        # slopfinity provides a tagline; active "demo" provides a name.
        # Keys neither provides come from BUILTIN_DEFAULTS.
        _write(branding_env, "slopfinity.json",
               {"app": {"tagline": "From Slopfinity"}})
        _write(branding_env, "demo.json",
               {"app": {"name": "DemoApp"}})
        result = br.load("demo")
        assert result["app"]["name"] == "DemoApp"          # from active
        assert result["app"]["tagline"] == "From Slopfinity"  # from slopfinity
        assert result["app"]["logo_emoji"] == \
            br.BUILTIN_DEFAULTS["app"]["logo_emoji"]        # from builtin

    def test_active_overrides_slopfinity_same_key(self, branding_env):
        _write(branding_env, "slopfinity.json", {"app": {"name": "Base"}})
        _write(branding_env, "demo.json", {"app": {"name": "Override"}})
        result = br.load("demo")
        assert result["app"]["name"] == "Override"

    def test_missing_active_still_applies_slopfinity_fallback(self, branding_env):
        # No "demo.json" exists, but slopfinity.json does — its values must
        # still layer over the builtin defaults.
        _write(branding_env, "slopfinity.json", {"app": {"name": "SlopOnly"}})
        result = br.load("demo")
        assert result["app"]["name"] == "SlopOnly"

    def test_no_profiles_at_all_returns_builtin(self, branding_env):
        result = br.load("demo")
        assert result == br.BUILTIN_DEFAULTS or \
            result["app"]["name"] == br.BUILTIN_DEFAULTS["app"]["name"]

    def test_slopfinity_active_does_not_double_load(self, branding_env):
        # When active == "slopfinity", the loader skips the separate
        # fallback step and loads it once as the chosen profile.
        _write(branding_env, "slopfinity.json", {"app": {"name": "Slop"}})
        result = br.load("slopfinity")
        assert result["app"]["name"] == "Slop"


# ---------------------------------------------------------------------------
# Deep-merge layering through load() (nested dicts merged, not clobbered)
# ---------------------------------------------------------------------------

class TestLoadDeepMerge:
    def test_partial_nested_colors_merge(self, branding_env):
        # Profile overrides only one color; the rest survive from builtin.
        _write(branding_env, "slopfinity.json",
               {"colors": {"primary": "#000000"}})
        result = br.load()
        assert result["colors"]["primary"] == "#000000"      # overridden
        assert result["colors"]["accent"] == \
            br.BUILTIN_DEFAULTS["colors"]["accent"]          # preserved

    def test_partial_labels_merge_across_two_profiles(self, branding_env):
        _write(branding_env, "slopfinity.json",
               {"labels": {"queue": "Base Queue"}})
        _write(branding_env, "demo.json",
               {"labels": {"inject": "Demo Inject"}})
        result = br.load("demo")
        assert result["labels"]["queue"] == "Base Queue"     # slopfinity
        assert result["labels"]["inject"] == "Demo Inject"   # active
        assert result["labels"]["pipeline"] == \
            br.BUILTIN_DEFAULTS["labels"]["pipeline"]        # builtin


# ---------------------------------------------------------------------------
# Env-aesthetics overlay wins over profile JSON
# ---------------------------------------------------------------------------

class TestEnvOverlayWinsOverProfile:
    def test_env_image_overrides_profile_aesthetics(self, branding_env, monkeypatch):
        _write(branding_env, "slopfinity.json",
               {"aesthetics": {"image_url": "profile.png", "dim_pct": 10}})
        monkeypatch.setenv("SLOPFINITY_BG_IMAGE", "env.png")
        result = br.load()
        assert result["aesthetics"]["image_url"] == "env.png"  # env wins
        # dim_pct not set by env, so the profile value survives.
        assert result["aesthetics"]["dim_pct"] == 10

    def test_env_overlay_preserves_other_blocks(self, branding_env, monkeypatch):
        _write(branding_env, "slopfinity.json", {"app": {"name": "Keep"}})
        monkeypatch.setenv("SLOPFINITY_BG_DIM_PCT", "88")
        result = br.load()
        assert result["aesthetics"]["dim_pct"] == 88
        assert result["app"]["name"] == "Keep"  # untouched by env overlay


# ---------------------------------------------------------------------------
# .json beats .json.example for the same profile name (via load)
# ---------------------------------------------------------------------------

class TestJsonPrecedence:
    def test_json_preferred_over_example(self, branding_env):
        _write(branding_env, "demo.json", {"app": {"name": "Real"}})
        _write(branding_env, "demo.json.example", {"app": {"name": "Example"}})
        _write(branding_env, "slopfinity.json", {})
        result = br.load("demo")
        assert result["app"]["name"] == "Real"

    def test_example_used_when_no_json(self, branding_env):
        _write(branding_env, "demo.json.example", {"app": {"name": "Example"}})
        result = br.load("demo")
        assert result["app"]["name"] == "Example"


# ---------------------------------------------------------------------------
# load() never raises on broken files (corruption resilience through chain)
# ---------------------------------------------------------------------------

class TestLoadResilience:
    def test_broken_active_falls_through_to_slopfinity(self, branding_env):
        (branding_env / "demo.json").write_text("{ not json")
        _write(branding_env, "slopfinity.json", {"app": {"name": "Slop"}})
        result = br.load("demo")
        assert result["app"]["name"] == "Slop"

    def test_all_broken_returns_builtin(self, branding_env):
        (branding_env / "demo.json").write_text("nope")
        (branding_env / "slopfinity.json").write_text("nope")
        result = br.load("demo")
        assert result["app"]["name"] == br.BUILTIN_DEFAULTS["app"]["name"]
