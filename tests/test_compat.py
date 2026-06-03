"""Known-broken-state registry (slopfinity/compat.py).

Pure-logic tests — no server/paths import, so they run anywhere.
"""
from slopfinity import compat


class TestClampErnieDims:
    def test_default_is_ceiling(self):
        assert compat.clamp_ernie_dims() == (compat.ERNIE_MAX_DIM, compat.ERNIE_MAX_DIM)

    def test_oversize_clamped(self):
        assert compat.clamp_ernie_dims(1024, 1024) == (512, 512)
        assert compat.clamp_ernie_dims(2048, 768) == (512, 512)

    def test_under_ceiling_preserved(self):
        assert compat.clamp_ernie_dims(256, 384) == (256, 384)

    def test_ceiling_is_512(self):
        # The guard value run_fleet.py / worker_sh.py depend on. If this ever
        # changes, the docker `--width/--height` args must move with it.
        assert compat.ERNIE_MAX_DIM == 512


class TestCheckConfig:
    def test_ernie_is_danger_with_autofix(self):
        warns = compat.check_config({"base_model": "ernie"})
        assert len(warns) == 1
        w = warns[0]
        assert w["id"] == "ernie-hires-vae-hang"
        assert w["severity"] == "danger"
        # auto_fix rule ⇒ message should say it's *capped*, not just broken
        assert "512" in w["message"]

    def test_qwen_is_clean(self):
        assert compat.check_config({"base_model": "qwen"}) == []

    def test_wan_video_flake(self):
        for m in ("wan2.2", "wan2.5"):
            ids = [w["id"] for w in compat.check_config({"video_model": m})]
            assert "wan-video-flake" in ids

    def test_qwen_tts_broken(self):
        ids = [w["id"] for w in compat.check_config({"tts_model": "qwen-tts"})]
        assert "qwen-tts-broken" in ids

    def test_multiple_issues_accumulate(self):
        warns = compat.check_config({"base_model": "ernie", "tts_model": "qwen-tts"})
        ids = {w["id"] for w in warns}
        assert ids == {"ernie-hires-vae-hang", "qwen-tts-broken"}

    def test_empty_config_no_warnings(self):
        assert compat.check_config({}) == []
