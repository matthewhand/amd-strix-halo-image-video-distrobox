"""Noise-gate must reject mid-gray low-std static; accept real scene stats."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "orchestration"))
from marathon_loop import is_noise_arr, image_stats  # noqa: E402


def test_noise_signature_rejects_static():
    assert is_noise_arr(117.0, 20.0) is True
    assert is_noise_arr(116.8, 20.3) is True


def test_real_scene_stats_pass():
    # Photoreal scenes have higher structure / different mean
    assert is_noise_arr(177.0, 74.0) is False
    assert is_noise_arr(74.0, 53.0) is False
    assert is_noise_arr(127.8, 41.3) is False


def test_image_stats_on_theme_file():
    p = ROOT / "comfy-outputs/mage/themes/theme_01_product_still.png"
    if not p.is_file():
        return  # skip if cleaned
    s = image_stats(p)
    assert s["noise"] is False
    assert s["std"] > 40 or not (90 < s["mean"] < 140)
