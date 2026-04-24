"""Smoke test for pipelines.mux — generate a tone, swap + mix onto canary clip."""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipelines import mux  # noqa: E402

# Defaults work out of the box; override via env for machine-specific paths.
# See .env.example for recommended names.
_OUTPUTS = os.environ.get("COMFY_OUTPUTS", "./comfy-outputs")
VIDEO = os.environ.get("MUX_SMOKE_VIDEO", os.path.join(_OUTPUTS, "canary_hero_00001_.mp4"))
TONE = os.environ.get("MUX_SMOKE_TONE", "/tmp/tone.wav")
OUT_REPLACE = os.environ.get("MUX_SMOKE_OUT_REPLACE", os.path.join(_OUTPUTS, "canary_hero_replaced.mp4"))
OUT_MIX = os.environ.get("MUX_SMOKE_OUT_MIX", os.path.join(_OUTPUTS, "canary_hero_mixed.mp4"))


def _sz(p):
    return os.path.getsize(p) if os.path.exists(p) else -1


def main():
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=10",
        TONE,
    ], check=True)
    print(f"tone: {TONE} ({_sz(TONE)} bytes)")

    ok1 = mux.replace_audio(VIDEO, TONE, OUT_REPLACE)
    print(f"replace_audio -> {ok1}: {OUT_REPLACE} ({_sz(OUT_REPLACE)} bytes)")

    ok2 = mux.mix_audio(VIDEO, TONE, OUT_MIX)
    print(f"mix_audio     -> {ok2}: {OUT_MIX} ({_sz(OUT_MIX)} bytes)")


if __name__ == "__main__":
    main()
