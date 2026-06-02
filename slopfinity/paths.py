import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(PKG_DIR, "templates")
STATIC_DIR = os.path.join(PKG_DIR, "static")

# Where generated artifacts live. In the container /workspace is mounted;
# in local dev (or when /workspace exists but isn't writable, e.g. a
# read-only bind in CI/sandboxes) we fall back to the experiments dir.
EXP_DIR = os.environ.get("SLOPFINITY_EXP_DIR") or "/workspace"
if not (os.path.isdir(EXP_DIR) and os.access(EXP_DIR, os.W_OK)):
    EXP_DIR = os.path.abspath("./comfy-outputs/experiments")
    os.makedirs(EXP_DIR, exist_ok=True)

TTS_OUT_DIR = os.path.join(EXP_DIR, "tts")
os.makedirs(TTS_OUT_DIR, exist_ok=True)
