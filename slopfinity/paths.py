import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(PKG_DIR, "templates")
STATIC_DIR = os.path.join(PKG_DIR, "static")

# Where generated artifacts live. Honour SLOPFINITY_STATE_DIR (set by
# bin/slopfinity and already read by config.py / db.py / coordinator.py) so a
# host-native run agrees with the rest of the stack; keep SLOPFINITY_EXP_DIR as
# a back-compat fallback. In the container /workspace is the mount; fall back to
# the local experiments dir when the chosen dir isn't set or isn't writable
# (e.g. a root-owned /workspace on host, or a read-only bind in CI/sandboxes).
EXP_DIR = os.environ.get("SLOPFINITY_STATE_DIR") or os.environ.get("SLOPFINITY_EXP_DIR") or "/workspace"
if not os.path.isdir(EXP_DIR) or not os.access(EXP_DIR, os.W_OK):
    EXP_DIR = os.path.abspath("./comfy-outputs/experiments")
os.makedirs(EXP_DIR, exist_ok=True)

TTS_OUT_DIR = os.path.join(EXP_DIR, "tts")
os.makedirs(TTS_OUT_DIR, exist_ok=True)
