#!/usr/bin/env bash
# Download official Wan-AI packs for wan-video-studio generate.py.
#
# The Comfy-Org repackaged fp8 safetensors + misnamed T5 .pth under
# Wan2.2-T2V-A14B-official do NOT work with generate.py (T5 state_dict
# key mismatch). Use this script for the native layout.
#
# Usage:
#   ./scripts/download_wan_official.sh ti2v-5b     # ~34 GB (recommended)
#   ./scripts/download_wan_official.sh t2v-a14b    # ~126 GB
#   ./scripts/download_wan_official.sh i2v-a14b    # ~126 GB
#
# Dest: $WAN_MODELS_ROOT/<name>  (default /mnt/downloads/comfy-models)

set -euo pipefail

ROOT="${WAN_MODELS_ROOT:-/mnt/downloads/comfy-models}"
# Prefer a user-writable HF cache (root-owned hub dirs break refs).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface-user}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$ROOT"

target="${1:-ti2v-5b}"
case "$target" in
  ti2v-5b|ti2v|5b)
    REPO="Wan-AI/Wan2.2-TI2V-5B"
    DEST="$ROOT/Wan2.2-TI2V-5B"
    ;;
  t2v-a14b|t2v)
    REPO="Wan-AI/Wan2.2-T2V-A14B"
    DEST="$ROOT/Wan2.2-T2V-A14B"
    ;;
  i2v-a14b|i2v)
    REPO="Wan-AI/Wan2.2-I2V-A14B"
    DEST="$ROOT/Wan2.2-I2V-A14B"
    ;;
  *)
    echo "Unknown target: $target" >&2
    echo "Use: ti2v-5b | t2v-a14b | i2v-a14b" >&2
    exit 2
    ;;
esac

echo "==> $REPO → $DEST"
echo "    HF_HOME=$HF_HOME"
python3 -u - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="$REPO",
    local_dir="$DEST",
    max_workers=4,
)
print("DONE", path)
PY

# Quick layout check
python3 - <<PY
import os, sys
dest = "$DEST"
need = ["models_t5_umt5-xxl-enc-bf16.pth"]
if "TI2V" in dest:
    need += ["Wan2.2_VAE.pth", "config.json"]
else:
    need += ["Wan2.1_VAE.pth", "high_noise_model", "low_noise_model"]
ok = True
for n in need:
    p = os.path.join(dest, n)
    if not os.path.exists(p):
        print("MISSING", p)
        ok = False
    else:
        sz = os.path.getsize(p) if os.path.isfile(p) else 0
        print("ok", n, f"({sz/1e9:.2f} GB)" if sz else "(dir)")
sys.exit(0 if ok else 1)
PY

echo "✅ Official pack ready. Point slopfinity at it with:"
echo "   export WAN_CKPT_DIR=$DEST"
if [[ "$DEST" == *TI2V* ]]; then
  echo "   export WAN_TASK=ti2v-5B WAN_SIZE=704*1280 WAN_FRAME_NUM=17"
else
  echo "   export WAN_TASK=t2v-A14B WAN_SIZE=480*832 WAN_FRAME_NUM=17"
fi
