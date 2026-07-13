#!/usr/bin/env bash
# ComfyUI service entrypoint: kornia.pad shim + start main.py
set -euo pipefail

if (echo > /dev/tcp/127.0.0.1/8188) 2>/dev/null; then
  echo "ERROR: Port 8188 is already bound (likely a toolbox running start_docker.sh)." 1>&2
  echo "       Stop the other ComfyUI server first." 1>&2
  sleep 5
  exit 1
fi

# kornia dropped pyramid.pad — without this, ComfyUI-LTXVideo fails to import
# and all LTX nodes disappear.
python3 /opt/patch_ltxvideo_kornia.py || true

echo "🖼️  Starting ComfyUI (FP16 weights, FP32 VAE)..."
cd /opt/ComfyUI
# --fp32-vae: forces every VAE decode to fp32 regardless of the model's
# native dtype. On Strix Halo (gfx1151) bf16/fp16 VAE decodes produce a faint
# 8/16-px block-grid pattern in some models (LTX-Video-2.3 especially).
exec python3 main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --output-directory /opt/ComfyUI/output \
  --force-fp16 \
  --fp32-vae
