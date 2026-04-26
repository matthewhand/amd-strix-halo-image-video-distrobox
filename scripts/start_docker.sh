#!/bin/bash
set -e

echo '🚀 Starting Strix Halo Toolbox services...'

# Default configuration (env vars passed from docker-compose)
QWEN_PORT=${QWEN_PORT:-8000}
COMFYUI_PORT=${COMFYUI_PORT:-8188}

# --- Deploy Patched Launcher (Shim for ROCm) ---
if [ -f "/opt/qwen_launcher.py" ]; then
    echo "🔧 Deploying patched Qwen launcher with Flash Attention shim..."
    cp /opt/qwen_launcher.py /opt/qwen-image-studio/qwen-image-mps.py
fi

# --- Start Qwen Image Studio ---
if [[ "$QWEN_PORT" != "0" && "$QWEN_PORT" != "false" ]]; then
  echo "🎨 Starting Qwen Image Studio on port $QWEN_PORT..."
  cd /opt/qwen-image-studio/qwen-image-studio
  uvicorn server:app --host 0.0.0.0 --port "$QWEN_PORT" &
  QWEN_PID=$!
else
  echo '🎨 Qwen Image Studio: DISABLED'
  QWEN_PID=''
fi

# --- Start ComfyUI ---
if [[ "$COMFYUI_PORT" != "0" && "$COMFYUI_PORT" != "false" ]]; then
  echo "🖼️  Starting ComfyUI on port $COMFYUI_PORT..."
  cd /opt/ComfyUI
  # --fp32-vae kills the bf16 VAE-decode block-grid on Strix Halo (gfx1151).
  # See docker-compose.yaml's ComfyUI command for the full rationale.
  python3 main.py --listen 0.0.0.0 --port "$COMFYUI_PORT" --output-directory /opt/ComfyUI/output --fp32-vae &
  COMFYUI_PID=$!
else
  echo '🖼️  ComfyUI: DISABLED'
  COMFYUI_PID=''
fi

echo '✅ Services startup sequence complete.'
echo ''
echo "📊 Service status (PID: Qwen=$QWEN_PID, ComfyUI=$COMFYUI_PID)"

# --- Monitoring Loop ---
while true; do
  sleep 30
  
  # Monitor Qwen if enabled
  if [[ -n "$QWEN_PID" ]]; then
    if ! kill -0 "$QWEN_PID" 2>/dev/null; then
      echo '⚠️  Qwen service died, restarting...'
      cd /opt/qwen-image-studio/qwen-image-studio
      uvicorn server:app --host 0.0.0.0 --port "$QWEN_PORT" &
      QWEN_PID=$!
    fi
  fi
  
  # Monitor ComfyUI if enabled
  if [[ -n "$COMFYUI_PID" ]]; then
    if ! kill -0 "$COMFYUI_PID" 2>/dev/null; then
      echo '⚠️  ComfyUI service died, restarting...'
      cd /opt/ComfyUI
      # --fp32-vae kills the bf16 VAE-decode block-grid on Strix Halo (gfx1151).
  # See docker-compose.yaml's ComfyUI command for the full rationale.
  python3 main.py --listen 0.0.0.0 --port "$COMFYUI_PORT" --output-directory /opt/ComfyUI/output --fp32-vae &
      COMFYUI_PID=$!
    fi
  fi
  
  # Exit if both disabled or both died (and failed to restart)
  if [[ -z "$QWEN_PID" && -z "$COMFYUI_PID" ]]; then
     echo '❌ No services running. Exiting.'
     exit 0
  fi
done
