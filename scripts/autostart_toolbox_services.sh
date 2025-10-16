#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${HOME}/.local/share/toolbox-logs"
mkdir -p "${LOG_DIR}"

LOCK_FILE="${LOG_DIR}/autostart.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

patch_wan_nodes() {
  if [[ -x "${HOME}/amd-strix-halo-image-video-toolboxes/scripts/patch_wan_nodes.py" ]]; then
    python "${HOME}/amd-strix-halo-image-video-toolboxes/scripts/patch_wan_nodes.py" || echo "[autostart] WARN: failed to patch Wan nodes" >>"${LOG_DIR}/autostart.log"
  fi
}

port_in_use() {
  local port="$1"
  python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sys.exit(0)
sys.exit(1)
PY
}

start_qwen() {
  if pgrep -f 'start_qwen_studio' >/dev/null 2>&1; then
    return
  fi
  if port_in_use 8000; then
    return
  fi
  echo "[autostart] Launching Qwen Image Studio" >>"${LOG_DIR}/autostart.log"
  (
    cd /opt/qwen-image-studio
    nohup uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 \
      >>"${LOG_DIR}/qwen-image.log" 2>&1 &
  )
}

start_comfy() {
  if pgrep -f '/opt/ComfyUI/main.py' >/dev/null 2>&1; then
    return
  fi
  if port_in_use 8188; then
    return
  fi
  echo "[autostart] Launching ComfyUI" >>"${LOG_DIR}/autostart.log"
  nohup "${HOME}/.local/bin/start_comfy_ui" --listen 0.0.0.0 --port 8188 \
      >>"${LOG_DIR}/comfy.log" 2>&1 &
}

patch_wan_nodes
start_qwen
sleep 2
start_comfy
