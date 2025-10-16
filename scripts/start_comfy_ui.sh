#!/usr/bin/env bash
set -euo pipefail

# Wrapper for launching ComfyUI with sensible defaults on AMD Strix Halo.
# Provides automatic port fallback when the default (shared with Qwen) is busy.

DEFAULT_PORT=8000
FALLBACK_PORT=8188
LISTEN_ADDR="0.0.0.0"
OUTPUT_DIR="${HOME}/comfy-outputs"
USER_PORT_OVERRIDE=0
PORT_VALUE=""
EXTRA_ARGS=()

# Safe defaults tuned for WAN 2.2 on Strix Halo
export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-FALSE}"
export WAN22_DISABLE_FLASH_ATTN="${WAN22_DISABLE_FLASH_ATTN:-1}"
export WAN22_FORCE_TILED_VAE="${WAN22_FORCE_TILED_VAE:-1}"
export WAN22_SAFE_MODE="${WAN22_SAFE_MODE:-1}"
export WAN22_MAX_SIDE="${WAN22_MAX_SIDE:-640}"
export WAN22_MAX_FRAMES="${WAN22_MAX_FRAMES:-33}"
export WAN22_MAX_LATENT_ELEMENTS="${WAN22_MAX_LATENT_ELEMENTS:-2500000}"

usage() {
  cat <<'EOF'
Usage: start_comfy_ui [--port PORT] [--listen ADDR] [--output-directory DIR] [--]
       [additional ComfyUI arguments...]

Defaults:
  --listen             0.0.0.0
  --port               8000 (falls back to 8188 if busy and no override)
  --output-directory   $HOME/comfy-outputs

Any extra arguments after '--' are passed straight to ComfyUI.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help|-h)
        usage
        exit 0
        ;;
      --port)
        shift
        [[ $# -gt 0 ]] || { echo "start_comfy_ui: --port expects a value" >&2; exit 64; }
        PORT_VALUE="$1"
        USER_PORT_OVERRIDE=1
        ;;
      --port=*)
        PORT_VALUE="${1#*=}"
        USER_PORT_OVERRIDE=1
        ;;
      --listen)
        shift
        [[ $# -gt 0 ]] || { echo "start_comfy_ui: --listen expects a value" >&2; exit 64; }
        LISTEN_ADDR="$1"
        ;;
      --listen=*)
        LISTEN_ADDR="${1#*=}"
        ;;
      --host)
        shift
        [[ $# -gt 0 ]] || { echo "start_comfy_ui: --host expects a value" >&2; exit 64; }
        LISTEN_ADDR="$1"
        ;;
      --host=*)
        LISTEN_ADDR="${1#*=}"
        ;;
      --output-directory)
        shift
        [[ $# -gt 0 ]] || { echo "start_comfy_ui: --output-directory expects a value" >&2; exit 64; }
        OUTPUT_DIR="$1"
        ;;
      --output-directory=*)
        OUTPUT_DIR="${1#*=}"
        ;;
      --output-dir)
        shift
        [[ $# -gt 0 ]] || { echo "start_comfy_ui: --output-dir expects a value" >&2; exit 64; }
        OUTPUT_DIR="$1"
        ;;
      --output-dir=*)
        OUTPUT_DIR="${1#*=}"
        ;;
      --)
        shift
        EXTRA_ARGS+=("$@")
        break
        ;;
      *)
        EXTRA_ARGS+=("$1")
        ;;
    esac
    shift || true
  done
}

port_available() {
  local port="$1"
  local listen_addr="$2"
  python - "$port" "$listen_addr" <<'PY'
import socket
import sys

port = int(sys.argv[1])
addr = sys.argv[2]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((addr, port))
    except OSError:
        sys.exit(1)
sys.exit(0)
PY
}

describe_listener() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -Htlnp "sport = :${port}" 2>/dev/null || true
  fi
}

parse_args "$@"

PORT="${PORT_VALUE:-$DEFAULT_PORT}"

if ! port_available "$PORT" "$LISTEN_ADDR"; then
  echo "[WARN] Port ${PORT} is already in use on ${LISTEN_ADDR}." >&2
  describe_listener "$PORT" >&2
  if [[ $USER_PORT_OVERRIDE -eq 0 && "$PORT" == "$DEFAULT_PORT" ]]; then
    if port_available "$FALLBACK_PORT" "$LISTEN_ADDR"; then
      echo "[INFO] Falling back to port ${FALLBACK_PORT} for ComfyUI." >&2
      PORT="$FALLBACK_PORT"
    else
      echo "[ERROR] Neither ${DEFAULT_PORT} nor fallback ${FALLBACK_PORT} are available." >&2
      echo "        Stop the conflicting service or provide --port with a free port." >&2
      exit 98
    fi
  else
    echo "[ERROR] Specify a different port via --port or stop the conflicting service." >&2
    exit 98
  fi
fi

mkdir -p "${OUTPUT_DIR}"

cd /opt/ComfyUI

echo "Starting ComfyUI on ${LISTEN_ADDR}:${PORT} (outputs -> ${OUTPUT_DIR})"

exec python main.py \
  --listen "${LISTEN_ADDR}" \
  --port "${PORT}" \
  --output-directory "${OUTPUT_DIR}" \
  --disable-mmap \
  "${EXTRA_ARGS[@]}"
