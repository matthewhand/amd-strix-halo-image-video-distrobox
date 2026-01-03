#!/usr/bin/env python3
import sys
import os
import types
from unittest.mock import MagicMock

# === FLASH ATTENTION SHIM (Same as Qwen) ===
# Wan2.2 uses flash_attn library which tries to import flash_attn_2_cuda.
# On ROCm (TheRock), this might be named differently or need this mock to fallback/work.
if os.environ.get("WAN_ATTENTION_BACKEND") == "flash_attn":
    try:
        import flash_attn_2_cuda
    except ImportError:
        print("[Wan-Launcher] Injecting Flash Attention Shim for ROCm...", file=sys.stderr)
        fa_cuda = types.ModuleType("flash_attn_2_cuda")
        fa_cuda.varlen_fwd = MagicMock()
        fa_cuda.fwd = MagicMock()
        fa_cuda.bwd = MagicMock()
        fa_cuda.varlen_bwd = MagicMock()
        sys.modules["flash_attn_2_cuda"] = fa_cuda
# ============================

# Ensure we can import modules from the wan-video-studio directory
WAN_ROOT = "/opt/wan-video-studio"
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

# Change CWD to WAN_ROOT so relative asset loads work
os.chdir(WAN_ROOT)

try:
    from generate import generate, _parse_args
except ImportError as e:
    print(f"Error importing generate.py: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("[Wan-Launcher] Starting generation...", file=sys.stderr)
    args = _parse_args()
    generate(args)
