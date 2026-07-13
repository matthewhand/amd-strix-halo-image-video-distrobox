#!/usr/bin/env bash
# Live smoke for WAN via the same docker path workers use.
# Requires a complete official pack (prefer TI2V-5B from download_wan_official.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export WAN_CKPT_DIR="${WAN_CKPT_DIR:-/mnt/downloads/comfy-models/Wan2.2-TI2V-5B}"
export WAN_TASK="${WAN_TASK:-ti2v-5B}"
export WAN_SIZE="${WAN_SIZE:-704*1280}"
export WAN_FRAME_NUM="${WAN_FRAME_NUM:-9}"
export SLOPFINITY_WORKSPACE="$ROOT"

python3 - <<'PY'
import os, sys
from slopfinity.wan_cli import wan_paths, wan_launcher_argv, _is_complete_ckpt
cfg = wan_paths("wan2.2")
print("cfg", cfg)
if not _is_complete_ckpt(cfg["ckpt"], cfg["task"]):
    print("INCOMPLETE checkpoint tree:", cfg["ckpt"], file=sys.stderr)
    sys.exit(2)
ws = os.environ["SLOPFINITY_WORKSPACE"]
seed = os.path.join(ws, "comfy-outputs/experiments/wan_seed.png")
if not os.path.isfile(seed):
    seed = os.path.join(cfg["ckpt"], "examples/i2v_input.JPG")
out = os.path.join(ws, "comfy-outputs/experiments/wan_ti2v_smoke.mp4")
argv = wan_launcher_argv("neon city skyline, slow camera pan", seed, out, "wan2.2")
print("argv", argv)
# Write docker cmd for shell to run
ckpt = cfg["ckpt"]
base = os.path.basename(ckpt.rstrip("/"))
img = os.environ.get("SLOPFINITY_DOCKER_IMAGE", "amd-strix-halo-image-video-toolbox:latest")
# Prefer image id from running toolbox if present
open("/tmp/wan_smoke_cmd.sh", "w").write(
    "#!/bin/bash\nset -ex\n"
    f'docker run --rm '
    f'-e WAN_ATTENTION_BACKEND=sdpa '
    f'-v "{ws}:/workspace" '
    f'-v "{ckpt}:/models/{base}:ro" '
    f'-v "{os.path.expanduser("~/.cache/huggingface")}:/root/.cache/huggingface" '
    f'-w /workspace --device /dev/kfd --device /dev/dri '
    f'{img} '
    + " ".join(f"'{a}'" for a in argv)
    + "\n"
)
print("wrote /tmp/wan_smoke_cmd.sh")
PY

bash /tmp/wan_smoke_cmd.sh
ls -la "$ROOT/comfy-outputs/experiments/wan_ti2v_smoke.mp4"
echo "✅ WAN smoke ok"
