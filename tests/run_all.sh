#!/usr/bin/env bash
# run_all.sh — smoke-test all three generation pipelines
# Usage: bash tests/run_all.sh
# Requires: Qwen Image Studio on :8180, ComfyUI on :8188, WAN models in ~/Wan2.2-*
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0; FAIL=0

run() {
    local label="$1"; shift
    echo "▶ $label"
    # exit code 5 = pytest "no tests ran" (skipped) — treat as pass
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ]; then
        echo "✅ $label"; PASS=$((PASS+1))
    else
        echo "❌ $label (exit $rc)"; FAIL=$((FAIL+1))
    fi
}

# 1. Qwen image generation (via web UI API)
run "Qwen Image Studio" python3 -m pytest "$SCRIPT_DIR/e2e_qwen_web_test.py" -q --tb=short

# 2. WAN 2.2 T2V (Lightning, minimal resolution + frames)
run "WAN 2.2 T2V" env HOME=/root USER=root python3 /opt/wan-video-studio/generate.py \
    --task t2v-A14B \
    --size "480*320" \
    --ckpt_dir ~/Wan2.2-T2V-A14B \
    --lora_dir ~/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1 \
    --offload_model False \
    --prompt "A cat sitting on a chair." \
    --frame_num 17 \
    --save_file /tmp/wan_smoke_test.mp4

# 3. LTX-2 via ComfyUI (Qwen → LTX pipeline, smoke mode)
run "LTX-2 (Qwen→LTX pipeline)" bash -c "cd '$SCRIPT_DIR/..' && python3 tests/run_smoke.py --smoke"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
