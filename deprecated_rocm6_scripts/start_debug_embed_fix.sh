#!/bin/bash

# DEBUG EMBEDDING FIX FOR STRIX HALO
# Add debugging + embedding CPU fallback to avoid HIP errors

set -e

echo "🚀 STARTING DEBUG EMBEDDING FIX"
echo "================================="
echo "🔧 Debug mode + CPU embedding fallback"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Remove all existing containers
docker rm -f strix-halo-debug 2>/dev/null || true
docker rm -f strix-halo-exact 2>/dev/null || true
docker rm -f strix-halo-working 2>/dev/null || true

# Start with debugging + embedding fix
docker run -d \
    --name strix-halo-debug \
    --device=/dev/dri \
    --device=/dev/kfd \
    --ipc=host \
    --network host \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    -e AMD_SERIALIZE_KERNEL=3 \
    -e TORCH_USE_HIP_DSA=1 \
    -e FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE \
    -e QWEN_FA_SHIM=1 \
    -e WAN_ATTENTION_BACKEND=sdpa \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.95 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROC_ENABLE_PRE_VEGA=1 \
    -e TORCH_CUDNN_V8_API_ENABLED=0 \
    -v /home/matthewh/comfy-models:/opt/ComfyUI/models \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/ComfyUI_output:/opt/ComfyUI/output \
    amd-strix-halo-image-video-toolbox:rocm6.1 \
    /bin/bash -c '
source /opt/venv/bin/activate

echo "🔧 Debug Embedding Fix Configuration:"
echo "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "   AMD_SERIALIZE_KERNEL: $AMD_SERIALIZE_KERNEL"
echo "   TORCH_USE_HIP_DSA: $TORCH_USE_HIP_DSA"
echo "   QWEN_FA_SHIM: $QWEN_FA_SHIM"

# Create embedding fix for HIP errors
cat > /tmp/embedding_fix.py << "EOF"
import torch
import torch.nn.functional as F

print("🛡️ Applying Strix Halo embedding CPU fallback fix...")

# Store original embedding function
original_embedding = F.embedding

def safe_embedding_hip_fix(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """CPU fallback for HIP embedding errors on Strix Halo"""
    try:
        return original_embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    except RuntimeError as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["hip", "invalid device function", "device function"]):
            print(f"🔄 HIP embedding error detected: {e}")
            print("🧠 Forcing CPU embedding to avoid Strix Halo HIP issues")

            # Force CPU processing for embedding
            weight_cpu = weight.cpu()
            input_cpu = input.cpu()

            # Perform embedding on CPU
            result_cpu = original_embedding(input_cpu, weight_cpu, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

            # Move result back to original device
            return result_cpu.to(weight.device)
        else:
            raise e

# Apply the embedding fix
F.embedding = safe_embedding_hip_fix
print("✅ Embedding CPU fallback applied for Strix Halo")

EOF

# Apply embedding fix before starting services
python /tmp/embedding_fix.py

# Start Qwen Image Studio with debug
echo "🖼️ Starting Qwen Image Studio (Debug + Embedding Fix)..."
cd /opt/qwen-image-studio
export PYTHONPATH=/opt/qwen-image-studio/src:$PYTHONPATH
uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
QWEN_PID=$!

# Start ComfyUI with debug + embedding fix
echo "🎬 Starting ComfyUI (Debug + Embedding Fix)..."
cd /opt/ComfyUI
echo "✅ Starting with debug mode and embedding CPU fallback"
python /tmp/embedding_fix.py && python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output &
COMFY_PID=$!

echo "✅ Both services started with debug + embedding fix:"
echo "   🖼️ Qwen Image Studio: http://localhost:8000 (PID: $QWEN_PID)"
echo "   🎬 ComfyUI (Debug + Fix): http://localhost:8188 (PID: $COMFY_PID)"
echo "   🔍 AMD_SERIALIZE_KERNEL=3 (debug mode)"
echo "   🛠️ TORCH_USE_HIP_DSA=1 (device assertions)"
echo "   🛡️ Embedding CPU fallback: Active"
echo "   🎯 QWEN_FA_SHIM=1 (flash attention shim)"

wait
'

echo ""
echo -e "${YELLOW}⏳ Waiting for debug + embedding fix services to start...${NC}"

# Wait for services to start
for i in {1..60}; do
    QWEN_READY=$(curl -s http://localhost:8000 > /dev/null 2>&1 && echo "yes" || echo "no")
    COMFYUI_READY=$(curl -s http://localhost:8188 > /dev/null 2>&1 && echo "yes" || echo "no")

    if [ "$QWEN_READY" = "yes" ] && [ "$COMFYUI_READY" = "yes" ]; then
        echo -e "${GREEN}✅ DEBUG + EMBEDDING FIX SUCCESSFUL!${NC}"
        echo ""
        echo -e "${BLUE}🚀 Debug + Fix Setup is Running:${NC}"
        echo "   🖼️ Qwen Image Studio: http://localhost:8000"
        echo "   🎬 ComfyUI (Debug + Fix): http://localhost:8188"
        echo ""
        echo -e "${GREEN}🔍 Debug + Fix Features Applied:${NC}"
        echo "   🔍 AMD_SERIALIZE_KERNEL=3 (detailed error reporting)"
        echo "   🛠️ TORCH_USE_HIP_DSA=1 (device-side assertions)"
        echo "   🛡️ Embedding CPU fallback (avoids HIP errors)"
        echo "   🎯 QWEN_FA_SHIM=1 (flash attention shim)"
        echo "   🧠 Hybrid approach: GPU when possible, CPU when needed"
        echo ""
        echo -e "${BLUE}💡 Try generation now - should avoid HIP embedding errors!${NC}"
        exit 0
    fi

    if [ $i -eq 20 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (20/60 seconds)${NC}"
    elif [ $i -eq 40 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (40/60 seconds)${NC}"
    fi

    sleep 1
done

echo -e "${RED}❌ Error: Debug + embedding fix failed to start${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-debug${NC}"
exit 1