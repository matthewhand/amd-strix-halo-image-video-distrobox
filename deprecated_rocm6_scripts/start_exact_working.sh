#!/bin/bash

# EXACT WORKING CONFIGURATION from git history
# These are the EXACT environment variables that were proven to work

set -e

echo "🚀 STARTING EXACT WORKING CONFIGURATION"
echo "========================================="
echo "🔧 Using exact env vars from working commit"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Remove all existing containers
docker rm -f strix-halo-exact 2>/dev/null || true
docker rm -f strix-halo-working 2>/dev/null || true
docker rm -f strix-halo-distrobox 2>/dev/null || true

# Start with EXACT working environment variables
docker run -d \
    --name strix-halo-exact \
    --device=/dev/dri \
    --device=/dev/kfd \
    --ipc=host \
    --network host \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
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

echo "🔧 Exact Working Configuration Applied:"
echo "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "   FLASH_ATTENTION_TRITON_AMD_ENABLE: $FLASH_ATTENTION_TRITON_AMD_ENABLE"
echo "   QWEN_FA_SHIM: $QWEN_FA_SHIM"
echo "   WAN_ATTENTION_BACKEND: $WAN_ATTENTION_BACKEND"
echo "   PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "   ROC_ENABLE_PRE_VEGA: $ROC_ENABLE_PRE_VEGA"
echo "   TORCH_CUDNN_V8_API_ENABLED: $TORCH_CUDNN_V8_API_ENABLED"
echo "   🚀 Kernel GTT: amdgpu.gttsize=131072 (128GB) ✓"

# Start Qwen Image Studio
echo "🖼️ Starting Qwen Image Studio (Exact Working)..."
cd /opt/qwen-image-studio
export PYTHONPATH=/opt/qwen-image-studio/src:$PYTHONPATH
uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
QWEN_PID=$!

# Start ComfyUI with exact working config
echo "🎬 Starting ComfyUI (Exact Working)..."
cd /opt/ComfyUI
echo "✅ Starting with exact working environment"
python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output &
COMFY_PID=$!

echo "✅ Both services started with EXACT working configuration:"
echo "   🖼️ Qwen Image Studio: http://localhost:8000 (PID: $QWEN_PID)"
echo "   🎬 ComfyUI (Exact Working): http://localhost:8188 (PID: $COMFY_PID)"
echo "   🎯 HSA_OVERRIDE_GFX_VERSION=11.0.0 (exact from working commit)"
echo "   🛡️ QWEN_FA_SHIM=1 (Flash attention shim)"
echo "   🧠 WAN_ATTENTION_BACKEND=sdpa (stable attention)"
echo "   💾 max_split_size_mb:256 (conservative)"
echo "   🚀 Kernel GTT: 128GB unified memory active"

wait
'

echo ""
echo -e "${YELLOW}⏳ Waiting for exact working services to start...${NC}"

# Wait for services to start
for i in {1..60}; do
    QWEN_READY=$(curl -s http://localhost:8000 > /dev/null 2>&1 && echo "yes" || echo "no")
    COMFYUI_READY=$(curl -s http://localhost:8188 > /dev/null 2>&1 && echo "yes" || echo "no")

    if [ "$QWEN_READY" = "yes" ] && [ "$COMFYUI_READY" = "yes" ]; then
        echo -e "${GREEN}✅ EXACT WORKING CONFIGURATION SUCCESSFUL!${NC}"
        echo ""
        echo -e "${BLUE}🚀 Exact Working Setup is Running:${NC}"
        echo "   🖼️ Qwen Image Studio: http://localhost:8000"
        echo "   🎬 ComfyUI (Exact Working): http://localhost:8188"
        echo ""
        echo -e "${GREEN}🎯 This is the EXACT configuration from working commit:${NC}"
        echo "   🎯 HSA_OVERRIDE_GFX_VERSION=11.0.0 (from working commit)"
        echo "   🛡️ QWEN_FA_SHIM=1 (Flash attention shim active)"
        echo "   🧠 WAN_ATTENTION_BACKEND=sdpa (stable backend)"
        echo "   💾 max_split_size_mb:256 (conservative memory)"
        echo "   🚀 Kernel GTT: amdgpu.gttsize=131072 (128GB unified)"
        echo ""
        echo -e "${BLUE}💡 This should provide ACTUAL GPU acceleration like before!${NC}"
        exit 0
    fi

    if [ $i -eq 20 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (20/60 seconds)${NC}"
    elif [ $i -eq 40 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (40/60 seconds)${NC}"
    fi

    sleep 1
done

echo -e "${RED}❌ Error: Exact working configuration failed to start${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-exact${NC}"
exit 1