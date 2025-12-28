#!/bin/bash

# CONSERVATIVE WORKING CONFIGURATION - Based on ac1aba1 commit
# max_split_size_mb:256 + GTT unified memory patches

set -e

echo "🚀 STARTING CONSERVATIVE WORKING CONFIGURATION"
echo "==============================================="
echo "🔧 Based on commit ac1aba1 - proven GPU stable"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Remove all existing containers
docker rm -f strix-halo-working 2>/dev/null || true
docker rm -f strix-halo-distrobox 2>/dev/null || true
docker rm -f strix-halo-distrobox-working 2>/dev/null || true

# Start with ROCm 6.0 conservative settings that were working
docker run -d \
    --name strix-halo-working \
    --device=/dev/dri \
    --device=/dev/kfd \
    --ipc=host \
    --network host \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8,expandable_segments:False" \
    -e AMD_SERIALIZE_KERNEL=1 \
    -e TORCH_USE_HIP_DSA=1 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE \
    -e TRITON_HIP_LLD_PATH="/opt/rocm-6.0.0/bin/ld.lld" \
    -e TRITON_HIP_CLANG_PATH="/opt/rocm-6.0.0/bin/clang++" \
    -e LD_LIBRARY_PATH="/opt/rocm-6.0.0/lib:$LD_LIBRARY_PATH" \
    -e GPU_MEMORY_FRACTION=0.8 \
    -e HSA_ENABLE_SDMA=0 \
    -e HSA_DISABLE_FRAGMENT=1 \
    -v /home/matthewh/comfy-models:/opt/ComfyUI/models \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/ComfyUI_output:/opt/ComfyUI/output \
    amd-strix-halo-image-video-toolbox:rocm6.1 \
    /bin/bash -c '
source /opt/venv/bin/activate

echo "🔧 Conservative ROCm Configuration:"
echo "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "   PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "   FLASH_ATTENTION_TRITON_AMD_ENABLE: $FLASH_ATTENTION_TRITON_AMD_ENABLE"
echo "   ROCm Version: Using ROCm 6.0 (conservative)"

# Start Qwen Image Studio
echo "🖼️ Starting Qwen Image Studio..."
cd /opt/qwen-image-studio
export PYTHONPATH=/opt/qwen-image-studio/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
QWEN_PID=$!

# Start ComfyUI with conservative settings
echo "🎬 Starting ComfyUI (Conservative)..."
cd /opt/ComfyUI
echo "✅ Starting with conservative memory settings"
python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output &
COMFY_PID=$!

echo "✅ Both services started with conservative configuration:"
echo "   🖼️ Qwen Image Studio: http://localhost:8000 (PID: $QWEN_PID)"
echo "   🎬 ComfyUI (Conservative): http://localhost:8188 (PID: $COMFY_PID)"
echo "   🔧 Conservative ROCm 6.0 with GTT memory patches"
echo "   🛡️ max_split_size_mb:256 (conservative)"
echo "   ⚡ Flash Attention: DISABLED (avoid crashes)"

wait
'

echo ""
echo -e "${YELLOW}⏳ Waiting for conservative services to start...${NC}"

# Wait for services to start
for i in {1..60}; do
    QWEN_READY=$(curl -s http://localhost:8000 > /dev/null 2>&1 && echo "yes" || echo "no")
    COMFYUI_READY=$(curl -s http://localhost:8188 > /dev/null 2>&1 && echo "yes" || echo "no")

    if [ "$QWEN_READY" = "yes" ] && [ "$COMFYUI_READY" = "yes" ]; then
        echo -e "${GREEN}✅ CONSERVATIVE WORKING CONFIGURATION SUCCESSFUL!${NC}"
        echo ""
        echo -e "${BLUE}🚀 Conservative Setup is Running:${NC}"
        echo "   🖼️ Qwen Image Studio: http://localhost:8000"
        echo "   🎬 ComfyUI (Conservative): http://localhost:8188"
        echo ""
        echo -e "${GREEN}🔧 This is the PROVEN WORKING CONFIG:${NC}"
        echo "   🔧 ROCm 6.0 (conservative, proven)"
        echo "   🛡️ max_split_size_mb:256 (prevents crashes)"
        echo "   ⚡ Flash Attention: DISABLED"
        echo "   💾 GTT unified memory patches active"
        echo "   🧠 ROCm 5.7-style stability"
        echo ""
        echo -e "${BLUE}💡 Try generating now - should use actual GPU!${NC}"
        exit 0
    fi

    if [ $i -eq 20 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (20/60 seconds)${NC}"
    elif [ $i -eq 40 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (40/60 seconds)${NC}"
    fi

    sleep 1
done

echo -e "${RED}❌ Error: Conservative configuration failed to start${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-working${NC}"
exit 1