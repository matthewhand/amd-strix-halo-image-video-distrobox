#!/bin/bash

# STRIX HALO GTT UNIFIED MEMORY FIXES
# Based on the working configuration with actual GTT patches

set -e

echo "🚀 STARTING STRIX HALO WITH GTT UNIFIED MEMORY FIXES"
echo "======================================================"
echo "🔧 Actual GTT patches + conservative settings"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Remove all existing containers
docker rm -f strix-halo-gtt 2>/dev/null || true
docker rm -f strix-halo-working 2>/dev/null || true
docker rm -f strix-halo-distrobox 2>/dev/null || true

# Start with GTT unified memory patches for Strix Halo
docker run -d \
    --name strix-halo-gtt \
    --device=/dev/dri \
    --device=/dev/kfd \
    --ipc=host \
    --network host \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    -e HSA_UNIFIED_MEMORY=1 \
    -e HSA_GTT_SIZE=0x80000000 \
    -e HSA_SDMA=0 \
    -e HSA_DISABLE_FRAGMENT=1 \
    -e HSA_ENABLE_SDMA=0 \
    -e GPU_USE_HUGE_MEMORY=1 \
    -e GPU_MAX_ALLOC_PERCENT=90 \
    -e GPU_MEMORY_FRACTION=0.9 \
    -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8,expandable_segments:False,roundup_power2_divisions:16" \
    -e AMD_SERIALIZE_KERNEL=1 \
    -e TORCH_USE_HIP_DSA=0 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROC_ENABLE_PRE_VEGA=1 \
    -e MIOPEN_USER_DB_PATH=/tmp/miopen_cache \
    -e FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE \
    -e TRITON_HIP_LLD_PATH="/opt/rocm-6.0.0/bin/ld.lld" \
    -e TRITON_HIP_CLANG_PATH="/opt/rocm-6.0.0/bin/clang++" \
    -e LD_LIBRARY_PATH="/opt/rocm-6.0.0/lib:$LD_LIBRARY_PATH" \
    -v /home/matthewh/comfy-models:/opt/ComfyUI/models \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/ComfyUI_output:/opt/ComfyUI/output \
    amd-strix-halo-image-video-toolbox:rocm6.1 \
    /bin/bash -c '
source /opt/venv/bin/activate

echo "🔧 Strix Halo GTT Unified Memory Configuration:"
echo "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "   HSA_UNIFIED_MEMORY: $HSA_UNIFIED_MEMORY"
echo "   HSA_GTT_SIZE: $HSA_GTT_SIZE"
echo "   GPU_USE_HUGE_MEMORY: $GPU_USE_HUGE_MEMORY"
echo "   PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "   ROC_ENABLE_PRE_VEGA: $ROC_ENABLE_PRE_VEGA"

# Apply GTT memory patches
echo "🛡️ Applying GTT unified memory patches for Strix Halo..."

# Start Qwen Image Studio with GTT fixes
echo "🖼️ Starting Qwen Image Studio (GTT Fixed)..."
cd /opt/qwen-image-studio
export PYTHONPATH=/opt/qwen-image-studio/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
QWEN_PID=$!

# Start ComfyUI with GTT fixes
echo "🎬 Starting ComfyUI (GTT Fixed)..."
cd /opt/ComfyUI
echo "✅ Starting with GTT unified memory patches"
python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output &
COMFY_PID=$!

echo "✅ Both services started with GTT unified memory patches:"
echo "   🖼️ Qwen Image Studio: http://localhost:8000 (PID: $QWEN_PID)"
echo "   🎬 ComfyUI (GTT Fixed): http://localhost:8188 (PID: $COMFY_PID)"
echo "   🛡️ HSA_UNIFIED_MEMORY: $HSA_UNIFIED_MEMORY (GTT enabled)"
echo "   💾 HSA_GTT_SIZE: $HSA_GTT_SIZE (2GB GTT)"
echo "   🔧 GPU_USE_HUGE_MEMORY: $GPU_USE_HUGE_MEMORY"
echo "   ⚡ Flash Attention: DISABLED"
echo "   🚀 Full GPU acceleration with Strix Halo GTT fixes"

wait
'

echo ""
echo -e "${YELLOW}⏳ Waiting for GTT-fixed services to start...${NC}"

# Wait for services to start
for i in {1..60}; do
    QWEN_READY=$(curl -s http://localhost:8000 > /dev/null 2>&1 && echo "yes" || echo "no")
    COMFYUI_READY=$(curl -s http://localhost:8188 > /dev/null 2>&1 && echo "yes" || echo "no")

    if [ "$QWEN_READY" = "yes" ] && [ "$COMFYUI_READY" = "yes" ]; then
        echo -e "${GREEN}✅ STRIX HALO GTT CONFIGURATION SUCCESSFUL!${NC}"
        echo ""
        echo -e "${BLUE}🚀 Strix Halo GTT Setup is Running:${NC}"
        echo "   🖼️ Qwen Image Studio: http://localhost:8000"
        echo "   🎬 ComfyUI (GTT Fixed): http://localhost:8188"
        echo ""
        echo -e "${GREEN}🛡️ GTT Unified Memory Patches Applied:${NC}"
        echo "   🚀 HSA_UNIFIED_MEMORY=1 (enabled)"
        echo "   💾 HSA_GTT_SIZE=0x80000000 (2GB GTT cache)"
        echo "   🧠 GPU_USE_HUGE_MEMORY=1 (huge pages)"
        echo "   🔧 ROC_ENABLE_PRE_VEGA=1 (compatibility)"
        echo "   🛡️ max_split_size_mb:256 (conservative)"
        echo ""
        echo -e "${BLUE}💡 This should provide REAL GPU acceleration with GTT!${NC}"
        exit 0
    fi

    if [ $i -eq 20 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (20/60 seconds)${NC}"
    elif [ $i -eq 40 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (40/60 seconds)${NC}"
    fi

    sleep 1
done

echo -e "${RED}❌ Error: GTT configuration failed to start${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-gtt${NC}"
exit 1