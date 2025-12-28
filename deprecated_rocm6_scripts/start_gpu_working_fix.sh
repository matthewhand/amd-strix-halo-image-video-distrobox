#!/bin/bash

# WORKING GPU STARTUP - Flash Attention + Triton AMD Backend
# The original working configuration with GPU acceleration

set -e

echo "🚀 RESTORING WORKING GPU CONFIGURATION"
echo "======================================="
echo "🔧 Applying Triton AMD backend + ROCm fixes"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Remove existing container
docker rm -f strix-halo-distrobox-working 2>/dev/null || true
docker rm -f strix-halo-distrobox 2>/dev/null || true

# Start with working GPU configuration
docker run -d \
    --name strix-halo-distrobox-working \
    --device=/dev/dri \
    --device=/dev/kfd \
    --ipc=host \
    --network host \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8,expandable_segments:False" \
    -e AMD_SERIALIZE_KERNEL=1 \
    -e TORCH_USE_HIP_DSA=1 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    -e TRITON_HIP_LLD_PATH="/opt/rocm-6.0.0/bin/ld.lld" \
    -e TRITON_HIP_CLANG_PATH="/opt/rocm-6.0.0/bin/clang++" \
    -e LD_LIBRARY_PATH="/opt/rocm-6.0.0/lib:$LD_LIBRARY_PATH" \
    -v /home/matthewh/comfy-models:/opt/ComfyUI/models \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/ComfyUI_output:/opt/ComfyUI/output \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/docker/01-rocm-env-for-triton.sh:/tmp/rocm-env.sh \
    amd-strix-halo-image-video-toolbox:rocm6.1 \
    /bin/bash -c '
source /opt/venv/bin/activate

# Apply ROCm Triton environment
source /tmp/rocm-env.sh

echo "🔥 ROCm/Triton Configuration Applied:"
echo "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "   FLASH_ATTENTION_TRITON_AMD_ENABLE: $FLASH_ATTENTION_TRITON_AMD_ENABLE"
echo "   AMD_SERIALIZE_KERNEL: $AMD_SERIALIZE_KERNEL"

# Start Qwen Image Studio with GPU support
echo "🖼️ Starting Qwen Image Studio (GPU)..."
cd /opt/qwen-image-studio
export PYTHONPATH=/opt/qwen-image-studio/src:$PYTHONPATH
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
QWEN_PID=$!

# Start ComfyUI with GPU acceleration
echo "🎬 Starting ComfyUI (GPU)..."
cd /opt/ComfyUI
echo "✅ Starting with GPU acceleration and working ROCm configuration"
python main.py --listen 0.0.0.0 --port 8188 --gpu-only --output-directory /opt/ComfyUI/output &
COMFY_PID=$!

echo "✅ Both services started with working GPU configuration:"
echo "   🖼️ Qwen Image Studio: http://localhost:8000 (PID: $QWEN_PID)"
echo "   🎬 ComfyUI (GPU): http://localhost:8188 (PID: $COMFY_PID)"
echo "   🔥 GPU acceleration with Triton AMD backend"
echo "   ⚡ Working ROCm flash attention configuration"

wait
'

echo ""
echo -e "${YELLOW}⏳ Waiting for working GPU services to start...${NC}"

# Wait for services to start
for i in {1..60}; do
    QWEN_READY=$(curl -s http://localhost:8000 > /dev/null 2>&1 && echo "yes" || echo "no")
    COMFYUI_READY=$(curl -s http://localhost:8188 > /dev/null 2>&1 && echo "yes" || echo "no")

    if [ "$QWEN_READY" = "yes" ] && [ "$COMFYUI_READY" = "yes" ]; then
        echo -e "${GREEN}✅ WORKING GPU CONFIGURATION RESTORED!${NC}"
        echo ""
        echo -e "${BLUE}🚀 Your Original Working Setup is Back:${NC}"
        echo "   🖼️ Qwen Image Studio: http://localhost:8000"
        echo "   🎬 ComfyUI (GPU): http://localhost:8188"
        echo ""
        echo -e "${GREEN}🔧 This configuration includes:${NC}"
        echo "   🔥 GPU acceleration (Radeon 8060S)"
        echo "   ⚡ Triton AMD backend for flash attention"
        echo "   🛡️ ROCm 6.0 compatibility (working version)"
        echo "   📁 Working model paths"
        echo "   🎯 The exact setup that was working before"
        echo ""
        echo -e "${BLUE}💡 Try generating images and videos now - should work perfectly!${NC}"
        exit 0
    fi

    if [ $i -eq 20 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (20/60 seconds)${NC}"
    elif [ $i -eq 40 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (40/60 seconds)${NC}"
    fi

    sleep 1
done

echo -e "${RED}❌ Error: Working GPU configuration failed to start${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-distrobox${NC}"
exit 1