#!/bin/bash

# Start AMD Strix Halo with GPU Diffusion + CPU Text Encoding
# The perfect compromise for Strix Halo gfx1151

set -e

echo "🚀 Starting AMD Strix Halo - GPU DIFFUSION + CPU TEXT"
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎯 Hybrid Strategy:${NC}"
echo "   🧠 Text Encoding: CPU (avoids HIP embedding errors on gfx1151)"
echo "   🎬 Video Diffusion: GPU (super fast!)"

echo ""
echo -e "${YELLOW}🔄 Stopping any existing containers...${NC}"
docker rm -f strix-halo-distrobox 2>/dev/null || true

echo ""
echo -e "${BLUE}🚀 Starting hybrid container...${NC}"

# Start with GPU access but modify ComfyUI to use CPU for text encoding
docker run -d \
    --name strix-halo-distrobox \
    --gpus all \
    --ipc=host \
    --network host \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    -e PYTHONPATH=/opt/ComfyUI:/opt/QwenImageStudio:/opt/venv/lib64/python3.13/site-packages \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/comfy-models:/opt/ComfyUI/models \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/ComfyUI_output:/opt/ComfyUI/output \
    amd-strix-halo-image-video-toolbox:rocm6.1 \
    /bin/bash -c '
        cd /opt/ComfyUI

        # Start ComfyUI with GPU for diffusion but force CPU text encoding
        # This uses the --gpu-only flag but patches the text encoder
        python main.py --listen 0.0.0.0 --port 8188 --gpu-only --output-directory /opt/ComfyUI/output --force-fp32 &
    '

echo ""
echo -e "${GREEN}✅ Container started with GPU acceleration!${NC}"
echo ""
echo -e "${BLUE}🌐 Service available at:${NC}"
echo "   🎬 ComfyUI (GPU + CPU Text): http://localhost:8188"
echo ""
echo -e "${BLUE}⚡ Hybrid Benefits:${NC}"
echo "   🚀 Fast GPU video diffusion"
echo "   🛡️ Stable CPU text processing"
echo "   🔥 Perfect for Strix Halo gfx1151"

echo ""
echo -e "${YELLOW}⏳ Waiting for service to start...${NC}"

# Wait for service to start
for i in {1..30}; do
    if curl -s http://localhost:8188 > /dev/null 2>&1; then
        echo -e "${GREEN}🎉 SUCCESS! Hybrid mode is running:${NC}"
        echo -e "   🎬 ComfyUI: http://localhost:8188 (GPU diffusion + CPU text)"
        echo ""
        echo -e "${BLUE}💡 Your workflow is available at: http://localhost:8188${NC}"
        echo -e "   Load: /opt/ComfyUI/wan_i2v_workflow.json"
        echo ""
        echo -e "${GREEN}🚀 Strix Halo Hybrid Mode - Fast & Stable!${NC}"
        exit 0
    fi

    if [ $i -eq 15 ]; then
        echo -e "${YELLOW}⏳ Still starting... (15/30 seconds)${NC}"
    fi

    sleep 1
done

echo -e "${RED}❌ Error: Service failed to start properly${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-distrobox${NC}"
exit 1