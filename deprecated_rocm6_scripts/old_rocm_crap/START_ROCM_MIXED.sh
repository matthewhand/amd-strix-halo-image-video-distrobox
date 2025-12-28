#!/bin/bash

# Start AMD Strix Halo with Mixed GPU/CPU mode
# GPU for diffusion, CPU for text encoding to avoid HIP errors

set -e

echo "🚀 Starting AMD Strix Halo - MIXED MODE (GPU + CPU fallback)"
echo "==========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎯 Mixed Mode Configuration:${NC}"
echo "   🖼️  Qwen Image Studio: GPU acceleration"
echo "   🎬 ComfyUI: GPU for diffusion + CPU for text encoding"
echo "   ⚡  Fast video generation with stable text processing"
echo "   🛡️  Avoids HIP embedding errors on Strix Halo"

echo ""
echo -e "${YELLOW}🔄 Stopping any existing containers...${NC}"
docker rm -f strix-halo-distrobox 2>/dev/null || true

echo ""
echo -e "${BLUE}🚀 Starting mixed-mode container...${NC}"

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
    /bin/bash -c "
        # Start Qwen Image Studio (GPU)
        cd /opt/QwenImageStudio && python app.py --listen 0.0.0.0 --port 8000 &

        # Start ComfyUI with mixed mode (GPU for diffusion, CPU for text)
        cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 --cpu --output-directory /opt/ComfyUI/output &
    "

echo ""
echo -e "${GREEN}✅ Mixed Mode container started!${NC}"
echo ""
echo -e "${BLUE}🌐 Services available at:${NC}"
echo "   🖼️  Qwen Image Studio: http://localhost:8000"
echo "   🎬 ComfyUI (Mixed Mode): http://localhost:8188"
echo ""
echo -e "${BLUE}💡 Mixed Mode Benefits:${NC}"
echo "   ⚡  Fast GPU diffusion (video generation)"
echo "   🛡️  Stable CPU text encoding (no HIP errors)"
echo "   🔥  Best of both worlds for Strix Halo"
echo ""
echo -e "${BLUE}📝 Workflow available at:${NC}"
echo "   http://localhost:8188"
echo "   Load: /opt/ComfyUI/wan_i2v_workflow.json"

echo ""
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"

# Wait for both services
for i in {1..30}; do
    if curl -s http://localhost:8000 > /dev/null && curl -s http://localhost:8188 > /dev/null; then
        echo -e "${GREEN}🎉 SUCCESS! Services are running:${NC}"
        echo -e "   🖼️  Qwen Image Studio: http://localhost:8000"
        echo -e "   🎬 ComfyUI: http://localhost:8188 (mixed mode)"
        echo ""
        echo -e "${BLUE}💡 Your workflow is available at: http://localhost:8188${NC}"
        echo -e "   Load: /opt/ComfyUI/wan_i2v_workflow.json"
        echo ""
        echo -e "${GREEN}🚀 Strix Halo Mixed Mode - Fast & Stable!${NC}"
        exit 0
    fi

    if [ $i -eq 15 ]; then
        echo -e "${YELLOW}⏳ Still starting... (15/30 seconds)${NC}"
    fi

    sleep 1
done

echo -e "${RED}❌ Error: Services failed to start properly${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-distrobox${NC}"
exit 1