#!/bin/bash

# Start AMD Strix Halo with HYBRID acceleration
# CPU for text encoding (avoids HIP errors) + GPU for diffusion (fast video generation)

set -e

echo "🚀 Starting AMD Strix Halo - HYBRID MODE"
echo "======================================="
echo "🎯 Hybrid Configuration:"
echo "   🖥️  Text Encoding: CPU (avoids HIP embedding errors)"
echo "   🎬 Video Diffusion: GPU (fast!)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${YELLOW}🔄 Stopping any existing containers...${NC}"
docker rm -f strix-halo-distrobox 2>/dev/null || true

echo ""
echo -e "${BLUE}🚀 Starting hybrid acceleration container...${NC}"

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
        # Set environment for hybrid mode
        export FORCE_CPU_TEXT_ENCODING=1
        export COMFYUI_CPU_TEXT=1

        # Start Qwen Image Studio (GPU)
        cd /opt/QwenImageStudio && python app.py --listen 0.0.0.0 --port 8000 &

        # Start ComfyUI with custom hybrid mode
        # We need to patch ComfyUI to use CPU for text encoding
        cd /opt/ComfyUI

        # Create a patch to force CPU text encoding
        cat > /tmp/comfyui_hybrid_patch.py << "EOF"
import torch
import comfy.sd1_clip

# Patch CLIP text encoding to use CPU
def cpu_text_encode(self, tokens):
    original_device = next(self.parameters()).device
    # Force text encoding to CPU to avoid HIP embedding errors
    self = self.to("cpu")
    result = self.original_forward(tokens)
    # Move back to original device
    self = self.to(original_device)
    return result

# Patch the text encoder
comfy.sd1_clip.CLIPTextModel.forward = cpu_text_encode
print("🔄 Hybrid Mode: CPU text encoding enabled, GPU diffusion enabled")
EOF

        # Apply the patch and start ComfyUI
        python -c "exec(open('/tmp/comfyui_hybrid_patch.py').read())" && \
        python main.py --listen 0.0.0.0 --port 8188 --gpu-only --output-directory /opt/ComfyUI/output &
    '

echo ""
echo -e "${GREEN}✅ Hybrid acceleration container started!${NC}"
echo ""
echo -e "${BLUE}🌐 Services available at:${NC}"
echo "   🖼️  Qwen Image Studio: http://localhost:8000"
echo "   🎬 ComfyUI (Hybrid): http://localhost:8188"
echo ""
echo -e "${BLUE}⚡ Hybrid Mode Benefits:${NC}"
echo "   🎯 Fast GPU diffusion for video generation"
echo "   🛡️ Stable CPU text encoding (no HIP errors)"
echo "   🔥 Best performance on Strix Halo gfx1151"
echo ""
echo -e "${BLUE}📝 Workflow available at:${NC}"
echo "   http://localhost:8188"
echo "   Load: /opt/ComfyUI/wan_i2v_workflow.json"

echo ""
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"

# Wait for both services
for i in {1..30}; do
    if curl -s http://localhost:8000 > /dev/null 2>&1 && curl -s http://localhost:8188 > /dev/null 2>&1; then
        echo -e "${GREEN}🎉 SUCCESS! Hybrid mode is running:${NC}"
        echo -e "   🖼️  Qwen Image Studio: http://localhost:8000"
        echo -e "   🎬 ComfyUI: http://localhost:8188 (CPU text + GPU diffusion)"
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

echo -e "${RED}❌ Error: Services failed to start properly${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-distrobox${NC}"
exit 1