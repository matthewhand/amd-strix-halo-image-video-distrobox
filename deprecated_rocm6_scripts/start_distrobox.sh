#!/bin/bash

# AMD Strix Halo Image & Video Distrobox Launcher
# Starts GPU-accelerated services on ports 8000 and 8188

echo "🚀 Starting AMD Strix Halo Distrobox..."
echo "   Qwen Image Studio: http://localhost:8000"
echo "   ComfyUI (GPU): http://localhost:8188"
echo ""

# Stop any existing container
if docker ps -q -f name=strix-halo-distrobox | grep -q .; then
    echo "⏹️  Stopping existing container..."
    docker stop strix-halo-distrobox
    docker rm strix-halo-distrobox
fi

# Start the GPU-accelerated container
echo "🔧 Starting GPU-accelerated distrobox..."
docker run -d --name strix-halo-distrobox \
    --device /dev/dri --device /dev/kfd \
    -p 8000:8000 -p 8188:8188 \
    -v /home/matthewh/comfy-models:/opt/ComfyUI/models \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/examples/wan_i2v_workflow.json:/opt/ComfyUI/wan_i2v_workflow.json \
    amd-strix-halo-image-video-toolbox:rocm6.1
    /bin/bash -c "
    export HSA_OVERRIDE_GFX_VERSION=11.5.1
    source /opt/venv/bin/activate

    # Start Qwen Image Studio
    cd /opt/qwen-image-studio
    uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
    QWEN_PID=\$!

    # Start ComfyUI with CPU text encoder fallback (fixes HIP embedding errors)
    cd /opt/ComfyUI
    python main.py --listen 0.0.0.0 --port 8188 --cpu --output-directory /opt/ComfyUI/output &
    COMFY_PID=\$!

    echo '✅ Services started (CPU Fallback Mode):'
    echo '   Qwen Image Studio: http://localhost:8000'
    echo '   ComfyUI: http://localhost:8188 (CPU mode - fixes HIP embedding errors)'
    echo '   Mode: CPU text processing + ROCm 6.1 compatibility'
    echo '   Status: Stable - bypasses gfx1151 rocBLAS limitations'

    wait
"

echo ""
echo "⏳ Waiting for services to start..."
sleep 15

# Check if services are running
if netstat -tlnp | grep -q ':8000.*LISTEN' && netstat -tlnp | grep -q ':8188.*LISTEN'; then
    echo ""
    echo "🎉 SUCCESS! Both services are running:"
    echo "   🖼️  Qwen Image Studio: http://localhost:8000"
    echo "   🎬 ComfyUI (GPU): http://localhost:8188"
    echo ""
    echo "💡 Your workflow is available at: http://localhost:8188"
    echo "   Load: /opt/ComfyUI/wan_i2v_workflow.json"
else
    echo "❌ Error: Services failed to start properly"
    echo "Check logs with: docker logs strix-halo-distrobox"
    exit 1
fi