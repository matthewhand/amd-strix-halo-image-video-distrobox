#!/bin/bash

# AMD Strix Halo Fast Distrobox Launcher
# Provides both CPU (stable) and GPU (experimental) modes

MODE=${1:-cpu}  # Default to CPU mode for stability

echo "🚀 Starting AMD Strix Halo Distrobox..."
echo "   Mode: $MODE"
echo ""

# Stop any existing container
if docker ps -q -f name=strix-halo-distrobox | grep -q .; then
    echo "⏹️  Stopping existing container..."
    docker stop strix-halo-distrobox
    docker rm strix-halo-distrobox
fi

if [ "$MODE" = "gpu" ]; then
    echo "🔧 Starting GPU-ACCELERATED container (experimental - may crash)..."
    docker run -d --name strix-halo-distrobox \
        --device /dev/dri --device /dev/kfd \
        -p 8000:8000 -p 8188:8188 \
        -v /home/matthewh/comfy-models:/opt/ComfyUI/models \
        -v /home/matthewh/amd-strix-halo-image-video-toolboxes/examples/wan_i2v_workflow.json:/opt/ComfyUI/wan_i2v_workflow.json \
        amd-strix-halo-image-video-toolbox:rocm6.1 \
        /bin/bash -c "
        export HSA_OVERRIDE_GFX_VERSION=11.5.1
        export AMD_SERIALIZE_KERNEL=1
        export TORCH_USE_HIP_DSA=1
        export HIP_VISIBLE_DEVICES=0
        source /opt/venv/bin/activate

        # Start Qwen Image Studio
        cd /opt/qwen-image-studio
        uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
        QWEN_PID=\$!

        # Start ComfyUI with GPU acceleration (experimental)
        cd /opt/ComfyUI
        echo 'Starting ComfyUI with GPU acceleration (experimental)...'
        python main.py --listen 0.0.0.0 --port 8188 --gpu-only --output-directory /opt/ComfyUI/output &
        COMFY_PID=\$!

        echo '✅ GPU Services started (EXPERIMENTAL - may crash on text encoding):'
        echo '   Qwen Image Studio: http://localhost:8000 (PID: '\$QWEN_PID')'
        echo '   ComfyUI (GPU): http://localhost:8188 (PID: '\$COMFY_PID')'
        echo '   WARNING: GPU mode may crash during WAN video generation'
        echo '   💡 If it crashes, use: ./start_distrobox_fast.sh cpu'

        wait
    "

    echo ""
    echo "⚠️  GPU Mode Started (Experimental)"
    echo "   🖼️  Qwen Image Studio: http://localhost:8000"
    echo "   🎬 ComfyUI (GPU): http://localhost:8188"
    echo "   ⚡ Fast but may crash during text processing"
    echo "   💡 Fallback: ./start_distrobox_fast.sh cpu"

else
    echo "🔧 Starting CPU-STABLE container (recommended)..."
    docker run -d --name strix-halo-distrobox \
        --device /dev/dri --device /dev/kfd \
        -p 8000:8000 -p 8188:8188 \
        -v /home/matthewh/comfy-models:/opt/ComfyUI/models \
        -v /home/matthewh/amd-strix-halo-image-video-toolboxes/examples/wan_i2v_workflow.json:/opt/ComfyUI/wan_i2v_workflow.json \
        amd-strix-halo-image-video-toolbox:rocm6.1 \
        /bin/bash -c "
        export HSA_OVERRIDE_GFX_VERSION=11.5.1
        source /opt/venv/bin/activate

        # Start Qwen Image Studio
        cd /opt/qwen-image-studio
        uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
        QWEN_PID=\$!

        # Start ComfyUI with CPU text processing (stable)
        cd /opt/ComfyUI
        echo 'Starting ComfyUI with stable CPU mode...'
        python main.py --listen 0.0.0.0 --port 8188 --cpu --output-directory /opt/ComfyUI/output &
        COMFY_PID=\$!

        echo '✅ CPU-STABLE Services started:'
        echo '   Qwen Image Studio: http://localhost:8000 (PID: '\$QWEN_PID')'
        echo '   ComfyUI (CPU): http://localhost:8188 (PID: '\$COMFY_PID')'
        echo '   Mode: Stable CPU processing - no crashes'
        echo '   🎯 Slower but 100% reliable for WAN video generation'

        wait
    "

    echo ""
    echo "✅ CPU Mode Started (Stable)"
    echo "   🖼️  Qwen Image Studio: http://localhost:8000"
    echo "   🎬 ComfyUI (CPU): http://localhost:8188"
    echo "   🛡️  Stable - no crashes during video generation"
    echo "   ⏱️  Slower but reliable"
fi

echo ""
echo "⏳ Waiting for services to start..."
sleep 15

# Check if services are running
if netstat -tlnp | grep -q ':8000.*LISTEN' && netstat -tlnp | grep -q ':8188.*LISTEN'; then
    echo ""
    echo "🎉 SUCCESS! Services are running:"
    echo "   🖼️  Qwen Image Studio: http://localhost:8000"
    echo "   🎬 ComfyUI: http://localhost:8188 ($MODE mode)"
    echo ""
    echo "💡 Your workflow is available at: http://localhost:8188"
    echo "   Load: /opt/ComfyUI/wan_i2v_workflow.json"

    if [ "$MODE" = "cpu" ]; then
        echo ""
        echo "⏱️  CPU Mode Performance Tips:"
        echo "   - Slower but stable for video generation"
        echo "   - Uses your 128GB unified memory efficiently"
        echo "   - No crashes during text encoding"
        echo "   - For GPU speed: ./start_distrobox_fast.sh gpu (experimental)"
    else
        echo ""
        echo "⚡ GPU Mode Warning:"
        echo "   - Faster processing but may crash on text encoding"
        echo "   - If crashes occur, restart with: ./start_distrobox_fast.sh cpu"
    fi
else
    echo "❌ Error: Services failed to start properly"
    echo "Check logs with: docker logs strix-halo-distrobox"
    exit 1
fi