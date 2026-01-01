#!/bin/bash
# AMD Strix Halo Services Startup Script
# Unified script to start Qwen Image Studio, ComfyUI, or both

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CONTAINER_NAME="${CONTAINER_NAME:-strix-fresh}"
IMAGE_NAME="${IMAGE_NAME:-amd-strix-halo-image-video-toolbox:final-working-distrobox}"

# Critical environment for Strix Halo gfx1151
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HIP_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,garbage_collection_threshold:0.8

show_help() {
    echo "AMD Strix Halo Services Startup"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all       Start all services (Qwen + ComfyUI) [default]"
    echo "  --qwen      Start Qwen Image Studio only (port 8000)"
    echo "  --comfyui   Start ComfyUI only (port 8188)"
    echo "  --wan       Start WAN Video Studio CLI environment"
    echo "  --shell     Open interactive shell in container"
    echo "  --stop      Stop running services"
    echo "  --status    Show service status"
    echo "  -h, --help  Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                  # Start all services"
    echo "  $0 --qwen           # Qwen only"
    echo "  $0 --stop           # Stop services"
}

check_container() {
    if ! docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}❌ Container $CONTAINER_NAME not running${NC}"
        echo "Start with: docker compose up -d"
        exit 1
    fi
}

start_qwen() {
    echo -e "${BLUE}🎨 Starting Qwen Image Studio on port 8000...${NC}"
    docker exec -d -e HSA_OVERRIDE_GFX_VERSION=11.5.1 "$CONTAINER_NAME" bash -c '
        cd /opt/qwen-image-studio
        pkill -f "uvicorn.*8000" 2>/dev/null || true
        sleep 1
        QIM_CLI_PATH=/opt/qwen-image-studio/qwen-image-mps.py \
        uvicorn qwen_image_studio.server:app --host 0.0.0.0 --port 8000 &
    '
    echo -e "${GREEN}✅ Qwen Image Studio: http://localhost:8000${NC}"
}

start_comfyui() {
    echo -e "${BLUE}🖼️  Starting ComfyUI on port 8188...${NC}"
    docker exec -d -e HSA_OVERRIDE_GFX_VERSION=11.5.1 "$CONTAINER_NAME" bash -c '
        cd /opt/ComfyUI
        pkill -f "python main.py" 2>/dev/null || true
        sleep 1
        python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output &
    '
    echo -e "${GREEN}✅ ComfyUI: http://localhost:8188${NC}"
}

start_wan_shell() {
    echo -e "${BLUE}🎬 Opening WAN Video Studio shell...${NC}"
    echo -e "${YELLOW}Use: cd /opt/wan-video-studio && python generate.py --help${NC}"
    docker exec -it -e HSA_OVERRIDE_GFX_VERSION=11.5.1 "$CONTAINER_NAME" bash
}

show_status() {
    echo -e "${BLUE}📊 Service Status${NC}"
    echo "=================="
    
    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Status}}" | grep -q "Up"; then
        echo "Container: ✅ Running"
        
        # Check Qwen
        if docker exec "$CONTAINER_NAME" pgrep -f "uvicorn.*8000" >/dev/null 2>&1; then
            echo "Qwen Image Studio: ✅ Running (port 8000)"
        else
            echo "Qwen Image Studio: ⚪ Stopped"
        fi
        
        # Check ComfyUI
        if docker exec "$CONTAINER_NAME" pgrep -f "python main.py" >/dev/null 2>&1; then
            echo "ComfyUI: ✅ Running (port 8188)"
        else
            echo "ComfyUI: ⚪ Stopped"
        fi
    else
        echo "Container: ❌ Not running"
    fi
}

stop_services() {
    echo -e "${YELLOW}🛑 Stopping services...${NC}"
    docker exec "$CONTAINER_NAME" pkill -f "uvicorn.*8000" 2>/dev/null || true
    docker exec "$CONTAINER_NAME" pkill -f "python main.py" 2>/dev/null || true
    echo -e "${GREEN}✅ Services stopped${NC}"
}

# Parse arguments
MODE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)      MODE="all"; shift ;;
        --qwen)     MODE="qwen"; shift ;;
        --comfyui)  MODE="comfyui"; shift ;;
        --wan)      MODE="wan"; shift ;;
        --shell)    MODE="shell"; shift ;;
        --stop)     MODE="stop"; shift ;;
        --status)   MODE="status"; shift ;;
        -h|--help)  show_help; exit 0 ;;
        *)          echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Execute
case $MODE in
    all)
        check_container
        start_qwen
        start_comfyui
        echo ""
        echo -e "${GREEN}🚀 All services started!${NC}"
        ;;
    qwen)
        check_container
        start_qwen
        ;;
    comfyui)
        check_container
        start_comfyui
        ;;
    wan|shell)
        check_container
        start_wan_shell
        ;;
    stop)
        check_container
        stop_services
        ;;
    status)
        show_status
        ;;
esac
