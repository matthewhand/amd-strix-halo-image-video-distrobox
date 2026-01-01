#!/bin/bash
# AMD Strix Halo Diagnostic Tool
# Validates ROCm setup, GPU detection, and basic compute functionality

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 AMD Strix Halo Diagnostic Tool${NC}"
echo "=================================="
echo ""

# Detect if we're in container or on host
CONTAINER_NAME="${CONTAINER_NAME:-strix-fresh}"
IN_CONTAINER=false
if [ -f /.dockerenv ]; then
    IN_CONTAINER=true
fi

run_cmd() {
    if $IN_CONTAINER; then
        eval "$1"
    else
        docker exec "$CONTAINER_NAME" bash -c "$1"
    fi
}

echo -e "${BLUE}1. Host System Check${NC}"
echo "---------------------"
echo "Kernel: $(uname -r)"
echo "ROCk module: $(lsmod | grep -q amdgpu && echo "✅ Loaded" || echo "❌ Not loaded")"
echo ""

echo -e "${BLUE}2. GPU Detection${NC}"
echo "-----------------"
if command -v rocminfo &> /dev/null; then
    GPU_NAME=$(rocminfo 2>/dev/null | grep -A2 "Agent 2" | grep "Name:" | awk '{print $2}')
    GPU_ARCH=$(rocminfo 2>/dev/null | grep -A20 "Agent 2" | grep "Device Type:" | awk '{print $3}')
    echo "GPU: ${GPU_NAME:-Not detected}"
    echo "Arch: ${GPU_ARCH:-Unknown}"
else
    echo "rocminfo not available on host"
fi
echo ""

echo -e "${BLUE}3. Device Permissions${NC}"
echo "----------------------"
ls -la /dev/kfd 2>/dev/null | awk '{print "/dev/kfd: " $1 " " $3 ":" $4}' || echo "/dev/kfd: ❌ Missing"
ls -la /dev/dri/renderD128 2>/dev/null | awk '{print "/dev/dri/renderD128: " $1 " " $3 ":" $4}' || echo "/dev/dri/renderD128: ❌ Missing"
echo ""

echo -e "${BLUE}4. Container Check${NC}"
echo "-------------------"
if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Status}}" | grep -q "Up"; then
    echo "Container $CONTAINER_NAME: ✅ Running"
    
    echo ""
    echo -e "${BLUE}5. Container GPU Test${NC}"
    echo "----------------------"
    
    # Check HSA_OVERRIDE_GFX_VERSION
    HSA_VER=$(docker exec "$CONTAINER_NAME" bash -c 'echo $HSA_OVERRIDE_GFX_VERSION' 2>/dev/null)
    echo "HSA_OVERRIDE_GFX_VERSION: ${HSA_VER:-Not set}"
    if [ "$HSA_VER" != "11.5.1" ]; then
        echo -e "${YELLOW}⚠️  Should be 11.5.1 for gfx1151 (Strix Halo)${NC}"
    fi
    
    # PyTorch GPU test
    echo ""
    echo "Running PyTorch GPU test..."
    docker exec -e HSA_OVERRIDE_GFX_VERSION=11.5.1 "$CONTAINER_NAME" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA/HIP available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f'GPU matmul: ✅ Success')
    print(f'Memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB')
else:
    print('❌ GPU not available')
" 2>&1 || echo -e "${RED}❌ GPU test failed${NC}"

else
    echo "Container $CONTAINER_NAME: ❌ Not running"
    echo "Start with: docker compose up -d"
fi

echo ""
echo -e "${GREEN}Diagnostic complete.${NC}"
