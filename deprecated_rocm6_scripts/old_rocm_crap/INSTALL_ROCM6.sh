#!/bin/bash

# AMD ROCm 6.0 Direct Install Script
# Removes Ubuntu ROCm packages and installs AMD official ROCm 6.0

set -e

echo "🚀 AMD ROCm 6.0 Direct Install for Ubuntu 24.04"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Current ROCm status:${NC}"
if command -v rocm-smi &> /dev/null; then
    echo -e "   ${YELLOW}ROCm SMI found: $(rocm-smi --version 2>/dev/null | head -1 || echo 'Unknown version')${NC}"
else
    echo -e "   ${GREEN}No ROCm SMI detected${NC}"
fi

echo ""
echo -e "${BLUE}🗑️  Removing Ubuntu ROCm packages...${NC}"
sudo apt remove --purge -y rocm-smi librocm-smi64-1 || true
sudo apt autoremove -y || true

echo ""
echo -e "${BLUE}📦 Adding AMD official ROCm repository...${NC}"

# Add ROCm 6.0 repository
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Add AMD GPG key
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg

echo ""
echo -e "${BLUE}🔄 Updating package lists...${NC}"
sudo apt update

echo ""
echo -e "${BLUE}📦 Installing ROCm 6.0 packages...${NC}"
sudo apt install -y \
    rocm-dkms \
    rocm-dev \
    rocm-utils \
    rocm-smi \
    libhipblas-dev \
    librocblas-dev \
    libhipfft-dev \
    hipsparse-dev

echo ""
echo -e "${BLUE}🔧 Configuring ROCm for Strix Halo (gfx1151)...${NC}"

# Add ROCm to user environment
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/opencl/bin/x86_64' | tee -a ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' | tee -a ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=11.5.1' | tee -a ~/.bashrc

# Add ROCm udev rules
echo 'KERNEL=="kfd", MODE="0666"' | sudo tee /etc/udev/rules.d/99-kfd.rules

echo ""
echo -e "${GREEN}✅ ROCm 6.0 installation completed!${NC}"
echo ""
echo -e "${YELLOW}🔄 Please reboot your system to load the new ROCm drivers:${NC}"
echo "   sudo reboot"
echo ""
echo -e "${BLUE}📊 After reboot, verify installation:${NC}"
echo "   rocm-smi --showproductname"
echo "   rocm-smi --showmeminfo"
echo "   /opt/rocm/bin/rocminfo"
echo ""
echo -e "${GREEN}🎯 Expected results:${NC}"
echo "   - ROCm 6.0 with gfx1151 support"
echo "   - No more HIP embedding errors"
echo "   - Working GPU acceleration"
echo "   - Fast video generation"
echo ""
echo -e "${BLUE}🐳 After reboot, you can start the container with:${NC}"
echo "   ./start_distrobox_fast.sh gpu"
echo ""
echo -e "${YELLOW}⚠️  Make sure to rebuild the ROCm container to match host drivers:${NC}"
echo "   ./docker/build_rocm_container.sh 6.0"