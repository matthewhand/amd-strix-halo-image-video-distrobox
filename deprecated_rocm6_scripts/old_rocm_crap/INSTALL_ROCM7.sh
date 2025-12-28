#!/bin/bash

# AMD ROCm 7.1 Direct Install Script
# Removes Ubuntu ROCm packages and installs AMD official ROCm 7.1
# Latest ROCm with best gfx1151 support for Strix Halo

set -e

echo "🚀 AMD ROCm 7.1 Direct Install for Ubuntu 24.04"
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
echo -e "${BLUE}⚠️  WARNING: ROCm 7.1 is cutting-edge!${NC}"
echo "   - Latest ROCm with optimal gfx1151 support"
echo "   - May have stability issues on some hardware"
echo "   - For production use, consider ROCm 6.0 first"
echo ""

read -p "Continue with ROCm 7.1 installation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled. Use ./INSTALL_ROCM6.sh for stable ROCm 6.0."
    exit 0
fi

echo ""
echo -e "${BLUE}🗑️  Removing existing ROCm packages...${NC}"
sudo apt remove --purge -y rocm-smi librocm-smi64-1 rocm-* hip-* miopen-* rccl-* || true
sudo apt autoremove -y || true

echo ""
echo -e "${BLUE}📦 Adding AMD official ROCm 7.1 repository...${NC}"

# Add ROCm 7.1 repository
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.1 noble main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Add AMD GPG key
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg

echo ""
echo -e "${BLUE}🔄 Updating package lists...${NC}"
sudo apt update

echo ""
echo -e "${BLUE}📦 Installing ROCm 7.1 packages...${NC}"
sudo apt install -y \
    rocm-dkms \
    rocm-dev \
    rocm-utils \
    rocm-smi \
    libhipblas-dev \
    librocblas-dev \
    libhipfft-dev \
    hipsparse-dev \
    miopen-hip \
    rccl-hip

echo ""
echo -e "${BLUE}🔧 Configuring ROCm 7.1 for Strix Halo (gfx1151)...${NC}"

# Add ROCm to user environment
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/opencl/bin/x86_64' | tee -a ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' | tee -a ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=11.5.1' | tee -a ~/.bashrc

# ROCm 7.1 specific environment variables
echo 'export HIP_VISIBLE_DEVICES=0' | tee -a ~/.bashrc
echo 'export MIOPEN_USER_DB_PATH=/opt/rocm/miopen' | tee -a ~/.bashrc

# Add ROCm udev rules
echo 'KERNEL=="kfd", MODE="0666"' | sudo tee /etc/udev/rules.d/99-kfd.rules

# Initialize MIOpen database (ROCm 7.1 specific)
if [ -f "/opt/rocm/bin/miopen" ]; then
    echo -e "${BLUE}🔄 Initializing MIOpen database...${NC}"
    /opt/rocm/bin/miopen -d 0 2>/dev/null || echo "   Note: MIOpen database will be initialized on first use"
fi

echo ""
echo -e "${GREEN}✅ ROCm 7.1 installation completed!${NC}"
echo ""
echo -e "${YELLOW}🔄 Please reboot your system to load the new ROCm drivers:${NC}"
echo "   sudo reboot"
echo ""
echo -e "${BLUE}📊 After reboot, verify installation:${NC}"
echo "   rocm-smi --showproductname"
echo "   rocm-smi --showmeminfo"
echo "   /opt/rocm/bin/rocminfo"
echo "   /opt/rocm/bin/hipconfig"
echo ""
echo -e "${GREEN}🎯 Expected results:${NC}"
echo "   - ROCm 7.1 with enhanced gfx1151 support"
echo "   - Latest HIP runtime optimizations"
echo "   - Improved MIOpen performance"
echo "   - No more HIP embedding errors"
echo "   - Maximum GPU acceleration"
echo ""
echo -e "${BLUE}🐳 After reboot, you can start the container with:${NC}"
echo "   ./start_distrobox_fast.sh gpu"
echo ""
echo -e "${YELLOW}⚠️  Make sure to rebuild the ROCm container to match host drivers:${NC}"
echo "   ./docker/build_rocm_container.sh 7.1"
echo ""
echo -e "${RED}🚨 If you experience stability issues:${NC}"
echo "   - Fall back to ROCm 6.0: ./INSTALL_ROCM6.sh"
echo "   - Report issues to AMD ROCm GitHub"