#!/bin/bash

# Clean ROCm 6.0 Installation Script
# Removes all existing ROCm packages and installs fresh ROCm 6.0

set -e

echo "🚀 Clean ROCm 6.0 Installation for Ubuntu 24.04"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗑️  Removing ALL existing ROCm packages...${NC}"

# Remove ALL ROCm-related packages
sudo apt-get remove --purge -y \
    rocm-* \
    hip-* \
    miopen-* \
    rccl-* \
    rocblas-* \
    hipblas-* \
    rocfft-* \
    hipsparse-* \
    rocminfo \
    rocm-smi \
    librocm-smi64-* \
    kmod-rocm \
    rock-dkms \
    amdgpu-pro \
    amdgpu-dkms || true

# Clean up any remaining packages
sudo apt-get autoremove -y || true
sudo apt-get autoclean || true

echo ""
echo -e "${BLUE}🧹 Cleaning package cache...${NC}"
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get clean

echo ""
echo -e "${BLUE}📦 Adding AMD official ROCm 6.0 repository...${NC}"

# Remove any existing ROCm repository files
sudo rm -f /etc/apt/sources.list.d/rocm.list
sudo rm -f /etc/apt/trusted.gpg.d/rocm.gpg

# Add ROCm 6.0 repository
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Add AMD GPG key
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg

echo ""
echo -e "${BLUE}🔄 Updating package lists...${NC}"
sudo apt-get update

echo ""
echo -e "${BLUE}📦 Installing ROCm 6.0 core packages...${NC}"

# Install minimal ROCm 6.0 packages first
sudo apt-get install -y \
    rocm-dev \
    rocm-utils \
    rocm-smi

echo ""
echo -e "${BLUE}🔧 Configuring ROCm 6.0 for Strix Halo (gfx1151)...${NC}"

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
echo -e "${BLUE}🧪 Test GPU acceleration:${NC}"
echo "   docker exec \$(docker ps -q) python /opt/test_gpu_acceleration.py"