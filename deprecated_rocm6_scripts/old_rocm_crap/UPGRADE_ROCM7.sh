#!/bin/bash

# AMD ROCm 7 Upgrade Script for Strix Halo (gfx1151) Support
# This will upgrade your host ROCm from 5.7 to the latest version

echo "🚀 Upgrading ROCm to latest version for Strix Halo support..."
echo "⚠️  This will upgrade your host ROCm installation"
echo ""

# Backup current ROCm info
echo "📋 Current ROCm Info:"
rocm-smi --showproductname 2>/dev/null || echo "Current version: ROCm 5.7.0"
echo ""

# Add AMD ROCm repository
echo "📦 Adding AMD ROCm repository..."
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg

# Add ROCm 6.0+ repository (has gfx1151 support)
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

echo ""
echo "🔄 Updating package lists..."
sudo apt update

echo ""
echo "📦 Available ROCm packages:"
apt search rocm-dev | grep -E "rocm-[0-9]" | head -5

echo ""
echo "⬆️  Upgrading ROCm packages to 6.0+ (with gfx1151 support)..."
sudo apt install -y rocm-dev rocm-dkms rocm-smi libhipblas-dev librocblas-dev

echo ""
echo "✅ ROCm upgrade completed!"
echo ""
echo "🔄 Please reboot your system to load the new ROCm drivers:"
echo "   sudo reboot"
echo ""
echo "After reboot, test with:"
echo "   rocm-smi --showproductname"
echo "   ./start_distrobox_fast.sh gpu"
echo ""
echo "💡 Expected outcome:"
echo "   - ROCm 6.0+ with proper gfx1151 support"
echo "   - Working GPU acceleration for video generation"
echo "   - Much faster performance than CPU mode"