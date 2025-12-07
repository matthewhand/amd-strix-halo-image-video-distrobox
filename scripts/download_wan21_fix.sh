#!/bin/bash

# Quick WAN 2.1 I2V Fix Script
# This is the recommended solution for the channel mismatch issue

set -e

echo "🔧 WAN 2.1 I2V Fix for Channel Mismatch"
echo "===================================="
echo ""
echo "🎯 PROBLEM: WAN 2.2 I2V has '36 vs 64 channel' error"
echo "✅ SOLUTION: Use stable WAN 2.1 I2V models"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing huggingface-cli..."
    pip install "huggingface_hub[hf_transfer]"
fi

echo "❓ This will:"
echo "  1. Download WAN 2.1 I2V-14B-720P (stable version)"
echo "  2. Download WAN 2.1 VAE (compatible)"
echo "  3. Backup your current WAN 2.2 I2V models"
echo "  4. Replace with WAN 2.1 models (no channel issues)"
echo ""
read -p "Continue with WAN 2.1 fix? (Y/n): " response
if [[ "$response" =~ ^[Nn]$ ]]; then
    echo "❌ Cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting WAN 2.1 fix..."

# Run the main download script
./scripts/download_wan21_i2v.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! WAN 2.1 I2V fix completed!"
    echo ""
    echo "🔧 WHAT TO DO NOW:"
    echo "  1. Restart ComfyUI: docker restart strix-halo-toolbox"
    echo "  2. Wait 30 seconds for services to start"
    echo "  3. Test your Image-to-Video workflow"
    echo ""
    echo "✅ EXPECTED RESULT:"
    echo "  - No more '36 vs 64 channel' errors"
    echo "  - Image-to-Video should work properly"
    echo "  - VAE compatibility resolved"
    echo ""
    echo "📋 IF ISSUES PERSIST:"
    echo "  - Use ./scripts/rollback_wan22.sh to restore WAN 2.2"
    echo "  - Use T2V (Text-to-Video) models instead"
    echo "  - Check ComfyUI logs for other errors"
else
    echo ""
    echo "❌ WAN 2.1 fix failed!"
    echo "   Check error messages above"
    echo "   Your original models were not modified"
fi