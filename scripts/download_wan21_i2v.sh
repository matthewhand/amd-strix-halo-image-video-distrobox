#!/bin/bash

# WAN 2.1 I2V Download Script
# Downloads stable WAN 2.1 I2V models that don't have channel mismatch issues

set -e

echo "🔧 WAN 2.1 I2V Models Download"
echo "============================="
echo "This will download stable WAN 2.1 I2V models without channel mismatch issues"
echo ""

BASE_DIR="/home/matthewh"
COMFYUI_MODELS="$BASE_DIR/comfy-models"

# Create directories
mkdir -p "$COMFYUI_MODELS/diffusion_models"
mkdir -p "$COMFYUI_MODELS/vae"

echo "📋 Downloading WAN 2.1 I2V Models (720P stable version)..."

# 1. Download WAN 2.1 I2V-14B-720P main model
echo ""
echo "📥 Downloading WAN 2.1 I2V-14B-720P main model..."
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir "$COMFYUI_MODELS/diffusion_models/wan21_i2v_14b_720p" \
    --resume-download \
    --include "*.safetensors" \
    --include "*.json" \
    --include "*.bin"

# 2. Download WAN 2.1 VAE (required for I2V)
echo ""
echo "📥 Downloading WAN 2.1 VAE (required for I2V)..."
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir "$COMFYUI_MODELS/vae" \
    --resume-download \
    --include "vae/*.safetensors" \
    --include "vae/*.json"

# 3. Create symlinks for ComfyUI compatibility
echo ""
echo "🔗 Creating ComfyUI symlinks..."

cd "$COMFYUI_MODELS/diffusion_models"

# Create symlinks with names ComfyUI expects
if [ -f "wan21_i2v_14b_720p/model.safetensors" ]; then
    # Backup existing WAN 2.2 I2V models
    for model in wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors; do
        if [ -f "$model" ]; then
            cp "$model" "${model}.wan22_backup"
            echo "📦 Backed up $model"
        fi
    done

    # Create symlinks for WAN 2.1 models
    ln -sf "wan21_i2v_14b_720p/model.safetensors" "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    ln -sf "wan21_i2v_14b_720p/model.safetensors" "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    echo "✅ Created WAN 2.1 I2V model symlinks"
else
    echo "❌ WAN 2.1 I2V model not found in expected location"
fi

# Create VAE symlink
cd "$COMFYUI_MODELS/vae"
if [ -f "diffusion_pytorch_model.safetensors" ]; then
    ln -sf "diffusion_pytorch_model.safetensors" "wan2.1_vae.safetensors"
    echo "✅ Created WAN 2.1 VAE symlink"
fi

# Verify downloads
echo ""
echo "✅ Verifying downloads..."

cd "$COMFYUI_MODELS/diffusion_models"
if [ -f "wan21_i2v_14b_720p/model.safetensors" ]; then
    size=$(du -h "wan21_i2v_14b_720p/model.safetensors" | cut -f1)
    echo "📊 WAN 2.1 I2V Model: $size"
else
    echo "❌ WAN 2.1 I2V Model: NOT FOUND"
fi

cd "$COMFYUI_MODELS/vae"
if [ -f "diffusion_pytorch_model.safetensors" ]; then
    size=$(du -h "diffusion_pytorch_model.safetensors" | cut -f1)
    echo "📊 WAN 2.1 VAE: $size"
else
    echo "❌ WAN 2.1 VAE: NOT FOUND"
fi

echo ""
echo "🎉 WAN 2.1 I2V Download Completed!"
echo ""
echo "💡 What was downloaded:"
echo "  ✅ WAN 2.1 I2V-14B-720P (stable, no channel issues)"
echo "  ✅ WAN 2.1 VAE (compatible with I2V)"
echo "  ✅ ComfyUI symlinks created"
echo ""
echo "📦 Backups created:"
if ls wan2.2_i2v_*_fp8_scaled.safetensors.wan22_backup 1> /dev/null 2>&1; then
    echo "  📁 WAN 2.2 I2V models backed up with .wan22_backup extension"
fi
echo ""
echo "🔧 Next steps:"
echo "  1. Restart ComfyUI: docker restart strix-halo-toolbox"
echo "  2. Test I2V workflow - should work without channel errors"
echo "  3. If issues persist, check ComfyUI logs"
echo ""
echo "📋 Expected behavior:"
echo "  ✅ No '36 vs 64 channel' errors"
echo "  ✅ Image-to-Video should work properly"
echo "  ✅ VAE should be compatible"