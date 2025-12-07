#!/bin/bash

# Quick script to download missing WAN models
# You have most models - just need the VAE and potentially updated versions

set -e

echo "🔍 WAN Models - Missing Items Download"
echo "===================================="

BASE_DIR="/home/matthewh"
COMFYUI_MODELS="$BASE_DIR/comfy-models"
VAE_DIR="$COMFYUI_MODELS/vae"

# Create VAE directory
mkdir -p "$VAE_DIR"

echo "📋 Your Current Models:"
echo "✅ wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors (I2V - has channel issue)"
echo "✅ wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors (I2V - has channel issue)"
echo "✅ wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors (T2V - working!)"
echo "✅ wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors (T2V - working!)"
echo "✅ wan2.2_ti2v_5B_fp16.safetensors (TI2V)"
echo ""

echo "❌ Missing: wan2.2_vae.safetensors (Required for I2V)"

# Download missing VAE
echo ""
echo "📥 Downloading WAN VAE..."
if [ ! -f "$VAE_DIR/wan2.2_vae.safetensors" ]; then
    curl -L -o "$VAE_DIR/wan2.2_vae.safetensors" \
        "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/resolve/main/vae/diffusion_pytorch_model.safetensors"

    if [ $? -eq 0 ]; then
        echo "✅ WAN VAE downloaded successfully"
        echo "📊 Size: $(du -h "$VAE_DIR/wan2.2_vae.safetensors" | cut -f1)"
    else
        echo "❌ Failed to download WAN VAE"
    fi
else
    echo "✅ WAN VAE already exists"
fi

# Optional: Download ComfyUI repackaged (potentially fixed I2V)
echo ""
echo "❓ Download ComfyUI repackaged I2V models (may fix channel issue)? (y/N):"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "📥 Downloading ComfyUI WAN repackaged models..."

    # Backup existing I2V models
    if [ -f "$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" ]; then
        cp "$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
           "$COMFYUI_MODELS/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors.backup"
        echo "📦 Backed up existing I2V high noise model"
    fi

    # Download repackaged models
    huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
        --local-dir "$COMFYUI_MODELS/diffusion_models" \
        --resume-download \
        --include "wan2.2_i2v_14b.safetensors" \
        --include "wan2.2_vae.safetensors"

    # Create symlinks for ComfyUI compatibility
    cd "$COMFYUI_MODELS/diffusion_models"
    if [ -f "wan2.2_i2v_14b.safetensors" ]; then
        ln -sf "wan2.2_i2v_14b.safetensors" "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors.new"
        ln -sf "wan2.2_i2v_14b.safetensors" "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors.new"
        echo "🔗 Created new I2V model symlinks"
        echo "💡 Test with .new versions, replace if they work"
    fi
fi

echo ""
echo "🎉 Download completed!"
echo ""
echo "💡 Current Status:"
echo "  ✅ T2V Models: Working (use Text-to-Video)"
echo "  ⚠️  I2V Models: May have channel issues"
echo "  ✅ WAN VAE: Should be downloaded now"
echo ""
echo "🔧 If I2V still has channel issues:"
echo "   1. Try the ComfyUI repackaged models (.new files)"
echo "   2. Search for WAN I2V channel mismatch fixes"
echo "   3. Use T2V as a reliable alternative"