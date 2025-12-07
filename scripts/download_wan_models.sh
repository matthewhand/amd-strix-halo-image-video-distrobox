#!/bin/bash

# WAN 2.2 Models Download Script
# Quick bash script for downloading essential WAN models

set -e

BASE_DIR="/home/matthewh"
COMFYUI_MODELS="$BASE_DIR/comfy-models/diffusion_models"
CACHE_DIR="$BASE_DIR/.cache/huggingface/hub"

echo "🚀 WAN 2.2 Models Download Script"
echo "=================================="

# Create directories
mkdir -p "$COMFYUI_MODELS"
mkdir -p "$CACHE_DIR"

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing huggingface-cli..."
    pip install "huggingface_hub[hf_transfer]"
fi

# Core WAN 2.2 models to download
echo ""
echo "🔥 Downloading essential WAN 2.2 models..."

# 1. Official ComfyUI Repackaged (recommended - includes working I2V models)
echo "📥 Downloading ComfyUI WAN 2.2 repackaged models..."
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    --local-dir "$COMFYUI_MODELS" \
    --resume-download \
    --include "*.safetensors" \
    --include "*.json"

# 2. WAN Text-to-Video models (T2V - working)
echo ""
echo "📥 Downloading WAN T2V models..."
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B \
    --local-dir "$COMFYUI_MODELS/wan2.2_t2v_a14b" \
    --resume-download \
    --include "*.safetensors" \
    --include "*.json"

# 3. WAN Text-to-Image-to-Video (TI2V)
echo ""
echo "📥 Downloading WAN TI2V model..."
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir "$COMFYUI_MODELS/wan2.2_ti2v_5b" \
    --resume-download \
    --include "*.safetensors" \
    --include "*.json"

# 4. WAN Lightning LoRA (for faster generation)
echo ""
echo "📥 Downloading WAN Lightning LoRA..."
huggingface-cli download lightx2v/Wan2.2-Lightning \
    --local-dir-use-symlinks False \
    --resume-download \
    --include "*Lightning*.safetensors"

# Create symlinks for ComfyUI compatibility
echo ""
echo "🔗 Creating ComfyUI symlinks..."
cd "$COMFYUI_MODELS"

# Create symlinks for commonly expected model names
if [ -f "wan2.2_t2v_14b.safetensors" ]; then
    ln -sf "wan2.2_t2v_14b.safetensors" "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" 2>/dev/null || true
    ln -sf "wan2.2_t2v_14b.safetensors" "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" 2>/dev/null || true
    echo "  ✅ Created T2V model symlinks"
fi

if [ -f "wan2.2_i2v_14b.safetensors" ]; then
    ln -sf "wan2.2_i2v_14b.safetensors" "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" 2>/dev/null || true
    ln -sf "wan2.2_i2v_14b.safetensors" "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" 2>/dev/null || true
    echo "  ✅ Created I2V model symlinks"
fi

# Verify downloads
echo ""
echo "✅ Verifying downloads..."
TOTAL_SIZE=$(du -sh "$COMFYUI_MODELS"/*.safetensors 2>/dev/null | awk '{sum+=$1} END {print sum "GB"}' || echo "0GB")
MODEL_COUNT=$(ls -1 "$COMFYUI_MODELS"/*.safetensors 2>/dev/null | wc -l)

echo "📊 Download Summary:"
echo "  📁 Total models: $MODEL_COUNT"
echo "  💾 Total size: $TOTAL_SIZE"
echo "  📍 Location: $COMFYUI_MODELS"

# Check for VAE
if [ -f "$COMFYUI_MODELS/wan2.2_vae.safetensors" ]; then
    echo "  ✅ WAN VAE: Available"
else
    echo "  ⚠️  WAN VAE: Missing - download from huggingface"
fi

echo ""
echo "🎉 WAN 2.2 models download completed!"
echo ""
echo "💡 Usage in ComfyUI:"
echo "   1. Restart ComfyUI service"
echo "   2. Look for WAN models in checkpoint loader"
echo "   3. Use T2V models (working) or try I2V models"
echo ""
echo "⚠️  Note: I2V models may have channel compatibility issues"
echo "   Use T2V (Text-to-Video) for reliable generation"