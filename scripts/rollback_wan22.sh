#!/bin/bash

# WAN 2.2 Rollback Script
# Restores original WAN 2.2 I2V models if needed

set -e

echo "🔄 WAN 2.2 Rollback Script"
echo "========================="
echo "This will restore your original WAN 2.2 I2V models"
echo ""

BASE_DIR="/home/matthewh"
COMFYUI_MODELS="$BASE_DIR/comfy-models"

cd "$COMFYUI_MODELS/diffusion_models"

# Check for backups
echo "📋 Checking for WAN 2.2 backups..."

backups_found=false
for model in wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors; do
    if [ -f "${model}.wan22_backup" ]; then
        echo "  ✅ Found backup: ${model}.wan22_backup"
        backups_found=true
    fi
done

if [ "$backups_found" = false ]; then
    echo "❌ No WAN 2.2 backups found!"
    echo "   Cannot rollback to WAN 2.2 I2V models"
    exit 1
fi

echo ""
echo "⚠️  WARNING: This will replace WAN 2.1 I2V with WAN 2.2 I2V"
echo "   WAN 2.2 has known channel mismatch issues"
echo ""
read -p "Continue with rollback? (y/N): " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "❌ Rollback cancelled"
    exit 0
fi

# Restore WAN 2.2 models
echo ""
echo "🔄 Restoring WAN 2.2 I2V models..."

for model in wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors; do
    if [ -f "${model}.wan22_backup" ]; then
        echo "  🔄 Restoring $model"
        # Remove current symlink/model
        rm -f "$model"
        # Restore backup
        mv "${model}.wan22_backup" "$model"
        echo "  ✅ Restored $model"
    fi
done

echo ""
echo "🎉 WAN 2.2 rollback completed!"
echo ""
echo "📋 What happened:"
echo "  ✅ Original WAN 2.2 I2V models restored"
echo "  ❌ Channel mismatch issues may return"
echo "  ⚠️  WAN 2.1 models still in wan21_i2v_14b_720p/ directory"
echo ""
echo "🔧 Next steps:"
echo "  1. Restart ComfyUI: docker restart strix-halo-toolbox"
echo "  2. Test I2V workflow"
echo "  3. If channel errors occur, use WAN 2.1 or T2V models"