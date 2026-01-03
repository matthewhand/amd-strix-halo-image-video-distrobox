#!/bin/bash
set -e

# Target directory (mapped to host ~/wan-models)
# Log everything
exec > >(tee -a /tmp/download.log) 2>&1

TARGET_DIR="/workspace/wan-models"
mkdir -p "$TARGET_DIR"

export HF_HUB_ENABLE_HF_TRANSFER=0

echo "⬇️ Downloading Wan2.2-T2V-A14B..."
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir "$TARGET_DIR/Wan2.2-T2V-A14B"

echo "⬇️ Downloading Wan2.2-I2V-A14B..."
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir "$TARGET_DIR/Wan2.2-I2V-A14B"

echo "⬇️ Downloading Wan2.2-Lightning (LoRA)..."
hf download lightx2v/Wan2.2-Lightning --local-dir "$TARGET_DIR/Wan2.2-Lightning"

echo "✅ Download complete."
