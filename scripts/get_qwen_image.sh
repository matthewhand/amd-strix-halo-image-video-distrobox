#!/usr/bin/env bash
# Qwen model download with resume support
# Supports both HuggingFace format (for Qwen Image Studio) and ComfyUI split format

set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Use huggingface-cli (check for available commands)
if [[ -x "/opt/venv/bin/hf" ]]; then
    HF_CMD="/opt/venv/bin/hf"
elif command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
else
    HF_CMD="python -m huggingface_hub.cli"
fi

MODEL_HOME="$HOME/comfy-models"
STAGE="$MODEL_HOME/.hf_stage_qwen"

mkdir -p "$MODEL_HOME"/{text_encoders,vae,diffusion_models}
mkdir -p "$STAGE"

echo "🔄 Qwen Model Download with Resume Support"
echo "==========================================="
echo ""

# Check if huggingface-cli is available
if [[ ! -x "$HF_CMD" ]] && ! command -v "$HF_CMD" &> /dev/null; then
    echo "❌ huggingface-cli not found!"
    echo "   Install with: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

echo "✓ Using: $HF_CMD"
echo ""

echo "Which download format do you need?"
echo "  1) HuggingFace format (for Qwen Image Studio) - auto-resumes"
echo "  2) ComfyUI split format (for ComfyUI integration)"
read -rp "Enter 1 or 2: " format_choice

echo ""
echo "Which Qwen variant do you want to download?"
echo "  1) Qwen-Image (20B text-to-image)"
echo "  2) Qwen-Image-Edit (image editing)"
echo "  3) Both"
read -rp "Enter 1, 2, or 3: " model_choice

# ComfyUI format download function
dl() {
  local repo="$1"; shift
  local remote="$1"; shift
  local subdir="$1"; shift
  local dest="$MODEL_HOME/$subdir/$(basename "$remote")"
  local staged="$STAGE/$remote"

  if [[ -f "$dest" ]]; then
    echo "✓ Already present: $dest"
    return
  fi

  echo "↓ Downloading $(basename "$remote") → $dest"
  mkdir -p "$(dirname "$staged")"
  $HF_CMD download "$repo" "$remote" \
      --repo-type model \
      --cache-dir "$HF_HOME" \
      --local-dir "$STAGE"
  mv -f "$staged" "$dest"
}

# HuggingFace format download function
download_hf_format() {
  local model="$1"

  echo "📥 Downloading $model (HuggingFace format with resume)..."
  echo ""

  # Clean up incomplete locks that might prevent resume
  echo "🧹 Cleaning up any stale download locks..."
  find "$HF_HOME/hub" -name "*.lock" -type f -delete 2>/dev/null || true

  # Download with resume support (hf command auto-resumes)
  $HF_CMD download "$model" \
      --repo-type model \
      --cache-dir "$HF_HOME"

  echo ""
  echo "✅ $model download complete!"
}

# ComfyUI format download function
download_comfyui_format() {
  local model_name="$1"

  echo "📥 Downloading $model_name (ComfyUI format with resume)..."
  echo ""

  case "$model_name" in
    "Qwen-Image")
      REPO="Comfy-Org/Qwen-Image_ComfyUI"
      dl "$REPO" "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors" "diffusion_models"
      dl "$REPO" "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" "text_encoders"
      dl "$REPO" "split_files/vae/qwen_image_vae.safetensors" "vae"
      ;;
    "Qwen-Image-Edit")
      REPO="Comfy-Org/Qwen-Image-Edit_ComfyUI"
      # Requires text encoder + VAE from Qwen-Image
      BASE="Comfy-Org/Qwen-Image_ComfyUI"
      dl "$BASE" "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" "text_encoders"
      dl "$BASE" "split_files/vae/qwen_image_vae.safetensors" "vae"
      dl "$REPO" "split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors" "diffusion_models"
      ;;
  esac

  echo ""
  echo "✅ $model_name download complete!"
}

# Main download logic
case "$format_choice" in
  1)
    # HuggingFace format
    case "$model_choice" in
      1|3)
        download_hf_format "Qwen/Qwen-Image"
        ;;
    esac

    case "$model_choice" in
      2|3)
        download_hf_format "Qwen/Qwen-Image-Edit"
        ;;
    esac
    ;;
  2)
    # ComfyUI format
    case "$model_choice" in
      1|3)
        download_comfyui_format "Qwen-Image"
        ;;
    esac

    case "$model_choice" in
      2|3)
        download_comfyui_format "Qwen-Image-Edit"
        ;;
    esac
    ;;
  *)
    echo "❌ Invalid format choice. Exiting."
    exit 1
    ;;
esac

echo ""
echo "==========================================="
echo "✨ All downloads complete!"
echo ""
echo "📊 Cache size:"
du -sh "$HF_HOME/hub/models--Qwen--"* 2>/dev/null || echo "  Unable to determine size"
echo ""
if [[ "$format_choice" == "2" ]]; then
  echo "📦 ComfyUI models in: $MODEL_HOME"
fi
echo ""
