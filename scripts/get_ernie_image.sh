#!/usr/bin/env bash
# ERNIE-Image (Baidu) downloader — Turbo by default, SFT optional.
# Mirrors get_qwen_image.sh conventions.

set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [[ -x "/opt/venv/bin/hf" ]]; then
    HF_CMD="/opt/venv/bin/hf"
elif command -v hf &> /dev/null; then
    HF_CMD="hf"
elif command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
else
    echo "❌ huggingface CLI not found. pip install -U 'huggingface_hub[cli,hf_transfer]'"
    exit 1
fi

echo "🔄 ERNIE-Image Download"
echo "======================="
echo "Using: $HF_CMD"
echo ""
echo "Which variant?"
echo "  1) ERNIE-Image-Turbo (8-step, fast — recommended first)"
echo "  2) ERNIE-Image (SFT, 50-step, higher quality)"
echo "  3) Both"
read -rp "Enter 1, 2, or 3: " choice

download() {
    local repo="$1"
    echo ""
    echo "📥 Downloading $repo ..."
    find "$HF_HOME/hub" -name "*.lock" -type f -delete 2>/dev/null || true
    $HF_CMD download "$repo" --repo-type model --cache-dir "$HF_HOME"
    echo "✅ $repo done."
}

case "$choice" in
    1) download "baidu/ERNIE-Image-Turbo" ;;
    2) download "baidu/ERNIE-Image" ;;
    3) download "baidu/ERNIE-Image-Turbo"; download "baidu/ERNIE-Image" ;;
    *) echo "❌ Invalid choice."; exit 1 ;;
esac

echo ""
echo "📊 Cache size:"
du -sh "$HF_HOME/hub/models--baidu--ERNIE-Image"* 2>/dev/null || true
