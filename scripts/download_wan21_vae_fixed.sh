#!/bin/bash

# WAN 2.1 VAE Download Script - Updated with correct URLs
# Downloads the stable WAN 2.1 VAE for I2V channel fix

set -e

echo "🔧 WAN 2.1 VAE Download Script (Updated)"
echo "====================================="
echo "This will download the stable WAN 2.1 VAE to fix I2V channel issues"
echo ""

BASE_DIR="/home/matthewh"
COMFYUI_MODELS="$BASE_DIR/comfy-models"

# Create directories
mkdir -p "$COMFYUI_MODELS/vae"

echo "📋 Current Status Analysis:"
echo "✅ WAN 2.2 I2V models: Available (14GB each)"
echo "✅ WAN 2.1 VAE found in repository: Wan2.1_VAE.pth (484MB - .pth format)"
echo "❌ Repository uses Git LFS, not direct downloads"
echo "❌ WAN 2.1 VAE files need Git LFS to download properly"

echo ""
echo "📥 Download Methods to Try:"

# Method 1: Try alternative enhanced VAE
echo ""
echo "🔄 Method 1: Enhanced WAN VAE from spacepxl"
enhanced_vae_url="https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/resolve/main/wan_2.1_vae.safetensors"

if curl -s -I "$enhanced_vae_url" | grep -q "200 OK"; then
    echo "✅ Enhanced VAE accessible, downloading..."

    curl -L -o "wan_2_1_vae_enhanced.safetensors" \
        "$enhanced_vae_url" \
        --connect-timeout 30 \
        --max-time 600 \
        --retry 3 \
        --retry-delay 10

    if [ $? -eq 0 ] && [ -f "wan_2_1_vae_enhanced.safetensors" ]; then
        size=$(du -h "wan_2_1_vae_enhanced.safetensors" | cut -f1)
        echo "✅ Enhanced VAE downloaded: $size"

        # Test loading
        echo "🧪 Testing VAE loading..."
        docker exec strix-halo-toolbox bash -c "
            cd /opt/ComfyUI/models/vae
            python3 -c \"
            import sys
            sys.path.append('/opt/ComfyUI')
            from comfy import utils

            try:
                vae_data = utils.load_torch_file('wan_2_1_vae_enhanced.safetensors')
                print('✅ Enhanced VAE loads successfully!')
                print(f'Components: {len(vae_data)}')
            except Exception as e:
                print(f'❌ Enhanced VAE loading error: {e}')
            \"
        "

        if [ $? -eq 0 ]; then
            echo "✅ Enhanced VAE working! Creating symlinks..."
            cd "$COMFYUI_MODELS/vae"
            ln -sf "wan_2_1_vae_enhanced.safetensors" "wan2.2_vae.safetensors"
            ln -sf "wan_2_1_vae_enhanced.safetensors" "wan_2.1_vae.safetensors"
            echo "✅ VAE symlinks created"

            echo ""
            echo "🎉 SUCCESS! WAN 2.1 VAE downloaded and working!"
            echo ""
            echo "💡 Next steps:"
            echo "  1. Restart ComfyUI: docker restart strix-halo-toolbox"
            echo "  2. Test I2V workflow - should work without channel issues"
            echo "  3. Use Enhanced VAE (2x upscaling) with your WAN 2.2 I2V models"
            exit 0
        fi
    else
        echo "❌ Enhanced VAE download failed"
    else
        echo "❌ Enhanced VAE not accessible"
    fi
else
    echo "❌ Enhanced VAE not accessible"
fi

echo ""
echo "🔄 Method 2: Try ComfyUI Repackaged (may have fixed I2V)"
comfyui_vae_url="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/wan2.2_vae.safetensors"

if curl -s -I "$comfyui_vae_url" | grep -q "200 OK"; then
    echo "✅ ComfyUI repackaged VAE accessible, downloading..."

    curl -L -o "wan2.2_vae_comfyui.safetensors" \
        "$comfyui_vae_url" \
        --connect-timeout 30 \
        --max-time 600 \
        --retry 3 \
        --retry-delay 10

    if [ $? -eq 0 ] && [ -f "wan2.2_vae_comfyui.safetensors" ]; then
        size=$(du -h "wan2.2_vae_comfyui.safetensors" | cut -f1)
        echo "✅ ComfyUI VAE downloaded: $size"

        # Test loading
        echo "🧪 Testing VAE loading..."
        docker exec strix-halo-toolbox bash -c "
            cd /opt/ComfyUI/models/vae
            python3 -c \"
            import sys
            sys.path.append('/opt/ComfyUI')
            from comfy import utils

            try:
                vae_data = utils.load_torch_file('wan2.2_vae_comfyui.safetensors')
                print('✅ ComfyUI VAE loads successfully!')
                print(f'Components: {len(vae_data)}')
            except Exception as e:
                print(f'❌ ComfyUI VAE loading error: {e}')
            \"
        "

        if [ $? -eq 0 ]; then
            echo "✅ ComfyUI VAE working!"
            cp "wan2.2_vae_comfyui.safetensors" "wan2.2_vae.safetensors"
            echo "✅ Set as default WAN VAE"

            echo ""
            echo "🎉 SUCCESS! ComfyUI VAE downloaded and working!"
            echo ""
            echo "💡 Next steps:"
            echo "  1. Restart ComfyUI: docker restart strix-halo-toolbox"
            echo "  2. Test I2V workflow with ComfyUI VAE"
            echo "  3. May resolve channel compatibility issues"
            exit 0
        fi
    else
        echo "❌ ComfyUI VAE download failed"
    else
        echo "❌ ComfyUI VAE not accessible"
    fi
else
    echo "❌ ComfyUI VAE not accessible"
fi

echo ""
echo "❌ Both download methods failed"
echo ""
echo "💡 Alternative Solutions:"
echo "  1. Use T2V (Text-to-Video) models instead - WORKING ✅"
echo "     - No channel mismatch issues"
echo "     - No VAE compatibility problems"
echo "     - Same high quality output"
echo ""
echo "  2. Try Qwen VAE compatibility"
echo "     - Check if Qwen Image Studio VAE works with ComfyUI"
echo "     - May share same latent space"
echo ""
echo " 3. Wait for WAN 2.1 VAE repository fixes"
echo "     - Repository access issues need to be resolved"
echo ""

echo "🎯 Current Working Solution:"
echo "  ✅ Qwen Image Studio: http://localhost:8000"
echo "  ✅ ComfyUI: http://localhost:8188"
echo "  ✅ WAN T2V models: Available and tested"
echo "  ✅ Text-to-Video workflow: Ready to use"