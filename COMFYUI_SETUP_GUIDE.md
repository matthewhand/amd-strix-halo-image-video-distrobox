# ComfyUI Setup Guide for AMD Strix Halo

## Overview
This setup provides a complete ComfyUI environment optimized for AMD Strix Halo with automatic model management.

## Quick Start

### 1. Start ComfyUI
```bash
# Option 1: Basic startup
distrobox enter strix-halo-image-video -- bash -c "cd /opt/ComfyUI && source /opt/venv/bin/activate && python main.py --listen 0.0.0.0 --port 8188"

# Option 2: Use the complete launcher (recommended)
distrobox enter strix-halo-image-video -- python3 /home/matthewh/amd-strix-halo-image-video-toolboxes/start_comfyui_complete.py
```

### 2. Access Web Interface
Open your browser to: http://localhost:8188

## Model Management

### Current Status
✅ **Available Models:**
- SDXL Base Model (`sd_xl_base_1.0.safetensors`)
- SDXL Refiner Model (`sd_xl_refiner_1.0.safetensors`)
- SDXL VAE (`sdxl_vae.safetensors`)
- CLIP Vision Model (`clip_vision_g.safetensors`)

⚠️ **Specialized Models (Placeholders):**
- `wan_2.1_vae.safetensors` - Wan VAE
- `qwen_image_vae.safetensors` - Qwen VAE
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` - Wan Video Generation
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` - Wan Video Generation
- `qwen_image_fp8_e4m3fn.safetensors` - Qwen Image Model
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` - UMT5 Text Encoder
- `qwen_2.5_vl_7b_fp8_scaled.safetensors` - Qwen Vision-Language Model

### Model Manager Commands

```bash
# Check current model status
distrobox enter strix-halo-image-video -- python3 /home/matthewh/amd-strix-halo-image-video-toolboxes/comfy_model_manager.py --status --check-specialized

# Install essential ComfyUI models
distrobox enter strix-halo-image-video -- python3 /home/matthewh/amd-strix-halo-image-video-toolboxes/comfy_model_manager.py --install

# Create placeholders for missing specialized models
distrobox enter strix-halo-image-video -- python3 /home/matthewh/amd-strix-halo-image-video-toolboxes/comfy_model_manager.py --create-placeholders
```

## Directory Structure

```
/opt/ComfyUI/
├── models/
│   ├── checkpoints/           # Main models (SDXL, etc.)
│   ├── vae/                   # VAE models
│   ├── diffusion_models/      # Specialized diffusion models
│   ├── text_encoders/         # Text encoder models
│   ├── clip_vision/           # CLIP vision models
│   ├── loras/                 # LoRA models
│   ├── controlnet/            # ControlNet models
│   ├── embeddings/            # Text embeddings
│   └── upscale_models/        # Upscaling models
├── input/                     # Input images
├── output/                    # Generated images
└── custom_nodes/              # Custom extensions
```

## Workflow Compatibility

### Working Workflows
- Standard SDXL workflows
- Image-to-image workflows
- Basic ControlNet workflows
- LoRA-enhanced workflows

### Requires Specialized Models
- Wan video generation workflows
- Qwen vision-language workflows
- Workflows requiring specific Alibaba models

## Environment Variables

The setup automatically configures:
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` - For AMD Strix Halo GPU compatibility

## Troubleshooting

### Common Issues

1. **Model Not Found Errors**
   ```bash
   # Check model status
   python3 comfy_model_manager.py --status --check-specialized
   ```

2. **GPU Not Detected**
   ```bash
   # Check ROCm environment
   echo $HSA_OVERRIDE_GFX_VERSION
   # Should show: 11.0.0
   ```

3. **Workflow Validation Errors**
   - Ensure all required models are present
   - Check if workflow uses specialized Wan/Qwen models
   - Use placeholder files to prevent validation errors

### Getting Help

1. Check ComfyUI logs for specific error messages
2. Verify model placement in correct directories
3. Ensure all required environment variables are set

## Advanced Usage

### Adding Custom Models
1. Place models in appropriate subdirectories under `/opt/ComfyUI/models/`
2. Restart ComfyUI to refresh model list
3. Use models in workflows

### Linking Existing Models
```bash
# Create symlinks to existing models
python3 comfy_model_manager.py --link /path/to/model1.safetensors /path/to/model2.safetensors
```

### Custom Model Downloads
The model manager can be extended to include additional models by adding them to the `essential_models` or `specialized_models` dictionaries.

## Performance Notes

- **GPU**: AMD Radeon Graphics (gfx1151) with ROCm 7.1
- **VRAM**: 128GB available
- **RAM**: 128GB system memory
- **Optimization**: Uses pytorch attention for GPU acceleration