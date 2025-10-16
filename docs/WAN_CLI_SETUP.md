# Wan Video Generation CLI Setup

## Overview
Direct command-line interface for Wan video generation models, bypassing ComfyUI entirely.

## Quick Start

### 1. Download Wan Models

**Essential Models (Required first):**
```bash
# Download text encoder + VAEs
/home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh common
```

**Video Generation Models:**
```bash
# 14B Image-to-Video models (high/low noise)
/home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh 14b-i2v

# 14B Text-to-Video models (high/low noise)
/home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh 14b-t2v

# 5B Text-to-Image-to-Video model
/home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh 5b
```

### 2. Check Model Status
```bash
python3 /home/matthewh/amd-strix-halo-image-video-toolboxes/wan_cli.py --list-models
```

### 3. Generate Videos

**Text-to-Video:**
```bash
python3 wan_cli.py t2v "a beautiful sunset over mountains" -o sunset_video.mp4
```

**Image-to-Video:**
```bash
python3 wan_cli.py i2v input_image.jpg "making the clouds move" -o animated_clouds.mp4
```

## Available Commands

### Model Management
```bash
# List all available models
python3 wan_cli.py --list-models

# Check specific model directory
python3 wan_cli.py --model-dir /path/to/models --list-models
```

### Text-to-Video Generation
```bash
python3 wan_cli.py t2v "your prompt here" [OPTIONS]

Options:
  -o, --output FILE      Output video path (default: output_video.mp4)
  --noise {high,low}     Noise level (default: high)
  --frames INT           Number of frames (default: 16)
```

### Image-to-Video Generation
```bash
python3 wan_cli.py i2v "path/to/image.jpg" "your prompt here" [OPTIONS]

Options:
  -o, --output FILE      Output video path (default: output_video.mp4)
  --noise {high,low}     Noise level (default: high)
  --frames INT           Number of frames (default: 16)
```

## Model Directory Structure

```
/home/matthewh/comfy-models/
├── text_encoders/
│   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors    # Text encoder (required)
├── vae/
│   ├── wan_2.1_vae.safetensors                    # VAE for 14B models
│   └── wan2.2_vae.safetensors                     # VAE for 5B models
└── diffusion_models/
    ├── wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors    # I2V high noise
    ├── wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors     # I2V low noise
    ├── wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors    # T2V high noise
    ├── wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors     # T2V low noise
    └── wan2.2_ti2v_5B_fp16.safetensors                       # 5B TI2V model
```

## Examples

### Basic Text-to-Video
```bash
python3 wan_cli.py t2v "a cat playing piano" -o cat_piano.mp4 --frames 24
```

### Image-to-Video with Low Noise
```bash
python3 wan_cli.py i2v portrait.jpg "subtle smile and blinking" -o living_portrait.mp4 --noise low
```

### High Quality (More Frames)
```bash
python3 wan_cli.py t2v "waterfall flowing in tropical forest" -o waterfall.mp4 --frames 32
```

## Model Information

### 14B Models (High Quality)
- **I2V**: Image-to-video, requires input image + prompt
- **T2V**: Text-to-video, requires only text prompt
- **Noise Levels**: High (more dynamic), Low (more stable)

### 5B Model (Faster)
- **TI2V**: Text-to-image-to-video, generates intermediate image then video

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended
- **VRAM**: 12GB+ for 14B models, 8GB+ for 5B model
- **RAM**: 32GB+ recommended

## Troubleshooting

### Common Issues

1. **"Required models not available"**
   ```bash
   # Download missing models
   /home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh common
   /home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh 14b-i2v
   ```

2. **"CUDA out of memory"**
   - Try the 5B model instead of 14B models
   - Reduce frame count: `--frames 8`
   - Use CPU mode if necessary

3. **Download Interruptions**
   ```bash
   # Resume downloads (script supports resume)
   /home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh common
   ```

4. **Model Path Issues**
   ```bash
   # Specify custom model directory
   python3 wan_cli.py --model-dir /custom/path --list-models
   ```

### Maintenance Commands

```bash
# Clean download cache (keeps models)
/home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh clean-stage

# Clean everything (including cache)
/home/matthewh/amd-strix-halo-image-video-toolboxes/scripts/get_wan22.sh clean-cache
```

## Current Limitations

- This is a CLI interface wrapper for Wan models
- Actual video generation requires the official Wan implementation
- Currently creates placeholder outputs until full implementation is integrated
- Models are large (10-80GB total) and require significant disk space

## Next Steps

To get actual video generation working:
1. Download the required models using the provided scripts
2. Integrate the official Wan video generation code
3. Test with different prompts and settings
4. Optimize for your specific hardware setup

## Model Download Progress

Check download progress with:
```bash
# Monitor model directory
watch -n 5 'ls -lh /home/matthewh/comfy-models/*/*'

# Check model availability
python3 wan_cli.py --list-models
```