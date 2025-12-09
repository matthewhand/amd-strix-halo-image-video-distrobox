# Example Workflows

This directory contains tested ComfyUI workflows that demonstrate the **Qwen → WAN I2V pipeline** functionality.

## Files

### `wan_i2v_workflow.json`
- **Type**: ComfyUI workflow JSON
- **Purpose**: Complete Image-to-Video generation pipeline
- **Requirements**: Qwen-generated image + WAN models + WAN 2.1 VAE
- **Source**: Original workflow source unknown/uncredited (user-provided working configuration)

## How to Use

### 1. Load Workflow in ComfyUI
1. Open ComfyUI: http://localhost:8188
2. Drag & drop `wan_i2v_workflow.json` into ComfyUI
3. Or use API: `curl -X POST http://localhost:8188/prompt -H "Content-Type: application/json" -d @wan_i2v_workflow.json`

### 2. Prepare Input Image
- Workflow expects: `qwen_sunset_lake.png` in ComfyUI input directory
- Generate with Qwen Image Studio at http://localhost:8000
- Or upload your own image via ComfyUI LoadImage node

### 3. Run Generation
- Click "Queue Prompt" in ComfyUI
- Monitor memory usage: 7-12GB during processing
- Output saved to: `/opt/ComfyUI/output/`

## Expected Results

- **Input**: Static image (your sunset lake or any image)
- **Output**: Animated video (WEBM + WEBP formats)
- **Processing**: 2-10 minutes depending on video length
- **Memory**: Peaks around 7-12GB
- **Temperature**: 36°C → 54°C (active computation)

## Key Components

This workflow demonstrates our **critical fixes**:
- ✅ **Flash Attention Patch**: No more dependency errors
- ✅ **WAN 2.1 VAE**: 64-channel compatibility fix
- ✅ **End-to-End Pipeline**: Qwen → WAN I2V working together

## Troubleshooting

If the workflow fails:
1. Check that WAN 2.1 VAE is loaded: `wan_2.1_vae.safetensors`
2. Verify image exists in ComfyUI input directory
3. Monitor memory usage for GPU availability
4. Check ComfyUI logs for any errors