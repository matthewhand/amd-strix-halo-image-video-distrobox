# ⚠️ Deprecated - ROCm 6.1 Scripts

**These scripts are DEPRECATED and should NOT be used.**

## Why These Were Deprecated

All scripts in this directory reference or use **ROCm 6.1**, which is **incompatible** with Qwen Image Studio and other modern AI vision models.

### Known Issue:
- **Qwen Vision Model Error**: `RuntimeError: HIP error: invalid device function`
- **Root Cause**: ROCm 6.1 lacks GPU kernels required by Qwen2.5 Vision Transformer

## What to Use Instead

### ✅ Working Alternatives

The remaining scripts in the parent directory use ROCm 7.10+ and are known to work:
- `start_gpu_rocm71.sh` - ROCm 7.1 configuration
- `start_hybrid_rocm71_fix.sh` - Hybrid with ROCm 7.1
- `start_gpu_strix_fixed.sh` - Strix-specific fixes
- `start_comfyui_rocm.sh` - ComfyUI with ROCm

### 📦 Recommended Docker Images

Use these tested images instead:
```bash
amd-strix-halo-image-video-toolbox:final-working-distrobox  # ✅ PyTorch 2.7.1+ROCm 7.10
amd-strix-halo-image-video-toolbox:gpu-fixed                  # ✅ Working configuration
```

### ❌ Avoid These Images

```bash
amd-strix-halo-image-video-toolbox:rocm6.1                   # ❌ Broken for Qwen
```

## Migration Guide

If you were using one of these scripts:

### Old (Broken):
```bash
./start_conservative_working.sh  # Uses ROCm 6.1 - BROKEN
```

### New (Working):
```bash
# Option 1: Use the final-working-distrobox container
docker run -d --name strix-halo-working \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /mnt/docker:/mnt/docker \
  -p 8000:8000 -p 8188:8188 \
  amd-strix-halo-image-video-toolbox:final-working-distrobox

# Option 2: Use a ROCm 7.10+ startup script
./start_gpu_rocm71.sh
```

## Details of Deprecated Scripts

| Script | Reason | Issue |
|--------|--------|-------|
| `start_conservative_working.sh` | Hardcoded ROCm 6.1 reference | Qwen fails |
| `start_debug_embed_fix.sh` | ROCm 6.1 environment | Incompatible |
| `start_distrobox_fast.sh` | ROCm 6.1 configuration | Broken |
| `start_distrobox.sh` | Uses ROCm 6.1 image | Doesn't work |
| `start_exact_working.sh` | ROCm 6.1 tag | Obsolete |
| `start_force_cpu_text.sh` | ROCm 6.1 workaround | Not needed |
| `start_gpu_diffusion_cpu_text.sh` | ROCm 6.1 hybrid | Broken |
| `start_gpu_working_fix.sh` | ROCm 6.1 attempt | Failed |
| `start_hybrid_acceleration.sh` | ROCm 6.1 configuration | Incompatible |
| `start_with_gtt_fixes.sh` | ROCm 6.1 + GTT tweaks | Still broken |

## See Also

- [ROCM_REQUIREMENTS.md](../ROCM_REQUIREMENTS.md) - Full ROCm version policy
- [README.md](../README.md) - Main project documentation

---

**Last Updated**: 2025-12-28
**Status**: DO NOT USE THESE SCRIPTS
