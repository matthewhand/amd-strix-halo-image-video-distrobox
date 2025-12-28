# ROCm Requirements Policy

## ❌ UNSUPPORTED: ROCm 6.1

**ROCm 6.1 is NOT supported** for this project due to incompatibilities with Qwen Image Studio and other modern AI models.

### Known Issues with ROCm 6.1:
- **Qwen Vision Model**: `RuntimeError: HIP error: invalid device function` when initializing Qwen2_5_VisionRotaryEmbedding
- **Missing GPU kernels**: PyTorch 2.6.0+rocm6.1 lacks certain operations required by vision transformers
- **Generation failures**: Image generation fails during model loading phase

### Images to AVOID:
- `amd-strix-halo-image-video-toolbox:rocm6.1` ❌
- Any image tagged with `rocm6` or `6.1` ❌

---

## ✅ SUPPORTED: ROCm 7.10+

**ROCm 7.10 or later is REQUIRED** for Qwen Image Studio and modern vision models.

### Working Configuration:
- **PyTorch**: 2.7.1+rocm7.10.0a20251117 or later
- **ROCm Runtime**: 7.10+
- **GPU**: Radeon 8060S (gfx1151) or compatible
- **Environment Variables**:
  ```
  HSA_OVERRIDE_GFX_VERSION=11.5.1
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8,expandable_segments:False
  FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
  ```

### Working Docker Images:
- `amd-strix-halo-image-video-toolbox:final-working-distrobox` ✅
- `amd-strix-halo-image-video-toolbox:gpu-fixed` ✅
- Any image with `rocm7` or newer ✅

---

## Migration from ROCm 6.1

If you have a container running ROCm 6.1:

```bash
# 1. Stop and remove the old container
docker stop <container-name>
docker rm <container-name>

# 2. Remove the old image (optional, to free space)
docker rmi amd-strix-halo-image-video-toolbox:rocm6.1

# 3. Use a supported image
docker run -d --name strix-halo-working \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /mnt/docker:/mnt/docker \
  -p 8000:8000 -p 8188:8188 \
  amd-strix-halo-image-video-toolbox:final-working-distrobox
```

---

## Why ROCm 6.1 Doesn't Work

The Qwen Vision Transformer uses specific GPU operations that were not available or were buggy in ROCm 6.1. The error occurs during initialization:

```
RuntimeError: HIP error: invalid device function
Compile with `TORCH_USE_HIP_DSA` to enable device-side assertions.
```

This happens in `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` at line 98:
```python
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
```

The operation `torch.arange` on the GPU with specific parameters is not supported in ROCm 6.1's PyTorch build.

### ⚠️ Experimental: ROCm 6.1 with Patches

**WARNING**: This is **unsupported** and may not work. Use at your own risk.

If you absolutely must use ROCm 6.1 (not recommended), there was a compatibility patch attempt that added:
- Flash Attention shim for AMD (`QWEN_FA_SHIM=1`)
- Pipeline-level patches to remove `offload_state_dict`
- Model-level patches for Qwen2_5_VL

**See commit**: [`0088bfc`](https://github.com/your-org/amd-strix-halo-image-video-toolboxes/tree/0088bfc) - "Add Qwen Image Studio compatibility patches for ROCm and distrobox environment"

**Files from that commit**:
- `start_qwen_studio_patched.py` - Patched launcher with Flash Attention shim
- `patched_cli_runner.py` - CLI runner with compatibility patches

**Even with these patches**, Qwen Image Studio **still fails** on ROCm 6.1 with the HIP error shown above. The patches fix some issues but not the fundamental incompatibility with the vision transformer.

**Recommendation**: Just use ROCm 7.10+. It works natively without any patches.

---

## Testing ROCm Version

To check your ROCm version in a container:

```bash
docker exec <container-name> python -c "import torch; print(torch.__version__)"
# Should output: 2.7.1+rocm7.10.0a20251117 or later
```

If you see `2.6.0+rocm6.1`, you need to upgrade.

---

## Dockerfile Guidelines

When creating new Docker images:
- **Use**: ROCm 7.10+ base images
- **Avoid**: ROCm 6.1 base images
- **Test**: Qwen Image Studio before tagging as "working"

Example bad Dockerfile:
```dockerfile
FROM rocm/pytorch:rocm6.1-runtime  # ❌ DON'T USE
```

Example good Dockerfile:
```dockerfile
FROM rocm/pytorch:rocm7.10-runtime  # ✅ USE THIS
```

---

## Scripts Status

### ✅ Working Scripts (ROCm 7.10+):
- `start_distrobox.sh` (if using compatible container)
- Any script that doesn't specify ROCm 6.1

### ❌ Deprecated Scripts (ROCm 6.1):
- All scripts with `rocm6` in the name
- `start_conservative_working.sh` (if hardcoded to ROCm 6.1)
- `start_exact_working.sh` (if hardcoded to ROCm 6.1)

---

## Last Updated: 2025-12-28

**TL;DR**: Use ROCm 7.10+. ROCm 6.1 is broken for Qwen. Don't use it.
