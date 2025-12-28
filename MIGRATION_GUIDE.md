# Migration Guide: ROCm 6.1 → ROCm 7.10+

## Quick Summary

**ROCm 6.1 is BROKEN for Qwen Image Studio.** You must upgrade to ROCm 7.10+.

## Symptoms of ROCm 6.1 Issue

```
RuntimeError: HIP error: invalid device function
```

This occurs when Qwen tries to initialize the vision transformer.

## Solution: Use the Correct Docker Image

### ❌ DON'T USE:
```bash
amd-strix-halo-image-video-toolbox:rocm6.1
```

### ✅ USE INSTEAD:
```bash
amd-strix-halo-image-video-toolbox:final-working-distrobox
```

## Steps to Migrate

### 1. Stop and Remove Old Container

```bash
docker stop strix-halo-working
docker rm strix-halo-working
```

### 2. Remove Old Image (Optional)

```bash
docker rmi amd-strix-halo-image-video-toolbox:rocm6.1
```

### 3. Start New Container with ROCm 7.10+

```bash
docker run -d --name strix-halo-working \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v /mnt/docker:/mnt/docker \
  -p 8000:8000 \
  -p 8188:8188 \
  amd-strix-halo-image-video-toolbox:final-working-distrobox
```

### 4. Verify ROCm Version

```bash
docker exec strix-halo-working python -c "import torch; print(torch.__version__)"
```

Expected output: `2.7.1+rocm7.10.0a20251117` or later

## What Changed?

### With ROCm 7.10+:
- ✅ Qwen Image Studio works with GPU acceleration
- ✅ No Flash Attention patches needed
- ✅ Vision transformer loads correctly
- ✅ Native GPU kernel support for all operations

### With ROCm 6.1 (BROKEN):
- ❌ Qwen fails with "invalid device function"
- ❌ Missing GPU kernels for vision transformer
- ❌ Requires workaround patches that don't fully work

## For Developers

If you're building your own Docker image:

### ❌ WRONG:
```dockerfile
FROM rocm/pytorch:rocm6.1-runtime
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
```

### ✅ CORRECT:
```dockerfile
FROM rocm/pytorch:rocm7.10-runtime
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.10
```

## Startup Scripts

### Deprecated Scripts (ROCm 6.1):
All moved to `deprecated_rocm6_scripts/` directory.

### Working Scripts (ROCm 7.10+):
- `start_gpu_rocm71.sh` - ROCm 7.1 configuration
- `start_hybrid_rocm71_fix.sh` - Hybrid with ROCm 7.1
- `start_gpu_strix_fixed.sh` - Strix-specific fixes

## Testing Your Setup

After migration, test image generation:

```bash
# Access Qwen Image Studio
http://localhost:8000

# Try generating a simple image
Prompt: "a blue circle on white background"
Settings: Ultra Fast, 2 steps

# Should complete without HIP errors
```

## Still Having Issues?

See [ROCM_REQUIREMENTS.md](ROCM_REQUIREMENTS.md) for full details.

---

**Last Updated**: 2025-12-28
