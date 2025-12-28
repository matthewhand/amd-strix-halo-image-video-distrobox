# Docker Images and Build Files

## ✅ Supported (ROCm 7.10+)

### Working Docker Images:
- `final-working-distrobox` - **PyTorch 2.7.1+ROCm 7.10** - RECOMMENDED
- `gpu-fixed` - Working configuration with GPU patches

### Dockerfile: `Dockerfile.rocm7.1`
Builds images with ROCm 7.1. Use this as a base for new images.

```bash
docker build -f docker/Dockerfile.rocm7.1 -t amd-strix-halo:rocm7.1 .
```

## ⚠️ Deprecated (DO NOT USE)

### ROCm 6.1 - BROKEN
- `Dockerfile.rocm6` - ❌ Does not support Qwen Vision
- `Dockerfile.rocm6-simple` - ❌ Incompatible with modern models

These have been moved to `../deprecated_rocm6_scripts/`

### ROCm 5.7 - OBSOLETE
- `Dockerfile.rocm5.7` - ❌ Too old, missing features

## Build Script

**`docker/build_rocm_container.sh`** - Builds Docker images with specified ROCm version.

**Default behavior**: Use ROCm 7.10+ for all new builds.

```bash
# Build with ROCm 7.1
./docker/build_rocm_container.sh rocm7.1

# Build with ROCm 7.10 (recommended)
./docker/build_rocm_container.sh rocm7.10
```

## Minimum Requirements

- **ROCm**: 7.10 or later
- **PyTorch**: 2.7.1+rocm7.10 or later
- **GPU**: Radeon 8060S (gfx1151) or compatible

See [ROCM_REQUIREMENTS.md](../ROCM_REQUIREMENTS.md) for details.

---

**Last Updated**: 2025-12-28
