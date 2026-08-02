# ComfyUI Container Fix

**Issue:** Container kept crashing/restarting due to missing environment variables and port mapping.

**Solution Applied:**
1. Added required environment variables: `USER=root`, `HSA_OVERRIDE_GFX_VERSION=11.5.1`
2. Added proper port forwarding: `-p 8188:8188`
3. Ensured correct user permissions for ROCm GPU access

**Current Status:** ✅ ComfyUI v0.26.0 running on http://localhost:8188

---

## Verification Commands

```bash
# Check if running
docker ps | grep strix-halo-comfyui

# Test API
curl -s http://localhost:8188/system_stats | python3 -m json.tool

# View queue
curl -s http://localhost:8188/queue
```

---

## Known Warning (Non-Critical)

The ComfyUI-LTXVideo custom node fails to import due to a kornia dependency issue, but this doesn't prevent server operation. Core functionality remains intact.
