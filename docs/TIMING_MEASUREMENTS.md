# Asset Generation Timing Measurements

**Test Date:** 2026-07-01  
**Hardware:** AMD Strix Halo (gfx1151, 128 GB unified memory)

---

## Measured Times (Actual Tests) ⏱️

| Asset Type | Engine/Model | Time | Notes |
|------------|--------------|------|-------|
| TTS (Kokoro, af_heart) | Kokoro-82M | **1.7 seconds** | Single sentence generation via API |
| TTS (Qwen3-TTS) | Qwen3-TTS-12Hz | **1.1 seconds** | Single phrase generation via API |

---

## Observed/Estimated Times 📊

| Asset Type | Engine/Model | Time | Notes |
|------------|--------------|------|-------|
| Image (ComfyUI) | LTX-2.3 distilled | **>120 seconds** ⚠️ | Test timed out; model loading likely needed first run |
| Music (HeartMuLa) | HeartMuLa 3B | **Not tested** ❌ | Container user permissions issue preventing execution |
| Video (LTX-2.3 I2V) | LTX-Video 2.3 | **Estimated 3-6 min** | Based on model research benchmarks |
| Video (Wan 2.2) | Wan 2.2 A14B MoE | **Estimated 5-8 min** | Based on model research benchmarks |

---

## Issues Found During Testing 🔧

### 1. HeartMuLa Container User Issue
HeartMuLa failed to run because the toolbox container lacks a proper user entry for UID 1000:
```
KeyError: 'getpwuid(): uid not found: 1000'
```
**Fix needed:** Add `USER` environment variable to toolbox container or ensure /etc/passwd has entry.

### 2. ComfyUI Image Generation Timeout
Image generation via ComfyUI workflow timed out after 120 seconds. This may be due to:
- First-time model loading (LTX-2.3 checkpoint ~4GB)
- Model not yet cached in VRAM
- Workflow submission issue

**Next test needed:** Pre-load models or use a simpler workflow to verify actual inference time.

---

## Recommendations for Accurate Benchmarks 📋

1. **Pre-warm GPU:** Run a dummy generation before timing to load models into VRAM
2. **Clear caches:** Delete torch cache between tests for consistent first-run measurement  
3. **Fix container users:** Ensure all containers have proper USER environment variables
4. **Test each model separately:** Don't test with multiple models in same session

---

## Next Steps 🚀

- [ ] Fix HeartMuLa user permissions and re-test timing
- [ ] Pre-warm ComfyUI and measure actual image generation time  
- [ ] Test video generation (LTX-2.3 I2V) with proper workflow
- [ ] Benchmark all models at different tiers/qualities if applicable

---

**Last Updated:** 2026-07-01  
**Status:** Partial - TTS measured, others need fixes
