# Compatibility & Interoperability Matrix

This matrix defines the supported resolutions, aspect ratios, and configurations for the various image and video models running on the AMD Strix Halo (gfx1151).

## 1. Video Models (I2V / T2V)

| Model | Task | Supported Resolutions (WxH) | Notes |
| :--- | :--- | :--- | :--- |
| **WAN 2.2 (14B)** | I2V | `1280*720`, `720*1280`, `832*480`, `480*832` | **Strict Validation:** Fails if not in this list. |
| **WAN 2.2 (14B)** | T2V | `1280*720`, `720*1280`, `832*480`, `480*832` | Uses Lightning 4-step or full A14B. |
| **LTX-2 / 2.3** | Video | Multiples of 32 (W, H) and 8 (Frames) | **Flexible:** Standard is `1216*704` or `768*512`. |
| **Hunyuan Video** | Video | `1280*720`, `848*480` | *Placeholder - Integration pending.* |

## 2. Image Models

| Model | Supported Resolutions (WxH) | Recommended | Notes |
| :--- | :--- | :--- | :--- |
| **Ernie-Image** | **≤ `512*512` on gfx1151** | `512*512` | **GPU-hangs above 512²** — CK grouped-conv kernel wedges the GPU during VAE decode (not a lib-path issue; symlink repair ruled out). Stay ≤512² (launcher/operator mitigation). See `docs/gfx1151-known-issues.md`. |
| **Qwen-Image** | Flexible (up to ~2048px) | `1664*928`, `1024*1024` | **Priority #2:** Very stable on ROCm. |
| **LTX-2.3 (Base)** | Multiples of 32 | `1280*720` | **Priority #3:** Used as fallback cinematic base. |

## 3. Aspect Ratio Conversions

When chaining models (e.g., LTX-2.3 Image -> WAN 2.2 Video), ensure the base image matches one of the **Strict Supported Sizes** of the downstream video model.

| Target Video Aspect | Recommended Base Resolution | Model Alignment |
| :--- | :--- | :--- |
| **16:9 (High)** | `1280*720` | WAN 2.2 / LTX-2.3 |
| **9:16 (Vertical)** | `720*1280` | WAN 2.2 |
| **16:9 (Low)** | `832*480` | WAN 2.2 / LTX-2.3 |
| **Wide cinematic** | `1216*704` | LTX-2.3 (Preferred) |

## 4. Hardware Constraints (Strix Halo 128GB)

- **Max Resolution:** `1280*720` is the currently tested stable limit for 73+ frame video generation.
- **VRAM Pressure:** Chaining at `1280*720` uses ~88% of Unified Memory. Avoid loading large LLMs (30B+) in FP16 simultaneously; use 4-bit quantizations via LM Studio for better headroom.
