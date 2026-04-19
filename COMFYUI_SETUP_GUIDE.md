# 🎨 ComfyUI Setup & Maintenance Guide (Strix Halo)

This guide provides a comprehensive technical manual for running ComfyUI on AMD Strix Halo (gfx1151) hardware.

## 🚀 Quick Start

### 1. Start ComfyUI
```bash
# Recommendation: Use the complete launcher which handles the ROCm environment
distrobox enter strix-halo-image-video -- python3 ~/amd-strix-halo-image-video-distrobox/start_comfyui_complete.py
```

### 2. Access Web Interface
Open your browser to: **http://localhost:8188**

---

## 🛠️ **The "Technical Hacks" Registry**

Due to the bleeding-edge nature of Strix Halo (gfx1151), several "hacks" are required to keep models running stably.

### 1. Host Kernel Requirement
- **Kernel Version**: **6.18.8-zabbly+** (or newer)
- **Why**: Fixes a critical "VGPR Mismatch" bug. Without this, the GPU will crash (Memory Access Fault) at 0% of any complex sampling job.
- **Maintenance**: If you update your OS, ensure you stay on a Zabbly-compatible kernel.

### 2. Gemma 3 Text Encoder (NATIVE GPU)
- **Status**: Running natively on GPU with Kernel 6.18+.
- **Reverted Patch**: The CPU offload hack has been removed.

### 3. Tokenizer & Multi-Shard Symlinking
- **Problem**: Stock loaders expect a single `.safetensors` file. Gemma 3 is split into 5 large shards + a separate `tokenizer.model` file.
- **The Hack**: We use **dual symlinks** in `models/text_encoders/` to trick the UI and the loaders:
    ```bash
    # Folder symlink (provides tokenizer.model to the code)
    ln -sf gemma-3-12b-it-qat-q4_0-unquantized gemma_3_12B_it

    # File symlink (tells the UI the model "exists")
    ln -sf gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors gemma_3_12B_it.safetensors
    ```

### 4. Audio Nodes Fix
- **Problem**: `torchaudio` has a version mismatch that crashes the entire Python process on boot.
- **The Hack**: `torchaudio` is uninstalled from the venv, and `audio_vae.py` is patched to make the import optional.

---

## 🛑 Critical: OOM Protection (VAE Decoding)

Even with 128GB of RAM, generating long LTX-2 videos (41+ frames) can trigger the **System OOM Killer** during the final decoding stage.

- **The Issue**: Loading the Text Encoder (49GB) + DiT (20GB) + standard VAE Decoding creates a peak memory draw that can exceed 128GB.
- **The Solution**: **DO NOT** use the standard `LTX VAE Decode` node. 
- **Use This Node Instead**: **`LTXV Spatio-Temporal Tiled VAE Decode`**
- **Recommended Settings**: 
    - `spatial_tiles`: 4
    - `temporal_tile_length`: 16
    - `working_device`: auto (or cpu if still hitting OOM)

---

## 🔗 Model Directory Structure

We use `/mnt/data` (mapped to `~/comfy-models`) to host models across multiple tools.

```text
/opt/ComfyUI/models/
├── checkpoints/           # Diffusion models
├── vae/                   # VAE models
└── text_encoders/         # Gemma, CLIP, T5
    ├── gemma_3_12B_it -> DIR SYMLINK ✅
    └── gemma_3_12B_it.safetensors -> FILE SYMLINK ✅
```

---

## 📋 Maintenance Checklist

If you ever **Refresh/Update** the toolbox, you MUST run these scripts to restore functionality:

1.  **Restore Links**: `/opt/set_extra_paths.sh`
2.  **Restore Gemma Fix**: `python3 patch_gemma_loader.py`
3.  **Restore Tokenizer Fix**: `python3 patch_lt_tokenizer.py`

---

## 📊 Performance Benchmark (LTX-2)
- **Generation Time**: ~6.3 minutes for 49 frames.
- **GPU Power**: ~90W active.
- **Stability**: Tested 100% stable on Kernel 6.18.8.