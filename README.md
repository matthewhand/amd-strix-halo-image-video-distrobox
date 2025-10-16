```
███████╗████████╗██████╗ ██╗██╗  ██╗      ██╗  ██╗ █████╗ ██╗      ██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██║╚██╗██╔╝      ██║  ██║██╔══██╗██║     ██╔═══██╗
███████╗   ██║   ██████╔╝██║ ╚███╔╝       ███████║███████║██║     ██║   ██║
╚════██║   ██║   ██╔══██╗██║ ██╔██╗       ██╔══██║██╔══██║██║     ██║   ██║
███████║   ██║   ██║  ██║██║██╔╝ ██╗      ██║  ██║██║  ██║███████╗╚██████╔╝
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝ 

                         I M A G E   &   V I D E O                        
```

# AMD Strix Halo — Image & Video Distrobox

> **Note:** This is a forked and adapted version of the original [AMD Strix Halo Image & Video Toolbox](https://github.com/kyuz0/amd-strix-halo-image-video-toolboxes) for Distrobox compatibility on Ubuntu and other distros. The YouTube video and related content below refer to the original Toolbox setup. While instructions have been updated, some details may still reference the original—please verify and report issues for accuracy.

A **distrobox** image with a full **ROCm environment** for **image & video generation** on **AMD Ryzen AI Max “Strix Halo” (gfx1151)**. It includes support for **Qwen Image/Edit** and **WAN 2.2** models. Compatible with Ubuntu and other Linux distros via Distrobox. If you’re looking for sandboxes to run LLMs with llama.cpp, see: [https://github.com/kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)

> Tested on Framework Desktop (Strix Halo, 128 GB unified memory). Works on other Strix Halo systems (GMKtec EVO X-2, HP Z2 G1a, etc).

---

## Table of Contents

- [1. Overview](#1-overview)  
- [2. Watch the YouTube Video](#2-watch-the-youtube-video)  
- [3. 🚨 Updates — 2025-09-06](#3--updates--2025-09-06)  
- [4. Components (What’s Included)](#4-components-whats-included)  
- [5. Creating the Toolbox](#5-creating-the-toolbox)  
  - [5.1. Enter & Update](#51-enter--update)  
  - [5.2. Ubuntu Users and Toolkits](#52-ubuntu-users-and-toolkits)  
- [6. Unified Memory Setup](#6-unified-memory-setup)  
- [7. Qwen Image Studio](#7-qwen-image-studio)  
  - [7.1. Download Models](#71-download-models)  
  - [7.2. How to Start](#72-how-to-start)  
  - [7.3. Paths & Persistence](#73-paths--persistence)  
  - [7.4. Attention Backend & Speed (Qwen)](#74-attention-backend--speed-qwen)  
- [8. WAN 2.2](#8-wan-22)  
  - [8.1. Download Models](#81-download-models)  
  - [8.2. Video Generation Examples](#82-video-generation-examples)  
    - [8.2.1. Text-to-Video (T2V, Lightning)](#821-text-to-video-t2v-lightning)  
    - [8.2.2. Image-to-Video (I2V, Lightning)](#822-image-to-video-i2v-lightning)  
    - [8.2.3. Speech-to-Video (S2V, 14B)](#823-speech-to-video-s2v-14b)  
    - [8.2.4. TI2V 5B Checkpoint (not recommended)](#824-ti2v-5b-checkpoint-not-recommended)  
  - [8.3. Notes](#83-notes)  
  - [8.4. Attention Backend & Speed (WAN)](#84-attention-backend--speed-wan)  
- [9. ComfyUI](#9-comfyui)  
  - [9.1. Setup (ComfyUI only)](#91-setup-comfyui-only)  
  - [9.2. Run](#92-run)  
  - [9.3. Running Image/Video Workflows in ComfyUI](#93-running-imagevideo-workflows-in-comfyui)  
- [10. Stability and Performance Notes](#10-stability-and-performance-notes)  
- [11. Credits & Links](#11-credits--links)  

---

## 1. Overview

This distrobox provides a ROCm nightly stack for Strix Halo (gfx1151), built from [ROCm/TheRock](https://github.com/ROCm/TheRock), plus three main tools. **All model weights are stored outside the distrobox** (in your HOME), so they survive container deletion or refresh.

---

## 2. Watch the YouTube Video

[![Watch the YouTube Video](https://img.youtube.com/vi/7-E0a6sGWgs/maxresdefault.jpg)](https://youtu.be/7-E0a6sGWgs)

---

## 3. 🚨 Updates — 2025-09-06

### 3.1. 🔥 Performance Improvements

* **Qwen Image Studio** and **WAN 2.2** now use **tiled VAE decoding/encoding** phases.
  This significantly reduces memory pressure and improves speed and stability on Strix Halo.

### 3.2. 🆕 New Model: Speech-to-Video (S2V)

* Added support for **speech-to-video** in WAN 2.2 (14B checkpoint).
* No Lightning LoRA adapters yet — so inference requires \~40 steps,
* Still, it enables audio + image + prompt–based video generation.

---

## 4. Components (What’s Included)

| Component                                                                                          | Path                     | Purpose                                                |
| -------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------ |
| **Qwen Image Studio** ([fork of qwen-image-mps](https://github.com/ivanfioravanti/qwen-image-mps)) | `/opt/qwen-image-studio` | Web UI + job manager with retries, CLI still available |
| **WAN 2.2** ([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2))                              | `/opt/wan-video-studio`  | CLI for text-to-video / image-to-video                 |
| **ComfyUI** ([ComfyUI](https://github.com/comfyanonymous/ComfyUI))                                 | `/opt/ComfyUI`           | Node-based UI, AMD GPU monitor plugin                  |

> **Note:** Scripts in `/opt` (`set_extra_paths.sh`, `get_qwen_image.sh`, `get_wan22.sh`) are **for ComfyUI only**. Skip them unless you use ComfyUI.

---

## 5. Creating the Distrobox

A distrobox is a containerized user environment that shares your home directory and user account. To use this distrobox, you need to **expose GPU devices** and add your user to the right groups so ROCm and Vulkan have access to Strix Halo’s GPU nodes.

First, install Distrobox if not already:

```bash
# On Ubuntu/Debian
sudo apt update && sudo apt install distrobox podman

# Or via curl (universal)
curl -s https://raw.githubusercontent.com/89luca89/distrobox/main/install | sudo sh
```

Create the distrobox:

```bash
First, build the image locally (requires ROCm on host for proper GPU support):

```bash
git clone https://github.com/matthewhand/amd-strix-halo-image-video-distrobox.git
cd amd-strix-halo-image-video-distrobox
docker build -t amd-strix-halo-image-video-distrobox .
```

Then create the distrobox:

```bash
distrobox create strix-halo-image-video \
  --image amd-strix-halo-image-video-distrobox \
  --additional-flags "--device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined"
```
```

**Explanation**

* `--device /dev/dri` → graphics & video devices
* `--device /dev/kfd` → required for ROCm compute
* `--group-add video, render` → ensures user has GPU access
* `--security-opt seccomp=unconfined` → avoids syscall sandbox issues with GPUs

Enter the distrobox:

```bash
distrobox enter strix-halo-image-video
```

Inside, your prompt looks normal but you’re in the container with:

* Full ROCm stack
* All tools under `/opt`
* Shared `$HOME` (so models and outputs are persistent)

### 5.1. Enter & Update

This distrobox will be updated regularly with new nightly builds from TheRock for ROCm 7 and updated support for image and video generation.

You can use `refresh-toolbox.sh` to pull updates:

```bash
chmod +x refresh-toolbox.sh
./refresh-toolbox.sh
```

> \[!WARNING] ⚠️ **Refreshing deletes the current distrobox**
> Running `refresh-toolbox.sh` **removes and recreates** the distrobox image/container. This should be safe if you followed this README as all model files and outputs are saved **OUTSIDE** the distrobox in your home directory.
>
> ❌ **Lost (deleted)** — anything stored **inside the container**, e.g. `/opt/...` or other non-HOME paths.

### 5.2. Ubuntu Users and Distroboxes

To get this distrobox to work on Ubuntu, you need to create a udev rule to allow all users to use GPU devices.

Create `/etc/udev/rules.d/99-amd-kfd.rules`:

```
SUBSYSTEM=="kfd", GROUP="render", MODE="0666", OPTIONS+="last_rule"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666", OPTIONS+="last_rule"
```

Then reload udev rules:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Ensure your user is in the `render` and `video` groups:

```bash
sudo usermod -aG render,video $USER
# Log out and back in for group changes to take effect
```

---

## 6. Unified Memory Setup

On the host, enable unified memory with kernel parameters. This is required to make full use of system memory and run large models without having to statically allocate RAM to the GPU:

```
amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432
```

| Parameter                  | Purpose                      |
| -------------------------- | ---------------------------- |
| `amd_iommu=off`            | lower latency                |
| `amdgpu.gttsize=131072`    | 128 GiB GTT (unified memory) |
| `ttm.pages_limit=33554432` | large pinned allocations     |

Set BIOS to allocate minimal VRAM (e.g. 512 MB) and rely on unified memory.

On Ubuntu (or Fedora), set these in `/etc/default/grub` under `GRUB_CMDLINE_LINUX`, then run:

```bash
# On Ubuntu
sudo update-grub

# On Fedora
sudo grub2-mkconfig -o /boot/grub2/grub.cfg

sudo reboot
```

---

## 7. Qwen Image Studio

**Path:** `/opt/qwen-image-studio`
**Run:** `start_qwen_studio` (serves at [http://localhost:8000](http://localhost:8000))

### 7.1. Download Models

Before starting the UI, fetch model weights (done once; stored in HOME outside the toolbox).

List models:

```bash
cd /opt/qwen-image-studio
python /opt/qwen-image-studio/qwen-image-mps.py download
```

Fetch all variants in one go (⚠️ >80 GB):

```bash
cd /opt/qwen-image-studio/
python /opt/qwen-image-studio/qwen-image-mps.py download all
```

* Models go to `~/.cache/huggingface/hub/` (outside toolbox)
* Available: `qwen-image`, `qwen-image-edit`, `lightning-lora-8`, `lightning-lora-4`
* LoRA adapters require the base models first

Outputs and job state are kept in `~/.qwen-image-studio/` (HOME, outside the toolbox) so they persist across updates or rebuilds.

### 7.2. How to Start

Start the Web UI:

```bash
start_qwen_studio
```

This launches a FastAPI/uvicorn server on port 8000.
Local machine: open [http://localhost:8000](http://localhost:8000)
Over SSH:

```bash
ssh -L 8000:localhost:8000 user@your-strix-box
```

Under the hood:

```bash
cd /opt/qwen-image-studio && \
uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000
```

You can also check the console log to see the exact CLI commands executed for each job.

### 7.3. Paths & Persistence

All generated images and job metadata are stored under `~/.qwen-image-studio/` in your HOME (outside the toolbox), so they persist outside the toolbox.

### 7.4. Attention Backend & Speed (Qwen)

* **Default:** **PyTorch SDPA** (Scaled Dot-Product Attention) — **stable path**.
* **Optional speed-up:** enable **Triton FlashAttention** (\~2× faster) **before** running Qwen:

```bash
export QWEN_FA_SHIM=1
```

> ⚠️ **Stability note (gfx1151):** Triton kernels can still be **buggy** and **crash** more often. With SDPA (default) users should **not** see crashes related to attention.

---

## 8. WAN 2.2

**Path:** `/opt/wan-video-studio` (CLI only, Web UI planned)

WAN 2.2 is Alibaba’s open-sourced text-to-video and image-to-video model. This toolbox includes support for both the full A14B checkpoints and the **Lightning LoRA adapters** that allow **4-step inference** for much faster generation.

### 8.1. Download Models

Always store model weights in your HOME so they survive toolbox refreshes.

First, fetch the Lightning adapters:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download lightx2v/Wan2.2-Lightning --local-dir ~/Wan2.2-Lightning
```

**Full Checkpoints (needed alongside Lightning)**

* **Text-to-Video (T2V):**

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Wan-AI/Wan2.2-T2V-A14B --local-dir ~/Wan2.2-T2V-A14B
```

* **Image-to-Video (I2V):**

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Wan-AI/Wan2.2-I2V-A14B --local-dir ~/Wan2.2-I2V-A14B
```

### 8.2. Video Generation Examples

#### 8.2.1. Text-to-Video (T2V, Lightning)

```bash
cd /opt/wan-video-studio
python generate.py \
  --task t2v-A14B \
  --size "832*480" \
  --ckpt_dir ~/Wan2.2-T2V-A14B \
  --lora_dir ~/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1 \
  --offload_model False \
  --prompt "Close-up cinematic shot inside a futuristic microchip environment, focusing on a GPU core processing a glowing neural network. Streams of neon-blue data pulses flow across intricate circuits, nodes light up in sequence as if the chip is thinking. Camera slowly pans through the GPU architecture, highlighting cybernetic details. High-tech, sci-fi atmosphere, sharp digital glow, cinematic lighting. no text, no watermark, no distortion." \
  --frame_num 73 \
  --save_file ~/output.mp4
```

* `--size "832*480"` → reduced resolution for better runtime on Strix Halo
* `--frame_num 73` → required to be `4n+1`, \~3 sec video in \~30 min runtime
* `--lora_dir` → points to the Lightning LoRA adapter

#### 8.2.2. Image-to-Video (I2V, Lightning)

```bash
cd /opt/wan-video-studio
python generate.py \
  --task i2v-A14B \
  --size "832*480" \
  --ckpt_dir ~/Wan2.2-I2V-A14B \
  --lora_dir ~/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1 \
  --offload_model False \
  --prompt "Describe the scene and the required change to the input image." \
  --frame_num 73 \
  --image ~/input.jpg \
  --save_file ~/output.mp4
```

#### 8.2.3. Speech-to-Video (S2V, 14B)

Download the checkpoint:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Wan-AI/Wan2.2-S2V-14B --local-dir ~/Wan2.2-S2V-14B
```

Run generation:

```bash
cd /opt/wan-video-studio
python generate.py \
  --task s2v-14B \
  --size "832*480" \
  --offload_model False \
  --ckpt_dir ~/Wan2.2-S2V-14B/ \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --image ~/input_image.jpg \
  --audio ~/input_audio.mp3 \
  --save_file ~/output.mp4
```

* No Lightning LoRA adapters are available yet for S2V.
* This means inference requires \~40 steps, making generation **slower** than T2V/I2V with Lightning.
* Still, it enables synchronized **audio + image + prompt → video** workflows.

#### 8.2.4. TI2V 5B Checkpoint (not recommended)

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ~/Wan2.2-TI2V-5B
```

```bash
cd /opt/wan-video-studio
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ~/Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype \
  --prompt "Two cats boxing under a spotlight" \
  --frame_num 41 \
  --save_file ~/video.mp4
```

### 8.3. Notes

* Lightning adapters (LoRA) drastically reduce generation time (4 steps).
* Use smaller resolutions (`832*480`) to balance quality and runtime on Strix Halo.
* Keep all model files under HOME (`~/Wan2.2-*`) so they survive toolbox updates.
* Official Lightning repo: [https://huggingface.co/lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)

### 8.4. Attention Backend & Speed (WAN)

* **Default:** **Triton FlashAttention** is **ON by default** (video denoising is very expensive; speed matters).
* **Switch to SDPA (more stable):**

```bash
export WAN_ATTENTION_BACKEND=sdpa
```

**Speed example (21-frame video, 4 steps):**

Triton:

```
100%|██████████| 4/4 [01:37<00:00, 24.28s/it]
```

SDPA:

```
100%|██████████| 4/4 [04:30<00:00, 67.67s/it]
```

The difference is considerable, especially as the number of frames increases.

---

## 9. ComfyUI

**Path:** `/opt/ComfyUI`

ComfyUI is a flexible node-based interface for building and running image and video generation workflows. In this toolbox it is pre-cloned and configured with an AMD GPU monitor plugin.

### 9.1. Setup (ComfyUI only)

Before running ComfyUI, download model weights to `~/comfy-models` in your home directory.

```bash
# Run this FIRST to create ~/comfy-models and config file to point ComfyUI there
/opt/set_extra_paths.sh 

# Fetch model weights to ~/comfy-models
/opt/get_qwen_image.sh   # fetches Qwen Image models
/opt/get_wan22.sh        # fetches Wan2.2 models
```

These scripts ensure model files are downloaded to `~/comfy-models/` where they survive toolbox refreshes.

### 9.2. Run

Start ComfyUI inside the toolbox:

```bash
start_comfy_ui
```

Alias details:

```bash
/usr/local/bin/start_comfy_ui      # wraps ComfyUI launch with port checks
# Equivalent manual invocation:
cd /opt/ComfyUI
python main.py --listen 0.0.0.0 --port 8000 \
  --output-directory "$HOME/comfy-outputs" --disable-mmap
```

> You will see an error message for missing `torchaudio`: this is **temporarily** removed as its presence causes ComfyUI to crash on boot.

* Outputs appear under `~/comfy-outputs/` in your HOME.
* The launcher tries port 8000 for parity with Qwen Image Studio and falls back to 8188 if it is in use.
* Remote over SSH:

```bash
ssh -L 8000:localhost:8000 user@your-strix-box
```

Open [http://localhost:8000](http://localhost:8000) locally to access the web interface.

> [TIP] If Qwen Image Studio (also on port 8000) is already running, the wrapper automatically
> falls back to port **8188** and informs you of the change. To pick a different port manually,
> pass `start_comfy_ui --port 9000`.

> [TIP] The launcher now disables Triton FlashAttention (`FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE`)
> and enables tiled WAN VAE decoding by default. This avoids the ROCm GPU hangs observed when
> running the WAN 14B FP8 workflows on Strix Halo. Override by exporting different values before
> calling `start_comfy_ui` if you want to experiment.

Upstream project: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

### 9.3. Running Image/Video Workflows in ComfyUI

You can load ready-made workflow files directly into ComfyUI:

* Qwen Image example: [https://comfyanonymous.github.io/ComfyUI\_examples/qwen\_image/](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/)
* Wan2.2 example: [https://comfyanonymous.github.io/ComfyUI\_examples/wan22/](https://comfyanonymous.github.io/ComfyUI_examples/wan22/)

---

## 10. Stability and Performance Notes

Instability has been **significantly reduced**. When using **PyTorch SDPA** (the **default for Qwen Image Studio**), users generally **do not** encounter attention-related crashes. Issues are **more likely** when enabling **Triton FlashAttention** on gfx1151 (see Qwen §7.4 and WAN §8.4 if you opt into Triton).

### Known Issues
- **ROCm Import Error**: "ModuleNotFoundError: No module named '_rocm_sdk_core'" may occur on container entry if ROCm is not properly installed in the image.
- **Password Setup Error**: "Authentication token manipulation error" during Distrobox initialization—harmless, but can be resolved by running `passwd` inside the container.



If a crash occurs, you may still see messages like:

```
Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege.
```

or:

```
/opt/ComfyUI/comfy/ldm/qwen_image/model.py:153: UserWarning: HIP warning: an illegal memory access was encountered ...
!!! Exception during processing !!! HIP error: an illegal memory access was encountered
```

These are tracked here: [https://gitlab.freedesktop.org/drm/amd/-/issues/4321#note\_3048205](https://gitlab.freedesktop.org/drm/amd/-/issues/4321#note_3048205)
A fix is expected in **ROCm 7.0.x**. **Qwen Image Studio** is generally more stable than ComfyUI and includes automatic retries (each job up to 3 attempts). ComfyUI may need a relaunch if it crashes.

Also track these tickets for performance issues on Strix Halo:

* [https://github.com/ROCm/ROCm/issues/4748](https://github.com/ROCm/ROCm/issues/4748)
* [https://github.com/ROCm/ROCm/issues/4499](https://github.com/ROCm/ROCm/issues/4499)

---

## 11. Credits & Links

* Qwen Image (original CLI): [https://github.com/ivanfioravanti/qwen-image-mps](https://github.com/ivanfioravanti/qwen-image-mps)
* ComfyUI: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* WAN 2.2: [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
* ROCm FlashAttention (AMD fork): [https://github.com/ROCm/flash-attention](https://github.com/ROCm/flash-attention)
* Distrobox: [https://distrobox.it/](https://distrobox.it/)

---

**Notes on persistence:** All model weights and outputs are stored in your **HOME** outside the distrobox (e.g., `~/.cache/huggingface/hub/`, `~/.qwen-image-studio/`, `~/Wan2.2-*`, `~/comfy-models`, `~/comfy-outputs`). This ensures they survive distrobox refreshes.

---

## 12. Debugging & Compatibility Patches

This section documents the current workarounds and patches needed for Qwen Image Studio to work properly in the distrobox environment. These patches address compatibility issues between the Qwen implementation and the ROCm/AMD GPU stack.

### 12.1. Current Patches Applied

#### 12.1.1. `offload_state_dict` Removal Patch
**Files**: `start_qwen_studio_patched.py`, `patched_cli_runner.py`

**Problem**: The `offload_state_dict` parameter in `DiffusionPipeline.from_pretrained()` and `Qwen2_5_VLForConditionalGeneration.__init__()` causes compatibility issues with the current ROCm stack.

**Solution**: Monkey patch these methods to remove the problematic parameter:

```python
# Pipeline-level patch
def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    if 'offload_state_dict' in kwargs:
        kwargs.pop('offload_state_dict')
    return original_from_pretrained(pretrained_model_name_or_path, **kwargs)

# Model-level patch
def patched_init(self, config, *args, **kwargs):
    if 'offload_state_dict' in kwargs:
        kwargs.pop('offload_state_dict')
    original_init(self, config, *args, **kwargs)
```

**Status**: ✅ Working - Required for both CLI and Web UI

#### 12.1.2. Distrobox Path Handling Fix
**File**: `start_qwen_studio_patched.py`

**Problem**: Web UI startup failed due to incorrect import paths when running from workspace directory instead of Qwen installation directory.

**Solution**: Change to correct directory before importing:

```python
import os
os.chdir('/opt/qwen-image-studio')
from qwen_image_studio.server import app
```

**Status**: ✅ Working - Required for Web UI startup

### 12.2. Legacy Patch Files (Deprecated)

The following patch files were created during debugging but are no longer needed with the current solution:

- `diffusers_patch.py` - Legacy diffusers patch
- `transformers_fix.py` - Legacy transformers patch
- `ultimate_fix.py` - Legacy comprehensive patch
- `comprehensive_qwen_fix.py` - Legacy fix attempt

**Status**: ❌ Deprecated - Can be removed

### 12.3. Test Files (Reference Only)

These files were used during debugging and can serve as reference for testing:

- `polished_qwen_test_final.py` - Working 4-step generation test
- `final_working_qwen.py` - Original working test
- `working_qwen_demo.py` - Demo implementation
- `direct_qwen_test.py` - Direct CLI test
- `simple_qwen_test.py` - Simple test case
- `ultra_simple_test.py` - Minimal test
- `working_qwen_launcher.py` - Alternative launcher

**Status**: 📋 Reference - Can be kept for testing

### 12.4. Future Compatibility

These patches may become unnecessary when upstream libraries are updated:

#### When to Remove Patches

1. **`offload_state_dict` Patch**: Monitor updates to:
   - `diffusers` library (check `DiffusionPipeline.from_pretrained`)
   - `transformers` library (check `Qwen2_5_VLForConditionalGeneration.__init__`)
   - Qwen Image Studio updates that handle this parameter gracefully

2. **Path Handling Fix**: Monitor updates to:
   - Qwen Image Studio startup scripts
   - Distrobox integration improvements

#### Testing Patch Removal

To test if patches are still needed:

1. Comment out patch applications in `start_qwen_studio_patched.py`
2. Try starting Web UI: `python3 start_qwen_studio_patched.py`
3. Try CLI generation: `python3 patched_cli_runner.py generate -p "test" --steps 4`
4. If both work without errors, patches can be removed

#### Version Tracking

Current working versions (as of 2025-10-11):
- ROCm: 7.0.x (nightly from TheRock)
- Python: 3.13
- PyTorch: ROCm-enabled version
- Qwen Image Studio: Fork of qwen-image-mps

### 12.5. Troubleshooting

If issues occur after updating libraries:

1. First check if patches are still applied correctly
2. Look for new error messages that might indicate changed APIs
3. Test with the reference files in section 12.3
4. Re-apply patches if necessary using the patterns in section 12.1

**Note**: Always backup working configurations before updating libraries or removing patches.
