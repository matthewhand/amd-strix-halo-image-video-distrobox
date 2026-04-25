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

A **distrobox / Docker** image with a full **ROCm environment** for **image & video generation** on **AMD Ryzen AI Max “Strix Halo” (gfx1151)**. It includes support for **Qwen Image/Edit**, **WAN 2.2**, and **LTX-2** models. Compatible with Ubuntu and other Linux distros via Distrobox or Docker Compose.

> Forked from the original [AMD Strix Halo Image & Video Toolbox](https://github.com/kyuz0/amd-strix-halo-image-video-toolboxes) (Fedora Toolbox). See [Background](#2-background) for what changed and why. If you’re looking for LLM sandboxes with llama.cpp, see: [amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes).

> Tested on Framework Desktop (Strix Halo, 128 GB unified memory). Works on other Strix Halo systems (GMKtec EVO X-2, HP Z2 G1a, etc).

---

## Table of Contents

- [1. Overview](#1-overview)  
- [2. Background](#2-background)  
- [3. Components (What’s Included)](#3-components-whats-included)  
- [4. Creating the Distrobox](#4-creating-the-distrobox)  
  - [4.1. Enter & Update](#41-enter--update)  
  - [4.2. Running as a Service (Docker Compose)](#42-running-as-a-service-docker-compose)  
  - [4.3. GUI Support (Distrobox)](#43-gui-support-distrobox)  
- [5. Unified Memory Setup](#5-unified-memory-setup)  
- [6. Qwen Image Studio](#6-qwen-image-studio)  
  - [6.1. Download Models](#61-download-models)  
  - [6.2. How to Start](#62-how-to-start)  
  - [6.3. Paths & Persistence](#63-paths--persistence)  
  - [6.4. Attention Backend & Speed (Qwen)](#64-attention-backend--speed-qwen)  
- [7. WAN 2.2](#7-wan-22)  
  - [7.1. Download Models](#71-download-models)  
  - [7.2. Video Generation Examples](#72-video-generation-examples)  
  - [7.3. Notes](#73-notes)  
  - [7.4. Attention Backend & Speed (WAN)](#74-attention-backend--speed-wan)  
- [8. ComfyUI](#8-comfyui)  
  - [8.1. Setup (ComfyUI only)](#81-setup-comfyui-only)  
  - [8.2. Run](#82-run)  
  - [8.3. Running Image/Video Workflows in ComfyUI](#83-running-imagevideo-workflows-in-comfyui)  
- [9. Stability & Technical Notes](#9-stability--technical-notes)  
- [10. Credits & Links](#10-credits--links)  

---

## 1. Overview

A ROCm nightly stack for Strix Halo (gfx1151), built from [ROCm/TheRock](https://github.com/ROCm/TheRock), with three generation tools and a workflow UI. All model weights are stored **outside the container** so they survive rebuilds.

### Distrobox vs Docker Compose

Both modes use the **same container image** built from the same Dockerfile. The difference is how you run it:

| | Distrobox | Docker Compose |
|---|-----------|----------------|
| **Use case** | Desktop / interactive — run tools on demand | Headless server / NAS — always-on services |
| **Startup** | `distrobox enter ...` then run commands manually | `docker compose up -d` starts everything |
| **HOME sharing** | Full HOME is shared automatically | Explicit volume mounts per directory |
| **User** | Runs as your host user | Runs as root inside the container |
| **Services** | Manual via shell aliases (`start_qwen_studio`, `start_comfy_ui`) | Auto-started by `start_docker.sh` with restart monitoring |
| **GUI access** | X11/Wayland forwarded (can open image viewers) | Web UI only (ports 8000, 8188) |

### Volume mapping (Docker Compose)

All data is stored in **project-local folders** (gitignored), keeping your HOME clean:

```
Project directory                       Container
─────────────────                       ─────────
./huggingface-cache/   ──────────────▶  /root/.cache/huggingface/   (Qwen model cache)
./qwen-outputs/        ──────────────▶  /root/.qwen-image-studio/   (Qwen outputs + jobs)
./wan-models/          ──────────────▶  /root/wan-models/            (WAN checkpoints + LoRA)
./comfy-models/        ──────────────▶  /opt/ComfyUI/models/         (ComfyUI models)
./comfy-outputs/       ──────────────▶  /opt/ComfyUI/output/         (ComfyUI outputs)
/dev/dri, /dev/kfd     ──────────────▶  GPU device access            (ROCm/HIP)
```

In Distrobox, these mounts are unnecessary — your entire HOME is shared, so all `~/` paths just work directly.

### Model storage

| What | Docker Compose (project-local) | Distrobox (HOME) |
|------|-------------------------------|------------------|
| Qwen models (HuggingFace cache) | `./huggingface-cache/` | `~/.cache/huggingface/` |
| Qwen outputs + job state | `./qwen-outputs/` | `~/.qwen-image-studio/` |
| WAN checkpoints + Lightning LoRA | `./wan-models/` | `~/wan-models/` (or anywhere) |
| ComfyUI models | `./comfy-models/` | `~/comfy-models/` |
| ComfyUI outputs | `./comfy-outputs/` | `~/comfy-outputs/` |

### Services & ports

| Service | Default port | Managed by |
|---------|-------------|------------|
| Qwen Image Studio | 8000 | `start_docker.sh` / `QWEN_PORT` env var |
| ComfyUI | 8188 | `start_docker.sh` / `COMFYUI_PORT` env var |

Set a port to `0` in docker-compose to disable that service.

---

## 2. Background

This project is a fork of [kyuz0/amd-strix-halo-image-video-toolboxes](https://github.com/kyuz0/amd-strix-halo-image-video-toolboxes), which was built for **Fedora Toolbox**. The adaptation to **Ubuntu + Distrobox** (and Docker Compose) required working through a series of ROCm compatibility problems on gfx1151:

1. **Fedora Toolbox to Distrobox/Docker** — The original project assumed Fedora’s `toolbox` command. This fork replaces that with [Distrobox](https://distrobox.it/) (works on any distro) and adds a `docker-compose.yaml` for persistent server deployments.

2. **ROCm nightly stack (TheRock)** — Strix Halo (gfx1151) is not yet supported by stable ROCm releases. The container pulls nightly builds from [ROCm/TheRock](https://github.com/ROCm/TheRock) targeting gfx1151 specifically. This required fixing `HSA_OVERRIDE_GFX_VERSION` to `11.5.1` (not `11.0.0` as some guides suggest).

3. **`offload_state_dict` crash** — ROCm 7.10+ removed support for the `offload_state_dict` parameter in diffusers/transformers. Loading Qwen models crashes without monkey-patching it out. The fix lives in `scripts/apply_qwen_patches.py`.

4. **QwenImagePipeline segfault** — Accessing `vae.temperal_downsample` during pipeline init triggers a segfault on ROCm. The same patch file hardcodes `vae_scale_factor=8` to avoid it.

5. **Flash Attention shim** — The ROCm flash-attention build needs a CUDA compatibility shim injected at import time. The Qwen and WAN launchers (`scripts/qwen_launcher.py`, `scripts/wan_launcher.py`) handle this automatically.

6. **Kernel 6.18+ required** — The in-tree `amdgpu` driver gained gfx1151 support in kernel 6.18. Earlier kernels will not detect the GPU.

> The [original YouTube walkthrough](https://youtu.be/7-E0a6sGWgs) covers the Fedora Toolbox setup. The concepts are the same, but the commands in this README reflect the Distrobox/Docker workflow.

---

## 3. Components (What’s Included)

| Component                                                                                          | Path                     | Purpose                                                |
| -------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------ |
| **Qwen Image Studio** ([fork of qwen-image-mps](https://github.com/ivanfioravanti/qwen-image-mps)) | `/opt/qwen-image-studio` | Web UI + job manager with retries, CLI still available |
| **WAN 2.2** ([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2))                              | `/opt/wan-video-studio`  | CLI for text-to-video / image-to-video                 |
| **ComfyUI** ([ComfyUI](https://github.com/comfyanonymous/ComfyUI))                                 | `/opt/ComfyUI`           | Node-based UI, AMD GPU monitor plugin                  |

> **Note:** Scripts in `/opt` (`set_extra_paths.sh`, `get_qwen_image.sh`, `get_wan22.sh`) are **for ComfyUI only**. Skip them unless you use ComfyUI.

---

## 4. Creating the Distrobox

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

Ensure distrobox has access to the image:

```bash
podman images | grep amd-strix-halo-image-video-distrobox
```

If no results are found, export the image from docker and import to podman:

```bash
docker save -o strix_image.tar amd-strix-halo-image-video-distrobox:latest

podman load -i strix_image.tar
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

### 4.1. Enter & Update

This distrobox will be updated regularly with new nightly builds from TheRock for ROCm 7 and updated support for image and video generation.

You can use `refresh-toolbox.sh` to pull updates:

```bash
chmod +x refresh-toolbox.sh
./refresh-toolbox.sh
```

> [[!WARNING] ⚠️ **Refreshing deletes the current distrobox**
> Running `refresh-toolbox.sh` **removes and recreates** the distrobox image/container. This should be safe if you followed this README as all model files and outputs are saved **OUTSIDE** the distrobox in your home directory.
>
> ❌ **Lost (deleted)** — anything stored **inside the container**, e.g. `/opt/...` or other non-HOME paths.

### 4.2. Running as a Service (Docker Compose)

For a persistent server setup (instead of interactive Distrobox), you can use Docker Compose. This starts both Qwen Image Studio and ComfyUI automatically.

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f
```

**Configuration (`.env` or `docker-compose.override.yaml`):**

You can customize ports or disable specific services using environment variables:

```yaml
services:
  strix-halo-toolbox:
    environment:
      - QWEN_PORT=8001      # Change Qwen port (default: 8000)
      - COMFYUI_PORT=0      # Disable ComfyUI (default: 8188)
```

### 4.3. GUI Support (Distrobox)

If you need X11/Wayland image viewers (like `feh`, `imv`) inside your Distrobox, build with `INSTALL_GUI=true`:

```bash
docker build --build-arg INSTALL_GUI=true -t amd-strix-halo-image-video-distrobox .
```

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

## 5. Unified Memory Setup

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

## 6. Qwen Image Generation

Qwen Image (54GB model) generates high-quality images from text prompts. Runs entirely on the Strix Halo GPU via ROCm.

### 6.1. Download Models

```bash
# Inside the container (distrobox or docker exec)
cd /opt/qwen-image-studio
python qwen-image-mps.py download all    # ~80 GB total
```

Models go to `~/.cache/huggingface/hub/`. Available: `qwen-image`, `qwen-image-edit`, `lightning-lora-8`, `lightning-lora-4`.

### 6.2. Generate Images (Script)

The most reliable way to generate images is via Docker directly:

```bash
docker run --rm \
  --device /dev/dri --device /dev/kfd \
  --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/qwen-outputs:/root/.qwen-image-studio \
  amd-strix-halo-image-video-toolbox:latest \
  python3 -c "
import sys
sys.path.insert(0, '/opt/qwen-image-studio/src')
sys.path.insert(0, '/opt')
from apply_qwen_patches import apply_comprehensive_patches
apply_comprehensive_patches()
from qwen_image_mps.cli import generate_image

class Args:
    prompt = 'YOUR PROMPT HERE'
    steps = 8          # 4 = fast, 8 = better quality
    num_images = 1
    size = '16:9'      # or '1:1', '9:16'
    ultra_fast = False  # True for 4-step
    model = 'Qwen/Qwen-Image'
    no_mmap = True
    lora = None
    edit = False
    input_image = None
    output_dir = '/tmp'
    seed = 42
    guidance_scale = 1.0
    negative_prompt = 'blurry, low quality, distorted, watermark'
    batman = False
    fast = False
    targets = 'all'

generate_image(Args())
"
```

Typical performance: **~2 minutes** per image (14s inference + 47s VAE decode + model loading).

### 6.3. Generate Images (Web UI)

For interactive use, the Qwen Image Studio web UI is also available:

```bash
# Distrobox
start_qwen_studio

# Docker Compose
docker compose up -d   # starts on port 8000
```

The container automatically applies ROCm compatibility patches (`scripts/apply_qwen_patches.py`).

### 6.4. Attention Backend & Speed

* **Default:** PyTorch SDPA — stable
* **Faster:** set `QWEN_FA_SHIM=1` for Triton FlashAttention (~2x faster, less stable on gfx1151)

### 6.5. Test Scripts

```bash
python tests/test_qwen_generation.py     # single image smoke test
python tests/test_qwen_variations.py     # multiple prompts/settings
python tests/test_waldo_birdseye.py      # birds-eye puzzle images
```

---

## 7. WAN 2.2

**Path:** `/opt/wan-video-studio` (CLI only, Web UI planned)

WAN 2.2 is Alibaba’s open-sourced text-to-video and image-to-video model. This toolbox includes support for both the full A14B checkpoints and the **Lightning LoRA adapters** that allow **4-step inference** for much faster generation.

### 7.1. Download Models

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

### 7.2. Video Generation Examples

#### 7.2.1. Text-to-Video (T2V, Lightning)

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

#### 7.2.2. Image-to-Video (I2V, Lightning)

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

#### 7.2.3. Speech-to-Video (S2V, 14B)

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

#### 7.2.4. TI2V 5B Checkpoint (not recommended)

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

### 7.3. Notes

* Lightning adapters (LoRA) drastically reduce generation time (4 steps).
* Use smaller resolutions (`832*480`) to balance quality and runtime on Strix Halo.
* Keep all model files under HOME (`~/Wan2.2-*`) so they survive toolbox updates.
* Official Lightning repo: [https://huggingface.co/lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)

### 7.4. Attention Backend & Speed (WAN)

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

## 8. LTX-2 Video Generation

LTX-2 generates video (with optional audio) from text prompts or input images, via ComfyUI's API.

### 8.1. Generate Video (Script)

Start ComfyUI, then submit workflows via the API:

```bash
# Start ComfyUI
docker run -d --name comfyui \
  --device /dev/dri --device /dev/kfd \
  --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib \
  -p 8188:8188 \
  -v ~/comfy-models:/opt/ComfyUI/models \
  -v ~/comfy-outputs:/opt/ComfyUI/output \
  amd-strix-halo-image-video-toolbox:latest \
  bash -c 'cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output --disable-mmap'

# Generate a workflow and submit it
python scripts/generate_ltx_workflow.py --prompt "your prompt" --output workflow.json
python scripts/comfyui_api.py workflow.json
```

### 8.2. Image to Video (Qwen → LTX-2 Pipeline)

Generate a still image with Qwen, then animate it with LTX-2:

```bash
# Step 1: Generate image with Qwen (see section 6)
# Step 2: Copy image into ComfyUI input
docker cp my_image.png comfyui:/opt/ComfyUI/input/

# Step 3: Submit image-to-video workflow
python tests/test_qwen_to_ltx2.py
```

The `LTXVImgToVideo` node takes the image as the first frame and generates motion + audio from a text prompt.

### 8.3. Convert Frames to MP4

LTX-2 outputs PNG frames. Convert with ffmpeg (inside the container):

```bash
docker exec comfyui bash -c 'ffmpeg -y -framerate 24 \
  -i /opt/ComfyUI/output/ltx2_output_%05d_.png \
  -c:v libsvtav1 -pix_fmt yuv420p -crf 30 \
  /opt/ComfyUI/output/output.mp4'
```

### 8.4. Required Models

LTX-2 needs these in `~/comfy-models/`:
- `checkpoints/ltx-2-19b-dev-fp8.safetensors` (27GB)
- `text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/` (multi-shard)

### 8.5. Test Scripts

```bash
python tests/test_ltx2_variations.py      # text-to-video (various resolutions/lengths)
python tests/test_ltx2_audio_video.py     # video with generated audio
python tests/test_qwen_to_ltx2.py         # image-to-video pipeline
```

### 8.6. Performance

| Setting | Time | Notes |
|---------|------|-------|
| 768x512, 49 frames (~2s) | ~6 min | Standard |
| 768x512, 97 frames (~4s) | ~12 min | Recommended |
| 768x512, 145 frames (~6s) | ~18 min | Long |
| 768x512, 241 frames (~10s) | ~30 min | May OOM |

Use tiled VAE decode (`spatial_tiles: 4`) to avoid OOM on longer videos.

---

## 9. ComfyUI (Web UI)

ComfyUI is also available as an interactive web UI at **http://localhost:8188** for drag-and-drop workflow building.

For detailed information on the symlink hacks, CPU offloading, and kernel requirements (6.18+) required for Strix Halo, see:

👉 **[COMFYUI_SETUP_GUIDE.md](COMFYUI_SETUP_GUIDE.md)**

---

## 10. Strix Halo Benchmarks

Measured on Framework Desktop (Ryzen AI Max 395, 128 GB unified memory, kernel 6.19.6-zabbly+, ROCm 7.13.0 nightly via TheRock, PyTorch 2.10.0).

### Image Generation (Qwen)

| Steps | Aspect | Output resolution | Inference | VAE decode | Total | File size |
|-------|--------|------------------|-----------|------------|-------|-----------|
| 4 | 16:9 | 1664x928 | ~14s | ~47s | ~120s | 1.7 MB |
| 8 | 16:9 | 1664x928 | ~30s | ~47s | ~140s | 2.4 MB |
| 8 | 1:1 | 1328x1328 | ~35s | ~50s | ~150s | 2.8 MB |

Model: Qwen/Qwen-Image (54 GB). VRAM usage: ~54 GB during inference. 4-step uses Lightning LoRA for faster generation at slightly lower quality.

### Video Generation (LTX-2 via ComfyUI)

| Resolution | Frames | Duration | Generation time | File size |
|-----------|--------|----------|----------------|-----------|
| 768x512 | 97 | ~4s | ~12 min | 420-496 KB |
| 848x480 | 97 | ~4s | ~12 min | 569 KB |
| 768x512 | 145 | ~6s | ~18 min | 592-860 KB |
| 768x512 | 193 | ~8s | ~24 min | TBD |
| 768x512 | 241 | ~10s | ~30 min | TBD (may OOM) |

Model: ltx-2-19b-dev-fp8.safetensors (27 GB) + Gemma 3 12B text encoder. Uses tiled VAE decode (`spatial_tiles: 4`) to avoid OOM.

### WAN 2.2 (CLI)

| Mode | Resolution | Frames | Steps | Time |
|------|-----------|--------|-------|------|
| T2V Lightning | 832x480 | 73 | 4 | ~30 min |
| I2V Lightning | 832x480 | 73 | 4 | ~30 min |
| S2V (no LoRA) | 832x480 | 73 | 40 | ~5 hours |

Model: Wan2.2-T2V-A14B / I2V-A14B (~28 GB each) + Lightning LoRA (~2 GB).

---

## 11. Stability & Technical Notes

For detailed information on the symlink hacks, CPU offloading, and kernel requirements (6.18+) required for Strix Halo, see the:

👉 **[COMFYUI_SETUP_GUIDE.md](COMFYUI_SETUP_GUIDE.md)**

### Project layout

All helper scripts live under `scripts/`. The Dockerfile copies them into the container at build time. Nothing in the repo root is needed at runtime.

| Script | Purpose |
|--------|---------|
| `apply_qwen_patches.py` | ROCm monkey-patches for Qwen (offload_state_dict, segfault fix) |
| `qwen_launcher.py` | Qwen CLI wrapper with flash-attention shim |
| `wan_launcher.py` | WAN CLI wrapper with flash-attention shim |
| `start_docker.sh` | Docker Compose entrypoint (starts Qwen + ComfyUI) |
| `patch_gemma_loader.py` | Force Gemma encoder to CPU (LTX-2 workaround, `--revert` to undo) |
| `generate_ltx_workflow.py` | Generate ComfyUI API workflow JSON for LTX-2 |
| `comfy_model_manager.py` | Download/manage ComfyUI models |
| `diagnose.sh` | Validate ROCm setup and GPU detection |

---

## 12. Roadmap

Patches and workarounds that may become unnecessary as ROCm matures:

| Hack | Why it exists | When to remove |
|------|--------------|----------------|
| `offload_state_dict` monkey-patch | ROCm 7.10+ broke this parameter in diffusers | When diffusers removes the parameter upstream |
| `vae.temperal_downsample` segfault fix | QwenImagePipeline init crashes on ROCm | When Qwen/diffusers fix the typo and the access pattern |
| `LIBRARY_PATH` for aiter JIT | TheRock nightly doesn't set linker paths | When TheRock packages configure `ldconfig` properly |
| Gemma CLIP CPU offload (LTX-2) | ROCm kernel crash in embedding layer | When amdgpu driver fixes the gfx1151 kernel |
| torchaudio stub (ComfyUI) | torchaudio crashes on import | When TheRock ships a compatible torchaudio build |
| `HSA_OVERRIDE_GFX_VERSION=11.5.1` | gfx1151 not in stable ROCm | When AMD adds gfx1151 to official ROCm |

Performance improvements to investigate:
- **VAE decode on GPU**: currently ~47s on CPU for Qwen, could be much faster on GPU if MIOpen kernels stabilise
- **Persistent model loading**: reloading 54GB per image wastes ~60s; a long-running inference server would amortise this
- **Flash Attention**: `QWEN_FA_SHIM=1` enables ~2x speedup but Triton kernels are unstable on gfx1151
- **fp8 quantisation**: Qwen currently runs at full precision; fp8 could halve memory and improve speed
- **Multi-GPU**: Strix Halo is single-GPU but future APUs may support split workloads

---

## 13. Credits & Links

* Qwen Image (original CLI): [https://github.com/ivanfioravanti/qwen-image-mps](https://github.com/ivanfioravanti/qwen-image-mps)
* ComfyUI: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* WAN 2.2: [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
* ROCm FlashAttention (AMD fork): [https://github.com/ROCm/flash-attention](https://github.com/ROCm/flash-attention)
* Distrobox: [https://distrobox.it/](https://distrobox.it/)

---

**Notes on persistence:** All model weights and outputs are stored in your **HOME** outside the distrobox (e.g., `~/.cache/huggingface/hub/`, `~/.qwen-image-studio/`, `~/Wan2.2-*`, `~/comfy-models`, `~/comfy-outputs`). This ensures they survive distrobox refreshes.
