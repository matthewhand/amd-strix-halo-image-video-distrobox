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

## 6. Qwen Image Studio

**Path:** `/opt/qwen-image-studio`
**Run:**
- `start_qwen_studio` (standard version)
- `start_qwen_studio_patched` (with ROCm 7.10+ compatibility fixes)

**Status:** See [QWEN_STATUS.md](QWEN_STATUS.md) for detailed status and known issues.

**Quick Summary:**
- ✅ Pipeline initialization works
- ✅ Components load to GPU successfully (53.79 GB / 128 GB)
- ✅ Image generation works with `HSA_OVERRIDE_GFX_VERSION=11.5.1`

### 6.1. Download Models

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

### 6.2. How to Start

#### Standard Web UI

```bash
start_qwen_studio
```

#### Patched Web UI (Recommended for AMD GPU)

```bash
start_qwen_studio_patched
```

The patched version applies critical ROCm 7.10+ compatibility fixes:
- Removes `offload_state_dict` from pipeline/model loading
- Fixes segfault in `QwenImagePipeline.__init__`

Both launch a FastAPI/uvicorn server on port 8000.
Local machine: open [http://localhost:8000](http://localhost:8000)
Over SSH:

```bash
ssh -L 8000:localhost:8000 user@your-strix-box
```

**Under the hood (patched):**

```bash
cd /opt/qwen-image-studio && \
QIM_CLI_PATH=/opt/qwen-image-studio/qwen-image-mps-patched.py \
uvicorn qwen-image-studio.server:app --host 0.0.0.0 --port 8000
```

You can also check the console log to see the exact CLI commands executed for each job.

> **Note:** Ensure `HSA_OVERRIDE_GFX_VERSION=11.5.1` is set for Strix Halo (gfx1151) GPU support.

### 6.3. Paths & Persistence

All generated images and job metadata are stored under `~/.qwen-image-studio/` in your HOME (outside the toolbox), so they persist outside the toolbox.

### 6.4. Attention Backend & Speed (Qwen)

* **Default:** **PyTorch SDPA** (Scaled Dot-Product Attention) — **stable path**.
* **Optional speed-up:** enable **Triton FlashAttention** (\~2× faster) **before** running Qwen:

```bash
export QWEN_FA_SHIM=1
```

> ⚠️ **Stability note (gfx1151):** Triton kernels can still be **buggy** and **crash** more often. With SDPA (default) users should **not** see crashes related to attention.

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

## 8. ComfyUI

**Path:** `/opt/ComfyUI`

ComfyUI is a flexible node-based interface for building and running image and video generation workflows. In this toolbox it is pre-cloned and configured with an AMD GPU monitor plugin.

### 8.1. Setup (ComfyUI only)

Before running ComfyUI, download model weights to `~/comfy-models` in your home directory.

```bash
# Run this FIRST to create ~/comfy-models and config file to point ComfyUI there
/opt/set_extra_paths.sh 

# Fetch model weights to ~/comfy-models
/opt/get_qwen_image.sh   # fetches Qwen Image models
/opt/get_wan22.sh        # fetches Wan2.2 models
```

These scripts ensure model files are downloaded to `~/comfy-models/` where they survive toolbox refreshes.

### 8.2. Run

Start ComfyUI inside the toolbox:

```bash
start_comfy_ui
```

Alias details:

```bash
cd /opt/ComfyUI
python main.py --port 8000 --output-directory "$HOME/comfy-outputs" --disable-mmap
```

> You will see an error message for missing `torchaudio`: this is **temporarily** removed as its presence causes ComfyUI to crash on boot.

* Outputs appear under `~/comfy-outputs/` in your HOME.
* Default ComfyUI port is 8188, but using `--port 8000` aligns it with Qwen Image Studio.
* Remote over SSH:

```bash
ssh -L 8000:localhost:8000 user@your-strix-box
```

Open [http://localhost:8000](http://localhost:8000) locally to access the web interface.

Upstream project: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

### 8.3. Running Image/Video Workflows in ComfyUI

You can load ready-made workflow files directly into ComfyUI:

* Qwen Image example: [https://comfyanonymous.github.io/ComfyUI\_examples/qwen\_image/](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/)
* Wan2.2 example: [https://comfyanonymous.github.io/ComfyUI\_examples/wan22/](https://comfyanonymous.github.io/ComfyUI_examples/wan22/)

---

## 9. Stability & Technical Notes

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

## 10. Credits & Links

* Qwen Image (original CLI): [https://github.com/ivanfioravanti/qwen-image-mps](https://github.com/ivanfioravanti/qwen-image-mps)
* ComfyUI: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* WAN 2.2: [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
* ROCm FlashAttention (AMD fork): [https://github.com/ROCm/flash-attention](https://github.com/ROCm/flash-attention)
* Distrobox: [https://distrobox.it/](https://distrobox.it/)

---

**Notes on persistence:** All model weights and outputs are stored in your **HOME** outside the distrobox (e.g., `~/.cache/huggingface/hub/`, `~/.qwen-image-studio/`, `~/Wan2.2-*`, `~/comfy-models`, `~/comfy-outputs`). This ensures they survive distrobox refreshes.
