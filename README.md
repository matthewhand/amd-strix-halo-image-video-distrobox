```
███████╗████████╗██████╗ ██╗██╗  ██╗      ██╗  ██╗ █████╗ ██╗      ██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██║╚██╗██╔╝      ██║  ██║██╔══██╗██║     ██╔═══██╗
███████╗   ██║   ██████╔╝██║ ╚███╔╝       ███████║███████║██║     ██║   ██║
╚════██║   ██║   ██╔══██╗██║ ██╔██╗       ██╔══██║██╔══██║██║     ██║   ██║
███████║   ██║   ██║  ██║██║██╔╝ ██╗      ██║  ██║██║  ██║███████╗╚██████╔╝
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝ 

                         I M A G E   &   V I D E O                        

```

# AMD Strix Halo — Image & Video Toolbox

A Fedora **toolbox** image with a full **ROCm environment** for **image & video generation** on **AMD Ryzen AI Max “Strix Halo” (gfx1151)**. It includes support for **Qwen Image/Edit,** and **WAN 2.2** models. If you are looking for sandboxes to run LLMs with llama.cpp, see here: [https://github.com/kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)

> Tested on Framework Desktop (Strix Halo, 128 GB unified memory). Works on other Strix Halo systems (GMKtec EVO X-2, HP Z2 G1a, etc).

---

## Table of Contents

1. [Overview](#overview)
2. [What’s Included](#whats-included)
3. [Creating the Toolbox](#creating-the-toolbox)
4. [Unified Memory Setup](#unified-memory-setup)
5. [Qwen Image Studio](#qwen-image-studio)
6. [WAN 2.2](#wan-22)
7. [ComfyUI](#comfyui)
8. [Stability Notes](#stability-notes)
9. [Credits & Links](#credits--links)

---

## Overview & Included Components

This toolbox provides a ROCm nightly stack for Strix Halo (gfx1151), built from [ROCm/TheRock](https://github.com/ROCm/TheRock), plus three main tools. All model weights are stored **outside the toolbox** (in your HOME), so they survive container deletion or refresh.

| Component                                                                                          | Path                     | Purpose                                                |
| -------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------ |
| **Qwen Image Studio** ([fork of qwen-image-mps](https://github.com/ivanfioravanti/qwen-image-mps)) | `/opt/qwen-image-studio` | Web UI + job manager with retries, CLI still available |
| **WAN 2.2** ([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2))                              | `/opt/wan-video-studio`  | CLI for text-to-video / image-to-video                 |
| **ComfyUI** ([ComfyUI](https://github.com/comfyanonymous/ComfyUI))                                 | `/opt/ComfyUI`           | Node-based UI, AMD GPU monitor plugin                  |

> **Note:** Scripts in `/opt` (`set_extra_paths.sh`, `get_qwen_image.sh`, `get_wan22.sh`) are **for ComfyUI only**. Skip them unless you use ComfyUI.

---

## Creating the Toolbox

A toolbox is a containerized user environment that shares your home directory and user account. To use this toolbox, you need to **expose GPU devices** and add your user to the right groups. This ensures ROCm and Vulkan have access to Strix Halo’s GPU nodes.

Run:

```bash
toolbox create image-video \
  --image docker.io/kyuz0/amd-strix-halo-image-video:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined
```

Explanation:

* `--device /dev/dri` → graphics & video devices
* `--device /dev/kfd` → required for ROCm compute
* `--group-add video, render` → ensures user has GPU access
* `--security-opt seccomp=unconfined` → avoids syscall sandbox issues with GPUs

Enter the toolbox:

```bash
toolbox enter image-video
```

Inside, your prompt looks normal but you’re in the container with:

* Full ROCm stack
* All tools under `/opt`
* Shared `$HOME` (so models and outputs are persistent).

### Updating the toolbox
This toolbox will be updated regularly with new nightly builds from TheRock for ROCm 7 and updated support for image and video generation.

You can use `refresh_toolbox.sh` to pull updates:

```bash
chmod +x refresh_toolbox.sh
./refresh_toolbox.sh
```

> [!WARNING] ⚠️ **Refreshing deletes the current toolbox**
> Running `refresh_toolbox.sh` **removes and recreates** the toolbox image/container. This should be safe if you followed this README as all model files and outputs are saved OUTSIDE the toolbox in your home directory
>
> ❌ **Lost (deleted)** — anything stored **inside the container**, e.g. `/opt/...` or other non-HOME paths.

---

## Unified Memory Setup

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

On Fedora 42 you can set these in `/etc/default/grub` under `GRUB_CMDLINE_LINUX`, then run:

```bash
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

---

## Qwen Image Studio

**Path:** `/opt/qwen-image-studio`
**Run:** `start_qwen_studio` (serves at [http://localhost:8000](http://localhost:8000))

### Download Models

Before starting the UI, you need to fetch the model weights. Without them the system cannot generate images. This step only has to be done once, as the models are stored in your HOME outside the toolbox.

Run to see a list of Qwen Image models:

```bash
python /opt/qwen-image-studio/qwen-image-mps.py download
```

Or to fetch all variants in one go (careful, together these exceed 80GB):

```bash
cd /opt/qwen-image-studio/
python /opt/qwen-image-studio/qwen-image-mps.py download all
```

* Models go to `~/.cache/huggingface/hub/` (outside toolbox)
* Available: `qwen-image`, `qwen-image-edit`, `lightning-lora-8`, `lightning-lora-4`
* LoRA adapters require the base models first

Outputs and job state are kept in `~/.qwen-image-studio/`, which lives in your HOME outside the toolbox so they persist across updates or rebuilds.

### How to Start

Start the Web UI with the alias prepared in the container:

```bash
start_qwen_studio
```

This will launch a FastAPI/uvicorn server on port 8000. If you are on the Strix Halo machine locally, open [http://localhost:8000](http://localhost:8000) in a browser. If you connect over SSH, forward the port first:

```bash
ssh -L 8000:localhost:8000 user@your-strix-box
```

Then visit the same URL locally. The UI provides job management, retry logic, and shows GPU stats.

Under the hood, `start_qwen_studio` simply runs:

```bash
cd /opt/qwen-image-studio && \
uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000
```

All generated images and job metadata are stored under `~/.qwen-image-studio/` in your HOME (outside the toolbox), so they persist outide the toolbox.&#x20;

You can also look at the console log to see the exact CLI commands executed for each job.

---

## WAN 2.2

**Path:** `/opt/wan-video-studio` (CLI only, Web UI planned)

WAN 2.2 is Alibaba’s open‑sourced text‑to‑video and image‑to‑video model. In this toolbox you run it through the CLI script provided in their repository. A Web UI (Wan Video Studio) is planned but not yet included.

### Download Models

Before running, you need to fetch model weights from Hugging Face. These are large, so store them in your HOME outside the toolbox so they survive rebuilds. For example, to fetch the smaller 5B model:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ~/Wan2.2-TI2V-5B
```

Other checkpoints (14B etc.) are also available, see the upstream repo.

### Video Generation Example

Run generation inside the toolbox by pointing to the local model path:

```bash
cd /opt/wan-video-studio
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ~/Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype \
  --prompt "Two cats boxing under a spotlight" \
  --frame_num 41   \
  --save_file ~/video.mp4
```

Explanation:

* `--task ti2v-5B` selects the checkpoint variant
* `--size` sets resolution (multiples of 16, high resolutions need lots of memory)
* `--ckpt_dir` points to the model folder you downloaded
* `--offload_model True` and `--convert_model_dtype` help fit models in memory on Strix Halo
* `--prompt` is your text prompt
* `--frame_num` number of frames to generate, needs to be 4n+1
* `--save_file` location where to save the video (ensure this is OUTSIDe the toolbox!)


### Notes

* Always download models under HOME so they persist.
* Outputs are written in the current directory unless you override.
* See official docs for more details: [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)

---

## ComfyUI

**Path:** `/opt/ComfyUI`

ComfyUI is a flexible node‑based interface for building and running image and video generation workflows. In this toolbox it is pre‑cloned and configured with an AMD GPU monitor plugin.

### Setup (ComfyUI only)

Before running ComfyUI, you need to download the model weights to `~/comfy-models` in your home directory.

```bash
# Run this FIRST to create ~/comfy-models and config file to point ComfyUI there
/opt/set_extra_paths.sh 

# Fetch model weights to ~/comfy-models
/opt/get_qwen_image.sh   # fetches Qwen Image models
/opt/get_wan22.sh        # fetches Wan2.2 models
```

These scripts ensure model files are downloaded to `~/comfy-models/` where they survive toolbox refreshes.

### Run

Start ComfyUI inside the toolbox:

```bash
start_comfy_ui
```

This is an alias for:

```bash
cd /opt/ComfyUI
python main.py --port 8000 --output-directory "$HOME/comfy-outputs" --disable-mmap
```

> You will see an error message for missing `torchaudio`: this is temporarely removed as its presence causes ComfyUI to crash on boot.

* Outputs will appear under `~/comfy-outputs/` in your HOME.
* By default ComfyUI listens on port 8188, but using `--port 8000` aligns it with Qwen Image Studio. That way you can forward a single port if you are using SSH and choose which UI to run.
* For remote access over SSH:

```bash
ssh -L 8000:localhost:8000 user@your-strix-box
```

Open [http://localhost:8000](http://localhost:8000) locally to access the ComfyUI web interface.

Upstream project: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

### Running Image/Video Workflows in ComfyUI

You can load ready‑made workflow files directly into the ComfyUI interface. These demonstrate how to connect nodes for Qwen image generation and Wan2.2 video generation.

* Qwen Image example: [https://comfyanonymous.github.io/ComfyUI\_examples/qwen\_image/](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/)
* Wan2.2 example: [https://comfyanonymous.github.io/ComfyUI\_examples/wan22/](https://comfyanonymous.github.io/ComfyUI_examples/wan22/)

## Stability Notes

ROCm 7 on Strix Halo may crash with:

```
Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege.
```

This is a known instability tracked here: [https://gitlab.freedesktop.org/drm/amd/-/issues/4321#note\_3048205](https://gitlab.freedesktop.org/drm/amd/-/issues/4321#note_3048205). It occurs randomly and is a memory management bug in the current ROCm 7.0.0rc release, not necessarily tied to workload size. When it happens the process usually crashes, but the system itself remains stable and only the process needs restarting.

* A fix is expected in ROCm 7.0.x.
* **Qwen Image Studio** is generally more stable than ComfyUI, likely due to simpler execution paths.
* Qwen Image Studio also includes automatic retry logic: each job is attempted up to 3 times, so transient GPU faults often recover without user intervention.
* ComfyUI can occasionally crash under the same conditions; if this happens, re‑launching it is usually enough.

Until the ROCm fix lands, expect occasional instability when running large models or long video sequences. Keeping jobs queued in Qwen Image Studio is a safer option for overnight or unattended runs.

---

## Credits & Links

* Qwen Image (original CLI): [https://github.com/ivanfioravanti/qwen-image-mps](https://github.com/ivanfioravanti/qwen-image-mps)
* ComfyUI: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* WAN 2.2: [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
* ROCm FlashAttention (AMD fork): [https://github.com/ROCm/flash-attention](https://github.com/ROCm/flash-attention)
* Toolbox (Fedora): [https://containertoolbx.org/](https://containertoolbx.org/)
