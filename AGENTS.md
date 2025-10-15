# Agent Playbook

## Environment Fast Facts
- Repo root: `/home/matthewh/amd-strix-halo-image-video-toolboxes`
- Primary toolbox container: `distrobox enter strix-halo-image-video`
- Container image already includes patched launchers for Qwen Image and WAN 2.2 under `/opt`.
- HIP/ROCm GPU access is required for actual inference jobs; expect failures if `/dev/kfd` or `/dev/dri` aren’t passed through.

## Running Commands Inside The Distrobox
- **Quick one-off:** `distrobox enter strix-halo-image-video -- <command>`
  - Example: `distrobox enter strix-halo-image-video -- pwd`
- **Interactive shell:** `distrobox enter strix-halo-image-video`
- Always assume commands must be wrapped this way unless you are already inside the container. Direct `/opt/...` paths do not exist on the host.

## Key Launchers & Aliases (available once inside the toolbox)
- `start_qwen_studio` → launches the patched Qwen Image Studio web UI on port 8000.
- `start_wan_cli` → runs the patched WAN CLI wrapper (`/opt/start_wan_cli_patched.py`).
- `start_comfy_ui` → starts ComfyUI on port 8000 with ROCm-friendly options.
- `/opt/run_wan_funny_showcase.sh` → demo script that renders three Lightning T2V clips (requires models downloaded beforehand).

## Model Paths & Downloads
- Qwen Image models cache under `~/.qwen-image-studio` and Hugging Face cache (`~/.cache/huggingface`).
- WAN checkpoints are expected in `~/Wan2.2-*`; Lightning adapters in `~/Wan2.2-Lightning/...`.
- ComfyUI helper scripts: `/opt/set_extra_paths.sh`, `/opt/get_qwen_image.sh`, `/opt/get_wan22.sh`.

## Troubleshooting Tips
- If you see `No HIP GPUs are available`, the container lacks GPU device access—check distrobox creation flags or run on the actual Strix Halo host.
- Patched launchers automatically strip `offload_state_dict`; if models crash complaining about that argument, verify you ran through the wrappers.
- `rg` is not installed inside the distrobox by default; use Python or `grep` for searches when working inside.

## Coordination Notes
- Respect the existing patched scripts; avoid reintroducing upstream launchers without the ROCm shims.
- Document any new helper scripts in `README.md` and announce aliases via `scripts/99-toolbox-banner.sh`.
- When in doubt, confirm with `distrobox enter strix-halo-image-video -- pwd` to ensure you are inside the expected environment before running GPU workloads.
