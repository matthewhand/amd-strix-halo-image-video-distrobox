"""Env-driven configuration for the pipelines package.

All knobs read from the environment with sensible defaults, so callers can
override per-host (e.g. SERVER, OUTPUT_DIR) without code changes.
"""
import os

SERVER = os.environ.get("COMFY_SERVER", "127.0.0.1:8188")
IMAGE = os.environ.get("COMFY_IMAGE", "amd-strix-halo-image-video-toolbox:latest")
CONTAINER = os.environ.get("COMFY_CONTAINER", "comfyui-ltx23")
OUTPUT_DIR = os.environ.get("COMFY_OUTPUTS", "/tmp/comfy-outputs")
MODELS_DIR = os.path.expanduser(os.environ.get("COMFY_MODELS", "~/comfy-models"))
HF_CACHE = os.path.expanduser(os.environ.get("HF_CACHE", "~/.cache/huggingface"))

DOCKER_ENV = [
    "-e", "HSA_OVERRIDE_GFX_VERSION=11.5.1",
    "-e", "LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib",
    "-e", "LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib",
]
DOCKER_GPU = [
    "--device", "/dev/dri", "--device", "/dev/kfd",
    "--security-opt", "seccomp=unconfined",
]
