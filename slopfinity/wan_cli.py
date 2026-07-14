"""WAN generate.py CLI mapping — fleet model ids → real launcher argv.

The upstream generate.py accepts --task/--ckpt_dir/--save_file/--image, not
the old workers fiction of --out/--model.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple


def _workspace() -> str:
    return os.environ.get("SLOPFINITY_WORKSPACE") or os.getcwd()


def _is_complete_ckpt(path: str, task: str) -> bool:
    """True if path looks like a usable generate.py checkpoint tree."""
    if not path or not os.path.isdir(path):
        return False
    t5 = os.path.join(path, "models_t5_umt5-xxl-enc-bf16.pth")
    if not os.path.isfile(t5) or os.path.getsize(t5) < 8_000_000_000:
        # Official native T5 is ~11 GB; Comfy fp8 (~6.7 GB) has wrong key layout.
        return False
    if task == "ti2v-5B":
        vae = os.path.join(path, "Wan2.2_VAE.pth")
        # shards or single safetensors at root
        has_dit = (
            os.path.isfile(os.path.join(path, "diffusion_pytorch_model.safetensors"))
            or os.path.isfile(
                os.path.join(path, "diffusion_pytorch_model-00001-of-00003.safetensors")
            )
            or os.path.isfile(os.path.join(path, "config.json"))
        )
        return os.path.isfile(vae) and has_dit
    # A14B MoE: high/low noise subfolders + VAE
    vae = os.path.join(path, "Wan2.1_VAE.pth")
    high = os.path.join(path, "high_noise_model")
    low = os.path.join(path, "low_noise_model")
    return (
        os.path.isfile(vae)
        and os.path.isdir(high)
        and os.path.isdir(low)
        and (
            os.path.isfile(os.path.join(high, "config.json"))
            or any(
                n.startswith("diffusion_pytorch_model")
                for n in os.listdir(high)
                if os.path.isfile(os.path.join(high, n))
            )
        )
    )


def _candidate_trees() -> List[Tuple[str, str, str]]:
    """Ordered (ckpt_path, task, size) candidates. Prefer smaller complete packs."""
    host_home = os.path.expanduser("~")
    comfy = "/mnt/downloads/comfy-models"
    return [
        # TI2V-5B: ~34 GB, supports text+image; best default when present.
        (f"{comfy}/Wan2.2-TI2V-5B", "ti2v-5B", "704*1280"),
        (f"{host_home}/Wan2.2-TI2V-5B", "ti2v-5B", "704*1280"),
        # Official T2V A14B (~126 GB) when fully materialized.
        (f"{comfy}/Wan2.2-T2V-A14B-official", "t2v-A14B", "480*832"),
        (f"{comfy}/Wan2.2-T2V-A14B", "t2v-A14B", "480*832"),
        (f"{host_home}/Wan2.2-T2V-A14B", "t2v-A14B", "480*832"),
        # I2V A14B when fully present.
        (f"{comfy}/Wan2.2-I2V-A14B", "i2v-A14B", "480*832"),
        (f"{host_home}/Wan2.2-I2V-A14B", "i2v-A14B", "480*832"),
    ]


def resolve_wan_ckpt(prefer_task: Optional[str] = None) -> Dict[str, str]:
    """Pick the best complete checkpoint tree on this host."""
    for ckpt, task, size in _candidate_trees():
        if prefer_task and task != prefer_task:
            continue
        if _is_complete_ckpt(ckpt, task):
            return {"ckpt": ckpt, "task": task, "size": size}
    # Fallback: first existing dir (may fail at load — better error than empty I2V)
    for ckpt, task, size in _candidate_trees():
        if os.path.isdir(ckpt):
            return {"ckpt": ckpt, "task": task, "size": size}
    # Last resort placeholder
    return {
        "ckpt": os.path.expanduser("~/Wan2.2-I2V-A14B"),
        "task": "i2v-A14B",
        "size": "480*832",
    }


def wan_paths(model: str = "wan2.2") -> Dict[str, str]:
    """Resolve task + checkpoint dirs for WAN generate.py.

    Prefers a complete pack:
      1. Wan2.2-TI2V-5B (smallest official generate.py tree)
      2. Wan2.2-T2V-A14B official
      3. Wan2.2-I2V-A14B

    Override with WAN_CKPT_DIR / WAN_TASK / WAN_LORA_DIR / WAN_SIZE / WAN_FRAME_NUM.
    Rejects the broken "official" tree that only has a Comfy-fp8 T5 misnamed as
    models_t5_umt5-xxl-enc-bf16.pth (wrong state_dict keys for wan-video-studio).
    """
    resolved = resolve_wan_ckpt()
    # wan2.5 is a UI/label alias of the same checkpoint resolver as wan2.2 —
    # there is no separate open-weight Wan 2.5 pack on this host. STAGE_BUDGETS
    # rows (84/96) are historical peaks only; config stage_budget_overrides → 0.
    defaults = {
        "wan2.2": {
            "task": resolved["task"],
            "ckpt": resolved["ckpt"],
            "lora": os.environ.get("WAN_LORA_DIR") or "",
            "size": resolved["size"],
            "frame_num": os.environ.get("WAN_FRAME_NUM") or (
                "17" if resolved["task"] != "ti2v-5B" else "41"
            ),
        },
        "wan2.5": {
            "task": resolved["task"],
            "ckpt": resolved["ckpt"],
            "lora": os.environ.get("WAN_LORA_DIR") or "",
            "size": resolved["size"],
            "frame_num": os.environ.get("WAN_FRAME_NUM") or (
                "17" if resolved["task"] != "ti2v-5B" else "41"
            ),
        },
    }
    cfg = dict(defaults.get(model) or defaults["wan2.2"])
    if os.environ.get("WAN_CKPT_DIR"):
        cfg["ckpt"] = os.environ["WAN_CKPT_DIR"]
    if os.environ.get("WAN_TASK"):
        cfg["task"] = os.environ["WAN_TASK"]
    if os.environ.get("WAN_SIZE"):
        cfg["size"] = os.environ["WAN_SIZE"]
    if os.environ.get("WAN_FRAME_NUM"):
        cfg["frame_num"] = os.environ["WAN_FRAME_NUM"]
    if os.environ.get("WAN_LORA_DIR"):
        cfg["lora"] = os.environ["WAN_LORA_DIR"]
    return cfg


def wan_launcher_argv(prompt: str, in_img: str, out: str, model: str = "wan2.2") -> List[str]:
    cfg = wan_paths(model)
    ckpt = cfg["ckpt"]
    lora = cfg["lora"]
    c_ckpt = f"/models/{os.path.basename(ckpt.rstrip('/'))}"
    c_lora = f"/models/lightning/{os.path.basename(lora.rstrip('/'))}"
    ws = _workspace()
    c_out = out
    c_img = in_img
    if out.startswith(ws):
        c_out = "/workspace" + out[len(ws) :]
    if in_img and in_img.startswith(ws):
        c_img = "/workspace" + in_img[len(ws) :]
    cmd = [
        "python3", "/opt/wan_launcher.py",
        "--task", cfg["task"],
        "--size", cfg["size"],
        "--frame_num", str(cfg["frame_num"]),
        "--ckpt_dir", c_ckpt,
        "--save_file", c_out,
        "--prompt", prompt,
        "--offload_model", "True",
    ]
    # ti2v/i2v/s2v accept --image; t2v does not need it but generate.py tolerates?
    # Only pass when we have a path and task is not pure t2v.
    if c_img and cfg["task"] != "t2v-A14B":
        cmd += ["--image", c_img]
    if lora and os.path.isdir(lora):
        cmd += ["--lora_dir", c_lora]
    return cmd
