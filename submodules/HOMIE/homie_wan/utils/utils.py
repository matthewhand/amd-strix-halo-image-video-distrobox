# Utility helpers for HOMIE-Wan inference.
#
#  * cache_video / str2bool: reused from Phantom phantom_wan/utils/utils.py.
#  * dataset loaders + qwen-feature helpers: ported from the NPU entry script
#    inference_sora_camera_ocr_padding_phantom_pretrain_with_qwen_reorg.py so the
#    same nips_new_100.jsonl meta files work unchanged.
import argparse
import binascii
import json
import os
import os.path as osp

import imageio
import torch
import torchvision
from PIL import Image

__all__ = [
    "cache_video", "str2bool", "load_homie_jsonl",
    "map_strings_to_letters", "truncate_qwen_feature", "load_image",
]


def rand_name(length=8, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        name += suffix
    return name


def cache_video(tensor, save_file=None, fps=30, suffix=".mp4", nrow=8,
                normalize=True, value_range=(-1, 1), retry=5):
    """tensor: [B, C, F, H, W]. Writes an mp4 and returns the path."""
    cache_file = osp.join("/tmp", rand_name(suffix=suffix)) if save_file is None else save_file
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize,
                                            value_range=value_range)
                for u in tensor.unbind(2)
            ], dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:  # noqa: BLE001
            error = e
            continue
    print(f"cache_video failed, error: {error}", flush=True)
    return None


def str2bool(v):
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif v_lower in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (True/False)")


# --------------------------------------------------------------------------------------
# Dataset helpers (ported from the NPU entry script)
# --------------------------------------------------------------------------------------
def load_image(path):
    return Image.open(path).convert("RGB")


def map_strings_to_letters(input_list):
    """[[s1, s2], [s3], ...] -> [[a], [a], [b], ...]: one letter per outer sublist."""
    result = []
    for idx, sublist in enumerate(input_list):
        current_char = chr(ord("a") + idx)
        for _ in sublist:
            result.append([current_char])
    return result


def truncate_qwen_feature(qwen_feature, max_seq_len=2048):
    """Cap the qwen sequence to max_seq_len and floor its length to a multiple of 8
    (matches the NPU truncate_qwen_feature)."""
    if qwen_feature is None:
        return None
    seqlen = qwen_feature.shape[1]
    seqlen_cp8 = seqlen // 8 * 8
    if seqlen > max_seq_len:
        qwen_feature = qwen_feature[:, :max_seq_len, :]
    qwen_feature = qwen_feature[:, :seqlen_cp8, :]
    return qwen_feature


def _img_field_multis(item):
    """Flatten reference_paths ([[p1], [p2], ...]) into a flat list of PIL images."""
    reference_paths = []
    for reference_path_per_obj in item["reference_paths"]:
        reference_paths.extend(reference_path_per_obj)
    return [load_image(ref_path) for ref_path in reference_paths]


def load_homie_jsonl(file):
    """Load a HOMIE meta file (jsonl/json). Returns lists aligned by sample:
        prompts:              List[str]
        images:               List[List[PIL.Image]]  (reference images per sample)
        qwen_features:        List[Optional[torch.Tensor]]  (truncated, [1, S, 2048])
        reference_id_labels:  List[List[List[str]]]  (e.g. [[['a'],['b'],['c']], ...])
    """
    if file.endswith(".jsonl"):
        with open(file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
    elif file.endswith(".json"):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported meta file format: {file}")

    prompts = [
        item["cap"] + ". There is no text on the screen." if "cap" in item else item["prompt"]
        for item in data
    ]
    images = [_img_field_multis(item) for item in data]

    qwen_features = []
    for item in data:
        qwen_file = item.get("mllm_feature", None)
        if qwen_file:
            qwen = torch.load(qwen_file, map_location="cpu")
            qwen_features.append(truncate_qwen_feature(qwen))
        else:
            qwen_features.append(None)

    reference_id_labels = [map_strings_to_letters(item["reference_paths"]) for item in data]
    return prompts, images, qwen_features, reference_id_labels
