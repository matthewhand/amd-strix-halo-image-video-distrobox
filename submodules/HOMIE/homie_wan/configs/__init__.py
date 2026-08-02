# Config registry, mirroring Phantom `phantom_wan/configs/__init__.py`.
import copy

from .homie_s2v_14B import s2v_14B

# The generation input_size in the NPU config is [97, 720, 1280] (F, H, W); the
# default resolution here is therefore 1280*720 (width*height).
HOMIE_CONFIGS = {
    "s2v-14B": s2v_14B,
}

# width*height -> (width, height)
SIZE_CONFIGS = {
    "1280*720": (1280, 720),
    "720*1280": (720, 1280),
    "832*480": (832, 480),
    "480*832": (480, 832),
    "1024*1024": (1024, 1024),
}

MAX_AREA_CONFIGS = {
    "1280*720": 1280 * 720,
    "720*1280": 720 * 1280,
    "832*480": 832 * 480,
    "480*832": 480 * 832,
    "1024*1024": 1024 * 1024,
}

SUPPORTED_SIZES = {
    "s2v-14B": ("1280*720", "720*1280", "832*480", "480*832"),
}


def get_config(task):
    """Return a deep copy so per-run mutations (e.g. sample_fps) don't leak."""
    return copy.deepcopy(HOMIE_CONFIGS[task])
