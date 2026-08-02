# Shared config values for HOMIE-Wan.
#
# Ported from homie_npu config `config/r2v/neurips_configs/homie_formal_100.jsonl`
# and Phantom `phantom_wan/configs/shared_config.py`. Values that control numeric
# behaviour (dtypes, timesteps, negative prompt) are kept identical to the NPU run.
import torch
from easydict import EasyDict

homie_shared_cfg = EasyDict()

# ----- dtypes (match homie_npu: everything runs in bf16, per-op fp32 upcasts live
#       inside the modules just like the NPU code and Phantom) -----
homie_shared_cfg.param_dtype = torch.bfloat16
homie_shared_cfg.t5_dtype = torch.bfloat16
homie_shared_cfg.vae_dtype = torch.bfloat16

# ----- text encoder -----
homie_shared_cfg.text_len = 512  # predictor.text_len in the jsonl config

# ----- diffusion (UniPCMultistepScheduler, flow matching) -----
# From the `diffusion` block of homie_formal_100.jsonl.
homie_shared_cfg.num_train_timesteps = 1000
homie_shared_cfg.sample_fps = 24
homie_shared_cfg.sample_steps = 50
homie_shared_cfg.sample_shift = 3.0        # flow_shift
homie_shared_cfg.sample_guide_scale = 5.0  # guidance_scale

# Negative prompt used by the NPU WanPipeline (NEGATIVE_PROMOPT in wan_pipeline.py).
homie_shared_cfg.sample_neg_prompt = (
    "Bright tones, text, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)
