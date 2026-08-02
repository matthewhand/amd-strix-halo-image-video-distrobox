# HOMIE-Wan S2V 14B config.
#
# The `predictor` block of homie_npu `homie_formal_100.jsonl` maps directly onto the
# HomieWanModel constructor arguments below. Names are kept identical to the NPU
# WanDiT so that the checkpoint keys line up 1:1.
from easydict import EasyDict

from .shared_config import homie_shared_cfg

s2v_14B = EasyDict(__name__="Config: HOMIE-Wan S2V 14B")
s2v_14B.update(homie_shared_cfg)

# ----- feature extraction models (loaded from the HF Wan2.1-T2V-14B-Diffusers dir,
#       exactly like the NPU `ae` / `text_encoder` / `tokenizer` config blocks) -----
s2v_14B.vae_subfolder = "vae"                 # AutoencoderKLWan
s2v_14B.text_encoder_subfolder = "text_encoder"  # UMT5EncoderModel
s2v_14B.tokenizer_subfolder = "tokenizer"     # AutoTokenizer
s2v_14B.vae_norm_mode = "channel_specified_shift_scale"  # ae.norm_mode

# ----- transformer / DiT (predictor block) -----
s2v_14B.model_type = "r2v"
s2v_14B.patch_size = (1, 2, 2)
s2v_14B.text_len = 512
s2v_14B.in_dim = 16
s2v_14B.dim = 5120            # hidden_size
s2v_14B.ffn_dim = 13824
s2v_14B.freq_dim = 256
s2v_14B.text_dim = 4096
s2v_14B.img_dim = 1280
s2v_14B.out_dim = 16
s2v_14B.num_heads = 40
s2v_14B.num_layers = 40
s2v_14B.qk_norm = True
s2v_14B.qk_norm_type = "rmsnorm"
s2v_14B.cross_attn_norm = True
s2v_14B.eps = 1e-6
s2v_14B.max_seq_len = 1024
s2v_14B.reference_num = 5     # predictor.reference_num / pipeline_config.reference_num

# qwen (MLLM) connector: source feature dim of the pre-extracted qwen features.
# homie_npu hard-codes qwen_hidden_size = 2048 in WanDiT.__init__.
s2v_14B.qwen_hidden_size = 2048
