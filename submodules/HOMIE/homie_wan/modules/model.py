# HomieWanModel — CUDA/PyTorch port of the NPU WanDiT (homie_npu wan_dit.py).
#
# All Megatron / torch-npu / pipeline-parallel machinery is stripped. What remains is
# the exact inference forward pass, with submodule names chosen so that the MindSpeed
# checkpoint (HOMIE-Wan-Model/Homie_Wan_14B*.safetensors) loads with strict=True.
#
# HOMIE-specific pieces preserved from the NPU model:
#   * reference latents concatenated along the temporal (frame) axis (r2v),
#   * qwen (MLLM) features projected by `qwen_3dvae_connector` and concatenated after
#     the video+reference tokens,
#   * `task_embedding` (noise/ref/mllm) + `ref_embedding` (per-reference id) added to
#     the token stream,
#   * CrossModalityAffine modulation of the video Q/K inside self-attention.
#
# Precision: matches Phantom's fp32 discipline (time embedding, modulation adds and
# all LayerNorm/RMSNorm compute run in fp32) while the bulk of the network runs in the
# param dtype (bf16). See the `amp.autocast(dtype=torch.float32)` blocks and `.float()`
# calls, which mirror the NPU `.to(torch.float32)` sites.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from einops import rearrange, repeat

from .attention import HomieSelfAttention, HomieCrossAttention
from .qwen_connector import MLPFeatureAligner
from .rope import RoPE3DWan

__all__ = ["HomieWanModel"]


# --------------------------------------------------------------------------------------
# Helper functions ported verbatim from wan_dit.py
# --------------------------------------------------------------------------------------
def first_occurrence_rank(s):
    char_first_rank = {}
    result = []
    for char in s:
        if char not in char_first_rank:
            char_first_rank[char] = len(char_first_rank)
        result.append(char_first_rank[char])
    return result


def list_to_string(lst):
    return "".join([sublist[0] for sublist in lst])


def sinusoidal_embedding_1d(dim, position, theta=10000):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            theta,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    embs = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return embs.to(position.dtype)


# --------------------------------------------------------------------------------------
# Text projection (checkpoint keys: text_embedding.linear_1 / linear_2)
# --------------------------------------------------------------------------------------
class TextProjection(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_size, bias=True)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# --------------------------------------------------------------------------------------
# DiT block (checkpoint keys: blocks.N.*)
# --------------------------------------------------------------------------------------
class HomieWanBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=True,
                 eps=1e-6, rope=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.self_attn = HomieSelfAttention(dim, num_heads, qk_norm=qk_norm, eps=eps, rope=rope)
        self.cross_attn = HomieCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)

        # norm1/norm2: no affine; norm3: affine (matches NPU + checkpoint norm3.weight/bias).
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True) if cross_attn_norm \
            else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, latents, context, time_emb, rope_freqs, video_ref_mllm_tokens):
        # Dtype discipline mirrors the NPU WanDiTBlock (_before/_after_self_attention)
        # LITERALLY, since the NPU model is the numerical ground truth (TASK 1.4):
        #   * modulation params are cast to time_emb's dtype before the add,
        #   * norm1 runs in fp32 (latents.float()), then result cast back,
        #   * norm2 / norm3 run in the latents dtype (NPU does NOT upcast them),
        #   * gated residual adds run in the latents dtype (no fp32 autocast).
        dtype = time_emb.dtype
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=dtype, device=time_emb.device) + time_emb
        ).chunk(6, dim=1)

        # self-attention (norm1 in fp32)
        self_attn_input = self.modulate(
            self.norm1(latents.float()), shift_msa, scale_msa
        ).to(latents.dtype)
        self_attn_out = self.self_attn(self_attn_input, rope_freqs, video_ref_mllm_tokens)
        latents = latents + gate_msa * self_attn_out

        # text cross-attention (norm3 in latents dtype)
        crs_attn_out = self.cross_attn(self.norm3(latents), context)
        latents = latents + crs_attn_out

        # ffn (norm2 in latents dtype)
        modu_out = self.modulate(self.norm2(latents), shift_mlp, scale_mlp)
        latents = latents + gate_mlp * self.ffn(modu_out)
        return latents


# --------------------------------------------------------------------------------------
# Output head (checkpoint keys: head.head / head.modulation)
# --------------------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, latents, times):
        # times: [B, H] -> [B, 1, H] so it broadcasts against modulation [1, 2, H].
        with amp.autocast(dtype=torch.float32):
            shift, scale = (self.modulation.to(times.device) + times.unsqueeze(1)).chunk(2, dim=1)
        out = self.head(self.norm(latents.float()) * (1 + scale) + shift)
        return out


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------
class HomieWanModel(nn.Module):
    def __init__(
        self,
        model_type="r2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        img_dim=1280,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        qk_norm=True,
        qk_norm_type="rmsnorm",
        cross_attn_norm=True,
        eps=1e-6,
        max_seq_len=1024,
        reference_num=5,
        qwen_hidden_size=2048,
        **kwargs,
    ):
        super().__init__()
        assert model_type == "r2v", "This port only supports the HOMIE r2v task."
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.eps = eps
        self.max_seq_len = max_seq_len
        self.reference_num = reference_num
        self.head_dim = dim // num_heads

        self.rope = RoPE3DWan(head_dim=self.head_dim, max_seq_len=max_seq_len)

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = TextProjection(text_dim, dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # qwen (MLLM) connector — hidden_dims=[2048, 5120] matches the checkpoint mlp.*
        self.qwen_3dvae_connector = MLPFeatureAligner(
            source_dim=qwen_hidden_size,
            target_dim=dim,
            hidden_dims=[2048, 5120],
            use_residual=True,
            use_adaptive_scale=True,
        )

        # task / reference embeddings
        self.task_embedding = nn.Embedding(3, dim)  # 0: noise, 1: reference, 2: mllm
        self.ref_embedding = nn.Embedding(reference_num, dim)

        # blocks
        self.blocks = nn.ModuleList([
            HomieWanBlock(dim, ffn_dim, num_heads, qk_norm=qk_norm,
                          cross_attn_norm=cross_attn_norm, eps=eps, rope=self.rope)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    # ---- patch helpers (verbatim math from wan_dit.py) ----
    def patchify(self, embs):
        grid_sizes = embs.shape[2:]
        patch_out = rearrange(embs, "b c f h w -> b (f h w) c").contiguous()
        return patch_out, grid_sizes

    def unpatchify(self, embs, frames, height, width):
        return rearrange(
            embs,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=frames, h=height, w=width,
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )

    def get_reference_embedding(self, embs, lst, max_len):
        # Stage-2 behaviour: use the original ids (no random re-assignment).
        indices = torch.tensor(lst, dtype=torch.long, device=embs.weight.device)
        embed_tensor = embs(indices)
        current_len, embed_dim = embed_tensor.shape
        pad_len = max_len - current_len
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, embed_dim, dtype=embed_tensor.dtype,
                                     device=embed_tensor.device)
            result = torch.cat([embed_tensor, pad_tensor], dim=0)
        else:
            result = embed_tensor
        return result

    def forward(self, x, timestep, prompt, reference, qwen_feature=None,
                reference_id_labels=None):
        """
        x:          video/noise latent [B, C, F, H, W]
        timestep:   [B]
        prompt:     raw text-encoder embeddings [B, text_len, text_dim]
        reference:  reference latents [B, C, reference_num, H, W]
        qwen_feature: [B, S_qwen, qwen_hidden_size] or None
        reference_id_labels: e.g. [['a'], ['b'], ['c']]
        Returns the predicted video-latent noise [B, C, F, H, W] (reference frames dropped).
        """
        device = x.device
        timestep = timestep.to(device)

        # ----- time embeddings (fp32) -----
        with amp.autocast(dtype=torch.float32):
            times = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).float())
            time_emb = self.time_projection(times).unflatten(1, (6, self.dim))

        # ----- prompt embeddings -----
        bs = prompt.size(0)
        prompt = prompt.view(bs, -1, prompt.size(-1))
        prompt_emb = self.text_embedding(prompt)

        # ----- r2v: concat reference along the frame axis -----
        reference = reference.to(x)
        x = torch.cat([x, reference], dim=2)  # [B, C, F+ref_num, H, W]

        # ----- patch embedding -----
        patch_emb = self.patch_embedding(x.to(self.patch_embedding.weight.dtype))
        embs, grid_sizes = self.patchify(patch_emb)
        batch_size, frames, height, width = embs.shape[0], grid_sizes[0], grid_sizes[1], grid_sizes[2]

        # ----- qwen (MLLM) feature injection -----
        has_qwen = isinstance(qwen_feature, torch.Tensor) and qwen_feature.dim() > 1 \
            and qwen_feature.shape[1] > 1
        if has_qwen:
            qwen_feature = qwen_feature.to(embs)
            qwen_proj = self.qwen_3dvae_connector(qwen_feature)
            embs = torch.cat([embs, qwen_proj], dim=1)
            qwen_seqlen = qwen_proj.shape[1]
        else:
            qwen_seqlen = 0

        # ----- task / reference embeddings -----
        tokens_per_frame = height * width
        task_emb_noise = repeat(self.task_embedding.weight[0], "c -> n l c",
                                n=batch_size, l=tokens_per_frame * (frames - self.reference_num))
        task_emb_ref = repeat(self.task_embedding.weight[1], "c -> n l c",
                              n=batch_size, l=tokens_per_frame * self.reference_num)

        reference_id_labels_refined = [label[0] for label in reference_id_labels]
        reference_class = first_occurrence_rank(
            list_to_string(reference_id_labels_refined)
        )[:self.reference_num]
        id_embedding = self.get_reference_embedding(self.ref_embedding, reference_class,
                                                    self.reference_num)
        id_embedding = torch.repeat_interleave(id_embedding, tokens_per_frame, dim=0)
        id_embedding = repeat(id_embedding, "l c -> n l c", n=batch_size)
        reference_embedding = task_emb_ref + id_embedding

        if has_qwen:
            task_emb_mllm = repeat(self.task_embedding.weight[2], "c -> n l c",
                                   n=batch_size, l=qwen_seqlen)
            customized = torch.cat([task_emb_noise, reference_embedding, task_emb_mllm], dim=1)
        else:
            customized = torch.cat([task_emb_noise, reference_embedding], dim=1)
        embs = embs + customized.to(embs.dtype)

        # ----- rope (with qwen text positions) -----
        rope_freqs = self.rope(frames, height, width, seq_text=qwen_seqlen, device=device)

        # token counts for the modality-split self-attention
        video_tokens = (frames - self.reference_num) * tokens_per_frame
        ref_tokens = self.reference_num * tokens_per_frame
        video_ref_mllm_tokens = (video_tokens, ref_tokens, qwen_seqlen)

        # ----- transformer blocks -----
        for block in self.blocks:
            embs = block(embs, prompt_emb, time_emb, rope_freqs, video_ref_mllm_tokens)

        # ----- drop qwen tokens, head, unpatchify -----
        if qwen_seqlen > 0:
            embs = embs[:, :-qwen_seqlen, :]
        embs_out = self.head(embs, times)
        out = self.unpatchify(embs_out, frames, height, width)

        # ----- drop reference frames -----
        out = out[:, :, :-self.reference_num]
        return out
