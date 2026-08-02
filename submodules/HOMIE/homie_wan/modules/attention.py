# Attention modules for HOMIE-Wan.
#
# Ported from the NPU WanVideoParallelAttention / CrossModalityAffine
# (homie_npu wan_dit.py:1019-1359). The NPU code operates in seq-first "sbh" layout
# with Megatron Column/Row-parallel linears; here everything is plain `nn.Linear`
# in batch-first "[B, S, H]" layout, single-GPU.
#
# Submodule names are kept identical to the checkpoint:
#   proj_q, proj_k, proj_v, proj_out, q_norm, k_norm,
#   cross_modality_affine_query.{scale_proj,shift_proj},
#   cross_modality_affine_key.{scale_proj,shift_proj}
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RMSNorm", "CrossModalityAffine", "HomieSelfAttention", "HomieCrossAttention"]


class RMSNorm(nn.Module):
    """Equivalent to megatron.legacy.model.RMSNorm used by MindSpeed `normalize`.

    Computes in fp32 then casts back, weight applied after cast — identical to
    Phantom's WanRMSNorm. `dim` is the *full hidden size* (norm is applied before
    the head reshape, exactly as in the NPU code), so `weight` has shape [hidden].
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class CrossModalityAffine(nn.Module):
    """Verbatim from wan_dit.py:1340. Modulates `feat1` with information pooled from
    `feat2` (the MLLM/qwen features). Batch-first: pooling is over the sequence dim=1
    (the NPU pools over dim=0 in seq-first layout — same axis)."""

    def __init__(self, dim, scale_init=0.01, shift_init=0.0):
        super().__init__()
        self.dim = dim
        self.scale_proj = nn.Linear(dim, dim, bias=False)
        self.shift_proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_normal_(self.scale_proj.weight)
        self.scale_proj.weight.data *= scale_init
        nn.init.xavier_normal_(self.shift_proj.weight)
        self.shift_proj.weight.data *= shift_init

    def forward(self, feat1, feat2):
        # feat1: injected feature [B, S1, H]; feat2: information source [B, S2, H]
        feat2_pool = torch.mean(feat2, dim=1, keepdim=True)  # [B, 1, H]
        shift = self.shift_proj(feat2_pool).expand_as(feat1)
        scale = self.scale_proj(feat2_pool).expand_as(feat1)
        return feat1 * (1 + scale) + shift


def sdpa(q, k, v):
    """Scaled-dot-product attention. q/k/v: [B, S, N, D] -> out [B, S, N, D].

    Uses torch SDPA (flash/mem-efficient kernel on CUDA, math on CPU). This is the
    portable equivalent of the fused kernel the NPU used; softmax_scale = 1/sqrt(D),
    non-causal, no key padding mask (the NPU self/cross attention pass no mask)."""
    # [B, S, N, D] -> [B, N, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    # [B, N, S, D] -> [B, S, N, D]
    return out.transpose(1, 2)


class HomieSelfAttention(nn.Module):
    """Self-attention with modality-split QKV and cross-modality affine.

    Reproduces `function_before_core_attention_multi_qkv` +
    `get_query_key_value_tensors_cross_modal_affine` from the NPU code: the token
    sequence is split into [video(noise), reference, mllm(qwen)] chunks; the video
    chunk's Q/K are affine-modulated by the (raw, un-projected) mllm hidden states,
    then all chunks are recombined, rope is applied, and full self-attention runs
    over the whole sequence."""

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6, rope=None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope

        self.proj_q = nn.Linear(dim, dim, bias=True)
        self.proj_k = nn.Linear(dim, dim, bias=True)
        self.proj_v = nn.Linear(dim, dim, bias=True)
        self.proj_out = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Only self-attention carries the cross-modality affine (matches checkpoint).
        self.cross_modality_affine_query = CrossModalityAffine(dim)
        self.cross_modality_affine_key = CrossModalityAffine(dim)

    def _qkv(self, hidden, cross_modality_states=None):
        q = self.q_norm(self.proj_q(hidden))
        k = self.k_norm(self.proj_k(hidden))
        v = self.proj_v(hidden)
        if cross_modality_states is not None:
            q = self.cross_modality_affine_query(q, cross_modality_states[0])
            k = self.cross_modality_affine_key(k, cross_modality_states[1])
        return q, k, v

    def forward(self, x, rope_freqs, video_ref_mllm_tokens):
        """x: [B, S, H]; video_ref_mllm_tokens: (n_video, n_ref, n_mllm)."""
        B = x.shape[0]
        n_video, n_ref, n_mllm = video_ref_mllm_tokens

        x_video = x[:, :n_video]
        x_ref = x[:, n_video:n_video + n_ref]
        x_mllm = x[:, n_video + n_ref:]

        # video (noise) chunk is modulated by the raw mllm hidden states.
        cms = [x_mllm, x_mllm] if n_mllm > 0 else None
        q_v, k_v, v_v = self._qkv(x_video, cross_modality_states=cms)
        q_r, k_r, v_r = self._qkv(x_ref)
        if n_mllm > 0:
            q_m, k_m, v_m = self._qkv(x_mllm)
            q = torch.cat([q_v, q_r, q_m], dim=1)
            k = torch.cat([k_v, k_r, k_m], dim=1)
            v = torch.cat([v_v, v_r, v_m], dim=1)
        else:
            q = torch.cat([q_v, q_r], dim=1)
            k = torch.cat([k_v, k_r], dim=1)
            v = torch.cat([v_v, v_r], dim=1)

        S = q.shape[1]
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)

        if self.rope is not None and rope_freqs is not None:
            q = self.rope.apply_rotary_pos_emb(q, rope_freqs)
            k = self.rope.apply_rotary_pos_emb(k, rope_freqs)

        out = sdpa(q, k, v)  # [B, S, N, D]
        out = out.reshape(B, S, self.dim)
        return self.proj_out(out)


class HomieCrossAttention(nn.Module):
    """Text cross-attention (t2v style). q from latents, k/v from prompt embeddings.
    No cross-modality affine here (matches the checkpoint key layout)."""

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.proj_q = nn.Linear(dim, dim, bias=True)
        self.proj_k = nn.Linear(dim, dim, bias=True)
        self.proj_v = nn.Linear(dim, dim, bias=True)
        self.proj_out = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context):
        """x: [B, Sq, H]; context: [B, Sk, H]."""
        B = x.shape[0]
        q = self.q_norm(self.proj_q(x)).view(B, -1, self.num_heads, self.head_dim)
        k = self.k_norm(self.proj_k(context)).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(context).view(B, -1, self.num_heads, self.head_dim)
        out = sdpa(q, k, v)
        out = out.reshape(B, -1, self.dim)
        return self.proj_out(out)
