# RoPE3DWan — 3D rotary position embedding with optional text (qwen) positions.
#
# Ported from homie_npu WanDiT.RoPE3DWan (wan_dit.py:1400-1477). The frequency
# construction math is kept identical; the only adaptation is the tensor layout:
# the NPU code operates in seq-first ("sbh") layout, whereas this GPU port keeps the
# attention tensors batch-first ([B, S, N, D]). Because the rope frequencies are
# identical across batch and head dimensions, `forward` returns a compact [S, D/2]
# complex tensor and `apply_rotary_pos_emb` broadcasts it over B and N.
import torch
import torch.nn as nn


class RoPE3DWan(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # freqs1/2/3 are the (F, H, W) axis frequencies; freqs_text is the extra axis
        # used for qwen/text token positions.
        self.freqs = self.get_freq(head_dim)

    def get_freq(self, head_dim):
        if head_dim <= 0:
            raise ValueError("head dimension must be greater than 0")

        dim1 = head_dim - 2 * (head_dim // 3)
        dim2 = head_dim // 3

        freqs1 = self.rope_params(self.max_seq_len, dim1)
        freqs2 = self.rope_params(self.max_seq_len, dim2)
        freqs3 = self.rope_params(self.max_seq_len, dim2)
        freqs_text = self.rope_params(self.max_seq_len * 2, head_dim)
        return freqs1, freqs2, freqs3, freqs_text

    def rope_params(self, max_seq_len, dim, theta=10000):
        if dim % 2 != 0:
            raise ValueError("Dimension must be even")
        # Always build on CPU: these are non-learnable constant buffers, and torch.polar
        # is not implemented on the "meta" device (so meta-device model construction in
        # verify.py still works). They are moved to the compute device lazily in forward().
        cpu = torch.device("cpu")
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=cpu)[: (dim // 2)].double() / dim))
        freqs = torch.outer(torch.arange(max_seq_len, device=cpu), freqs)
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def to_device(self, device):
        self.freqs = tuple(f.to(device) for f in self.freqs)

    def apply_rotary_pos_emb(self, tokens, freqs):
        """
        tokens: [B, S, N, D]
        freqs:  [S, D/2] complex
        Interleaved-pair RoPE, matching the NPU implementation exactly.
        """
        dtype = tokens.dtype
        B, S, N, D = tokens.shape

        # [S, D/2] complex -> real/imag -> cos/sin, each [S, D/2, 1]
        cos, sin = torch.chunk(torch.view_as_real(freqs.to(torch.complex64)), 2, dim=-1)

        def rotate_half(x):
            half_1, half_2 = torch.chunk(x.reshape((B, S, N, D // 2, 2)), 2, dim=-1)
            return torch.cat((-half_2, half_1), dim=-1).reshape((B, S, N, D))

        # duplicate each frequency across its pair -> [S, D] -> broadcast [1, S, 1, D]
        cos = cos.expand(-1, -1, 2).flatten(-2).view(1, S, 1, D)
        sin = sin.expand(-1, -1, 2).flatten(-2).view(1, S, 1, D)
        res = tokens * cos + rotate_half(tokens) * sin
        return res.to(dtype)

    def forward(self, f, h, w, seq_text=0, device=None):
        """
        Returns freqs of shape [f*h*w (+ seq_text), head_dim/2] as a complex tensor.
        The batch dimension is dropped (identical across batch) versus the NPU code.
        """
        freqs = self.freqs
        if device is not None and freqs[0].device != device:
            self.to_device(device)
            freqs = self.freqs

        seq_len = f * h * w
        vid_freqs = (
            torch.cat(
                [
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, -1)
        )

        if seq_text > 0:
            # Text/qwen positions occupy the RoPE "frame" axis (see NPU comments).
            text_freqs = freqs[3][:seq_text].reshape(seq_text, -1)
            vid_freqs = torch.cat([vid_freqs, text_freqs], dim=0)
        return vid_freqs
