# MLPFeatureAligner — verbatim port of
# homie_npu/MindSpeed-MM/mindspeed_mm/models/predictor/dits/qwenvl_3dvae_connector.py
#
# Projects the pre-extracted qwen2.5-vl (MLLM) features from `source_dim` (2048) into
# the DiT hidden space (`target_dim` = 5120). The submodule names below
# (`mlp.0/1/3/4/6/7`, `residual_proj`, `scale`) line up 1:1 with the checkpoint keys
# under `qwen_3dvae_connector.*`.
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPFeatureAligner(nn.Module):
    """Aligns a source-modality feature to the target-modality feature space."""

    def __init__(
        self,
        source_dim,
        target_dim,
        hidden_dims=[512, 256],
        use_residual=True,
        use_adaptive_scale=True,
    ):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.use_residual = use_residual
        self.use_adaptive_scale = use_adaptive_scale

        # MLP mapping network.
        layers = []
        prev_dim = source_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, target_dim))
        layers.append(nn.LayerNorm(target_dim))
        self.mlp = nn.Sequential(*layers)

        # Residual projection (used when dims differ).
        if use_residual and source_dim != target_dim:
            self.residual_proj = nn.Linear(source_dim, target_dim)
        else:
            self.residual_proj = None

        # Adaptive scale.
        if use_adaptive_scale:
            # 1D tensor (numel==1) rather than a 0-dim scalar: FSDP cannot shard
            # scalar parameters. Broadcasting in `mapped * self.scale` is unchanged.
            self.scale = nn.Parameter(torch.tensor([0.1]))

    @staticmethod
    def count_parameters(model):
        total_params = 0
        trainable_params = 0
        for param in model.parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
        }

    def forward(self, source_feat, target_feat=None):
        orig_shape = source_feat.shape[:-1]
        flat_feat = source_feat.view(-1, self.source_dim)

        mapped = self.mlp(flat_feat)

        if self.use_residual:
            flat_source = flat_feat
            if self.residual_proj is not None:
                flat_source = self.residual_proj(flat_source)
            mapped = mapped + flat_source

        if self.use_adaptive_scale:
            mapped = mapped * self.scale

        aligned_feat = mapped.view(*orig_shape, self.target_dim)

        if target_feat is not None:
            similarity = F.cosine_similarity(aligned_feat, target_feat, dim=-1, eps=1e-6)
            similarity = similarity.unsqueeze(-1)
            aligned_feat = aligned_feat * similarity + target_feat * (1 - similarity)

        return aligned_feat
