# HOMIE-Wan NVIDIA GPU inference package.
#
# This package is a CUDA/PyTorch re-implementation of the HOMIE subject-consistent
# video customization model that was originally developed on Huawei NPU with the
# MindSpeed-MM framework (see ../homie_npu). The code layout follows the Phantom
# repository (../Phantom/phantom_wan) so that the two GPU codebases feel familiar.
#
# Only the *inference* path is ported. The model is Wan2.1-T2V-14B extended with:
#   * qwen (MLLM) feature injection via an MLPFeatureAligner connector,
#   * learnable task/reference embeddings, and
#   * a CrossModalityAffine mechanism inside the self-attention.
from .subject2video import HomieWanS2V

__all__ = ["HomieWanS2V"]
