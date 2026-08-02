# FSDP sharding — reused from Phantom phantom_wan/distributed/fsdp.py.
#
# Wraps the DiT (or T5) in FullyShardedDataParallel with bf16 params / fp32 reduce,
# sharding at the block granularity (each transformer block is its own FSDP unit).
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


def _get_transformer_blocks(model):
    """Return the list of transformer blocks to shard as individual FSDP units.

    Handles both the DiT (``HomieWanModel`` exposes ``.blocks``) and the HF
    ``UMT5EncoderModel`` text encoder (blocks live at ``.encoder.block``).
    """
    if hasattr(model, "blocks"):
        return list(model.blocks)
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "block"):
        return list(encoder.block)
    raise AttributeError(
        f"Cannot locate transformer blocks on {type(model).__name__} for FSDP wrapping"
    )


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    blocks = set(_get_transformer_blocks(model))
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in blocks,
        ),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        ),
        device_id=device_id,
        sync_module_states=sync_module_states,
    )
    return model
