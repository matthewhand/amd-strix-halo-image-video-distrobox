# Context-parallel (Ulysses/Ring) support — INTENTIONALLY NOT IMPLEMENTED yet.
#
# The NPU HOMIE model supports context parallel via MindSpeed's ulysses machinery.
# Porting it to xDiT USP is non-trivial here because HOMIE's self-attention is not a
# plain full-sequence attention: it splits the token stream into
# [video, reference, mllm] chunks and applies a CrossModalityAffine modulation to the
# video chunk using the *pooled* mllm features (see modules/attention.py). A correct
# Ulysses port must all-gather across the sequence-parallel group before the modality
# split (as the NPU code does in function_before_core_attention_multi_qkv) and
# re-split afterwards. This is deferred to a later task (TODO_CLAUDE TASK3 / todo.md
# TASK3: "--ulysses_size 8").
#
# Until then, generate.py accepts --ulysses_size but asserts it is 1, and FSDP
# (--dit_fsdp / --t5_fsdp) is the supported multi-GPU path.


def usp_attn_forward(*args, **kwargs):
    raise NotImplementedError(
        "Context parallel (Ulysses/Ring) is not yet implemented for HOMIE-Wan. "
        "Use single-GPU or FSDP (--dit_fsdp/--t5_fsdp) with --ulysses_size 1. "
        "See homie_wan/distributed/context_parallel.py for why."
    )


def usp_dit_forward(*args, **kwargs):
    raise NotImplementedError(
        "Context parallel (Ulysses/Ring) is not yet implemented for HOMIE-Wan."
    )
