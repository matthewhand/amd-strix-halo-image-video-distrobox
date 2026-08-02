set -e

CKPT_DIR=${CKPT_DIR:-/path/to/hf_models/Wan2.1-T2V-14B-Diffusers}
HOMIE_CKPT=${HOMIE_CKPT:-/path/to/hf_models/Homie-Wan-Models}
INPUT_JSON=${INPUT_JSON:-./eval_examples/meta_file_with_mllm.jsonl}
SAVE_PATH=${SAVE_PATH:-./video_results_480p}

# -------------------------------------------------------------------------------------
# 1) Single-GPU, batch inference over a HOMIE meta jsonl (reference_paths/prompt/qwen).
# -------------------------------------------------------------------------------------
# python generate.py \
#     --task s2v-14B \
#     --size 832*480 \
#     --frame_num 97 \
#     --sample_fps 24 \
#     --ckpt_dir "$CKPT_DIR" \
#     --homie_ckpt "$HOMIE_CKPT" \
#     --input_json "$INPUT_JSON" \
#     --save_path "$SAVE_PATH" \
#     --base_seed 6666


# -------------------------------------------------------------------------------------
# 2) Multi-GPU with FSDP (DiT + T5 sharded across 8 GPUs). Samples are data-parallel.
#    NOTE: context parallel (--ulysses_size) is NOT supported yet; keep it at 1.
# -------------------------------------------------------------------------------------
torchrun --nproc_per_node=8 --master_port 12345 generate.py \
    --task s2v-14B --size 832*480 --frame_num 97 --sample_fps 24 \
    --ckpt_dir "$CKPT_DIR" --homie_ckpt "$HOMIE_CKPT" \
    --input_json "$INPUT_JSON" --save_path "$SAVE_PATH" \
    --dit_fsdp --t5_fsdp --base_seed 6666
