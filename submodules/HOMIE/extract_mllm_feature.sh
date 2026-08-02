#!/usr/bin/env bash
# Extract Qwen3-VL MLLM features for HOMIE test samples that lack them.
#
# Reads $INPUT_JSON, writes one .pt per sample into $FEATURE_DIR, and emits a new
# meta jsonl ($OUTPUT_JSON) with an "mllm_feature" path added to each line. Feed
# $OUTPUT_JSON to generate.py (infer.sh) afterwards.
#
# Prerequisites:
#   * Qwen3-VL-2B-Thinking checkpoint in $MLLM_CKPT.
set -e

MLLM_CKPT=${MLLM_CKPT:-/mnt/bn/bes-mllm-shared/caiyiyang/hf_models/Qwen3-VL-2B-Thinking}
INPUT_JSON=${INPUT_JSON:-eval_examples/meta_file.jsonl}
FEATURE_DIR=${FEATURE_DIR:-eval_examples/mllm_features}
OUTPUT_JSON=${OUTPUT_JSON:-eval_examples/meta_file_with_mllm.jsonl}

python generate_mllm_feature.py \
    --meta_file "$INPUT_JSON" \
    --output_meta "$OUTPUT_JSON" \
    --feature_dir "$FEATURE_DIR" \
    --model_path "$MLLM_CKPT"
