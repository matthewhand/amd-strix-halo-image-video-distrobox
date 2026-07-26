#!/usr/bin/env bash
# Download microsoft/Mage-Flow-Turbo weights into a local dir for ROCm inference.
# Verifies large safetensors against published HF LFS SHA-256 (etag).
# Resumes partial downloads. Prefer HF_TOKEN / ~/.cache/huggingface/token.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="${MAGE_LOCAL_DIR:-$ROOT/huggingface-cache/local/Mage-Flow-Turbo}"
TOKEN="${HF_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
  TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [[ -z "$TOKEN" && -f "$ROOT/huggingface-cache/token" ]]; then
  TOKEN="$(cat "$ROOT/huggingface-cache/token")"
fi

BASE="https://huggingface.co/microsoft/Mage-Flow-Turbo/resolve/main"
mkdir -p "$LOCAL"/{scheduler,text_encoder,transformer,vae}

# HF LFS oid / etag for the multi-GB shards (prevents "correct size, zero weights").
declare -A EXPECT_SHA=(
  ["vae/diffusion_pytorch_model.safetensors"]=34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
  ["text_encoder/model-00001-of-00002.safetensors"]=30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9
  ["text_encoder/model-00002-of-00002.safetensors"]=046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6
  ["transformer/diffusion_pytorch_model.safetensors"]=6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d
)

auth=()
if [[ -n "$TOKEN" ]]; then
  auth=(-H "Authorization: Bearer $TOKEN")
fi

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

verify_weight() {
  local rel="$1"
  local out="$LOCAL/$rel"
  local exp="${EXPECT_SHA[$rel]:-}"
  [[ -z "$exp" ]] && return 0
  [[ -f "$out" ]] || return 1
  local got
  got="$(sha256_file "$out")"
  if [[ "$got" != "$exp" ]]; then
    echo "HASH MISMATCH $rel" >&2
    echo "  got    $got" >&2
    echo "  expect $exp" >&2
    return 1
  fi
  echo "OK $rel sha256=$got"
}

fetch() {
  local rel="$1"
  local out="$LOCAL/$rel"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    if [[ -n "${EXPECT_SHA[$rel]:-}" ]]; then
      if verify_weight "$rel"; then
        return 0
      fi
      echo "re-fetch corrupt $rel"
      rm -f "$out"
    elif [[ "$rel" != *.safetensors ]]; then
      echo "skip $rel"
      return 0
    fi
  fi
  echo "GET $rel"
  curl -fL --retry 5 --retry-delay 2 -C - \
    "${auth[@]}" \
    -o "$out" \
    "$BASE/$rel"
  if [[ -n "${EXPECT_SHA[$rel]:-}" ]]; then
    verify_weight "$rel" || {
      rm -f "$out"
      exit 2
    }
  fi
}

# Metadata / tokenizer
for f in \
  model_index.json \
  scheduler/scheduler_config.json \
  text_encoder/chat_template.json \
  text_encoder/config.json \
  text_encoder/generation_config.json \
  text_encoder/merges.txt \
  text_encoder/model.safetensors.index.json \
  text_encoder/preprocessor_config.json \
  text_encoder/tokenizer.json \
  text_encoder/tokenizer_config.json \
  text_encoder/video_preprocessor_config.json \
  text_encoder/vocab.json \
  transformer/config.json \
  vae/config.json
do
  fetch "$f"
done

# Large weights (~17.5 GB total) — always hash-checked
for f in \
  vae/diffusion_pytorch_model.safetensors \
  text_encoder/model-00001-of-00002.safetensors \
  text_encoder/model-00002-of-00002.safetensors \
  transformer/diffusion_pytorch_model.safetensors
do
  fetch "$f"
done

echo "DONE $LOCAL"
du -sh "$LOCAL"
find "$LOCAL" -type f -printf '%s %p\n' | sort -n | tail -10
