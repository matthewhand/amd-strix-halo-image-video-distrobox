#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="${MAGE_LOCAL_DIR:-$ROOT/huggingface-cache/local/Mage-Flow-Turbo}"
TOKEN="${HF_TOKEN:-}"
[[ -z "$TOKEN" && -f "$HOME/.cache/huggingface/token" ]] && TOKEN="$(cat "$HOME/.cache/huggingface/token")"
[[ -z "$TOKEN" && -f "$ROOT/huggingface-cache/token" ]] && TOKEN="$(cat "$ROOT/huggingface-cache/token")"
BASE="https://huggingface.co/microsoft/Mage-Flow-Turbo/resolve/main"
mkdir -p "$LOCAL"/{scheduler,text_encoder,transformer,vae}

# Prefer complete VAE blob from hub cache if present.
HUB_VAE="$ROOT/huggingface-cache/hub/models--microsoft--Mage-Flow-Turbo/blobs/34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0"
if [[ -f "$HUB_VAE" && $(stat -c%s "$HUB_VAE") -eq 345053056 ]]; then
  cp -n "$HUB_VAE" "$LOCAL/vae/diffusion_pytorch_model.safetensors" || true
  echo "vae from hub cache"
fi

# HF LFS oid / etag — reject "correct size, all-zero weights" corrupt downloads.
declare -A EXPECT_SHA=(
  ["vae/diffusion_pytorch_model.safetensors"]=34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
  ["text_encoder/model-00001-of-00002.safetensors"]=30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9
  ["text_encoder/model-00002-of-00002.safetensors"]=046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6
  ["transformer/diffusion_pytorch_model.safetensors"]=6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d
)

verify_weight() {
  local rel="$1" out="$LOCAL/$rel" exp="${EXPECT_SHA[$rel]:-}"
  [[ -z "$exp" || ! -f "$out" ]] && return 0
  local got; got="$(sha256sum "$out" | awk '{print $1}')"
  if [[ "$got" != "$exp" ]]; then
    echo "HASH MISMATCH $rel got=$got expect=$exp" >&2
    return 1
  fi
  echo "OK $rel"
}

fetch() {
  local rel="$1"
  local out="$LOCAL/$rel"
  mkdir -p "$(dirname "$out")"
  if [[ -n "${EXPECT_SHA[$rel]:-}" && -f "$out" ]] && verify_weight "$rel"; then
    return 0
  fi
  echo "START $rel"
  aria2c -x 16 -s 16 -k 1M --continue=true --auto-file-renaming=false \
    ${TOKEN:+--header="Authorization: Bearer $TOKEN"} \
    -d "$(dirname "$out")" -o "$(basename "$out")" \
    "$BASE/$rel"
  echo "DONE $rel $(stat -c%s "$out")"
  if [[ -n "${EXPECT_SHA[$rel]:-}" ]]; then
    verify_weight "$rel" || { rm -f "$out"; exit 2; }
  fi
}

# Metadata (cheap)
for f in \
  model_index.json \
  scheduler/scheduler_config.json \
  text_encoder/chat_template.json text_encoder/config.json \
  text_encoder/generation_config.json text_encoder/merges.txt \
  text_encoder/model.safetensors.index.json \
  text_encoder/preprocessor_config.json text_encoder/tokenizer.json \
  text_encoder/tokenizer_config.json text_encoder/video_preprocessor_config.json \
  text_encoder/vocab.json transformer/config.json vae/config.json
do
  out="$LOCAL/$f"
  if [[ ! -s "$out" ]]; then
    curl -fsSL ${TOKEN:+-H "Authorization: Bearer $TOKEN"} -o "$out" "$BASE/$f" || true
  fi
done

fetch vae/diffusion_pytorch_model.safetensors
fetch text_encoder/model-00001-of-00002.safetensors
fetch text_encoder/model-00002-of-00002.safetensors
fetch transformer/diffusion_pytorch_model.safetensors

echo ALL_DONE
du -sh "$LOCAL"
find "$LOCAL" -type f -printf '%s %p\n' | sort -n | tail -10
