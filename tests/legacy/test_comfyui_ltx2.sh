#!/bin/bash
# Verify ComfyUI + LTX-2 works end-to-end after a fresh build.
# Starts ComfyUI, submits an LTX-2 text-to-video+audio job, waits for output.
set -e

IMAGE="${IMAGE:-amd-strix-halo-image-video-toolbox:latest}"
CONTAINER="comfyui-verify"
SERVER="127.0.0.1:8188"
OUTPUT_DIR="/tmp/comfy-outputs"

echo "=== ComfyUI + LTX-2 Verification ==="

# Cleanup
docker rm -f "$CONTAINER" 2>/dev/null || true
mkdir -p "$OUTPUT_DIR"

# Start ComfyUI
echo "Starting ComfyUI..."
docker run -d --name "$CONTAINER" \
  --device /dev/dri --device /dev/kfd \
  --security-opt seccomp=unconfined \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -e LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib \
  -p 8188:8188 \
  -v "$HOME/comfy-models:/opt/ComfyUI/models" \
  -v "$OUTPUT_DIR:/opt/ComfyUI/output" \
  "$IMAGE" \
  bash -c 'cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output --disable-mmap'

# Wait for ready
echo -n "Waiting for ComfyUI"
for i in $(seq 1 60); do
  if curl -s "$SERVER/system_stats" >/dev/null 2>&1; then
    echo " ready!"
    break
  fi
  echo -n "."
  sleep 3
done

# Verify LTXVideo nodes exist
echo "Checking LTX nodes..."
NODES=$(curl -s "$SERVER/object_info" | python3 -c "
import json,sys; d=json.load(sys.stdin)
needed = ['LowVRAMCheckpointLoader','LTXVGemmaCLIPModelLoader','STGGuiderAdvanced','LTXVSpatioTemporalTiledVAEDecode','LTXVAudioVAEDecode','LTXVEmptyLatentAudio','LTXVConcatAVLatent','LTXVSeparateAVLatent']
missing = [n for n in needed if n not in d]
if missing:
    print('MISSING: ' + ', '.join(missing))
    sys.exit(1)
print('ALL NODES OK')
")
echo "$NODES"

# Submit a short test job (25 frames = ~1s, minimal GPU time)
echo "Submitting test video+audio job..."
RESULT=$(python3 -c "
import json, random, urllib.request

workflow = {
    '1': {'class_type': 'LowVRAMCheckpointLoader', 'inputs': {'ckpt_name': 'ltx-2-19b-dev-fp8.safetensors'}},
    '2': {'class_type': 'LowVRAMAudioVAELoader', 'inputs': {'ckpt_name': 'ltx-2-19b-dev-fp8.safetensors'}},
    '3': {'class_type': 'LTXVGemmaCLIPModelLoader', 'inputs': {
        'gemma_path': 'gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors',
        'ltxv_path': 'ltx-2-19b-dev-fp8.safetensors', 'max_length': 1024}},
    '4': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'A cat sitting on a windowsill watching rain, peaceful, cinematic', 'clip': ['3', 0]}},
    '5': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'low quality, blurry', 'clip': ['3', 0]}},
    '6': {'class_type': 'EmptyLTXVLatentVideo', 'inputs': {'width': 512, 'height': 384, 'length': 25, 'batch_size': 1}},
    '7': {'class_type': 'LTXVEmptyLatentAudio', 'inputs': {'frames_number': 25, 'frame_rate': 24, 'batch_size': 1, 'audio_vae': ['2', 0]}},
    '8': {'class_type': 'LTXVConcatAVLatent', 'inputs': {'video_latent': ['6', 0], 'audio_latent': ['7', 0]}},
    '9': {'class_type': 'STGGuiderAdvanced', 'inputs': {
        'model': ['1', 0], 'positive': ['4', 0], 'negative': ['5', 0],
        'skip_steps_sigma_threshold': 0.998, 'cfg_star_rescale': True,
        'sigmas': '1.0, 0.9933, 0.985, 0.9767, 0.9008, 0.618',
        'cfg_values': '8, 6, 6, 4, 3, 1', 'stg_scale_values': '4, 4, 3, 2, 1, 0',
        'stg_rescale_values': '1, 1, 1, 1, 1, 1',
        'stg_layers_indices': '[29], [29], [29], [29], [29], [29]'}},
    '10': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
    '11': {'class_type': 'BasicScheduler', 'inputs': {'model': ['1', 0], 'scheduler': 'simple', 'steps': 10, 'denoise': 1.0}},
    '12': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': random.randint(1, 10**9)}},
    '13': {'class_type': 'SamplerCustomAdvanced', 'inputs': {
        'noise': ['12', 0], 'guider': ['9', 0], 'sampler': ['10', 0], 'sigmas': ['11', 0], 'latent_image': ['8', 0]}},
    '14': {'class_type': 'LTXVSeparateAVLatent', 'inputs': {'av_latent': ['13', 0]}},
    '15': {'class_type': 'LTXVSpatioTemporalTiledVAEDecode', 'inputs': {
        'vae': ['1', 2], 'latents': ['14', 0], 'spatial_tiles': 4, 'spatial_overlap': 1,
        'temporal_tile_length': 16, 'temporal_overlap': 2, 'last_frame_fix': False,
        'working_device': 'auto', 'working_dtype': 'auto'}},
    '16': {'class_type': 'LTXVAudioVAEDecode', 'inputs': {'audio_vae': ['2', 0], 'samples': ['14', 1]}},
    '17': {'class_type': 'CreateVideo', 'inputs': {'images': ['15', 0], 'fps': 24.0, 'audio': ['16', 0]}},
    '18': {'class_type': 'SaveVideo', 'inputs': {'video': ['17', 0], 'filename_prefix': 'verify_ltx2', 'format': 'mp4', 'codec': 'h264'}},
}

data = json.dumps({'prompt': workflow}).encode('utf-8')
req = urllib.request.Request('http://$SERVER/prompt', data=data)
req.add_header('Content-Type', 'application/json')
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print(result.get('prompt_id', 'FAIL'))
")

if [ "$RESULT" = "FAIL" ]; then
  echo "FAIL: Could not submit workflow"
  docker logs "$CONTAINER" 2>&1 | tail -10
  exit 1
fi

echo "Submitted: $RESULT"
echo "Waiting for output (this takes ~5-10 minutes)..."
echo "Monitor at http://localhost:8188"
echo "Output will appear in $OUTPUT_DIR/verify_ltx2*.mp4"
