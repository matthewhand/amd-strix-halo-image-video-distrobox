#!/bin/bash
# Automated Qwen→LTX-2 image-to-video+audio pipeline
# Generates images with Qwen, then animates them with LTX-2
set -e

IMAGE="amd-strix-halo-image-video-toolbox:latest"
CONTAINER="comfyui-test"
SERVER="127.0.0.1:8188"
OUTPUT="/tmp/comfy-outputs"
SCRIPTS="$(cd "$(dirname "$0")/../scripts" && pwd)"

DOCKER_GPU="--device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined"
DOCKER_ENV="-e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib -e LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib"

mkdir -p "$OUTPUT"

# --- Phase 1: Generate images with Qwen ---
echo "=============================="
echo "Phase 1: Qwen Image Generation"
echo "=============================="

docker rm -f "$CONTAINER" 2>/dev/null || true

generate_image() {
    local label="$1"
    local prompt="$2"
    local output_file="$OUTPUT/qwen_input_${label}.png"

    if [ -f "$output_file" ]; then
        echo "  [$label] exists, skipping"
        return 0
    fi

    echo "  [$label] generating..."
    docker run --rm $DOCKER_GPU $DOCKER_ENV \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        -v "$OUTPUT:/output" \
        -v "$SCRIPTS/apply_qwen_patches.py:/opt/apply_qwen_patches.py:ro" \
        "$IMAGE" \
        python3 -c "
import sys, shutil, glob, os
sys.path.insert(0, '/opt/qwen-image-studio/src')
sys.path.insert(0, '/opt')
from apply_qwen_patches import apply_comprehensive_patches
apply_comprehensive_patches()
from qwen_image_mps.cli import generate_image

class Args:
    prompt = '''$prompt'''
    steps = 8
    num_images = 1
    size = '16:9'
    ultra_fast = False
    model = 'Qwen/Qwen-Image'
    no_mmap = True
    lora = None
    edit = False
    input_image = None
    output_dir = '/tmp'
    seed = $RANDOM
    guidance_scale = 1.0
    negative_prompt = 'blurry, low quality, distorted, watermark'
    batman = False
    fast = False
    targets = 'all'

generate_image(Args())
files = glob.glob('/root/.qwen-image-studio/*.png')
if files:
    latest = max(files, key=os.path.getmtime)
    shutil.copy2(latest, '/output/qwen_input_${label}.png')
    print(f'Saved: qwen_input_${label}.png')
" 2>&1 | tail -5

    if [ -f "$output_file" ]; then
        echo "  [$label] OK"
    else
        echo "  [$label] FAILED"
    fi
}

generate_image "octopus_accountant" \
    "hyperrealistic photograph of an octopus wearing tiny reading glasses sitting at a desk covered in spreadsheets and tax forms, each tentacle holding a different pen or calculator, office cubicle background, fluorescent lighting"

generate_image "cats_boardroom" \
    "corporate photograph of a serious business meeting in a luxury boardroom, but all the executives are cats in tiny suits and ties, one cat standing at a whiteboard with a laser pointer, charts showing fish stock prices"

generate_image "angry_ai_user" \
    "photorealistic close up of a frustrated middle aged man in a messy home office, red faced and furious, gripping a keyboard with white knuckles, multiple monitors showing AI chatbot responses, energy drink cans everywhere"

# --- Phase 2: Start ComfyUI ---
echo ""
echo "=============================="
echo "Phase 2: Start ComfyUI + LTX-2"
echo "=============================="

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" $DOCKER_GPU $DOCKER_ENV \
    -p 8188:8188 \
    -v "$HOME/comfy-models:/opt/ComfyUI/models" \
    -v "$OUTPUT:/opt/ComfyUI/output" \
    "$IMAGE" \
    bash -c 'cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output --disable-mmap'

echo -n "Waiting for ComfyUI"
for i in $(seq 1 90); do
    if curl -s "$SERVER/system_stats" >/dev/null 2>&1; then
        echo " ready!"
        break
    fi
    echo -n "."
    sleep 3
done

# --- Phase 3: Submit i2v+audio jobs ---
echo ""
echo "=============================="
echo "Phase 3: Submit Video+Audio Jobs"
echo "=============================="

submit_i2v() {
    local label="$1"
    local video_prompt="$2"
    local frames="${3:-97}"
    local image_file="qwen_input_${label}.png"

    if [ ! -f "$OUTPUT/$image_file" ]; then
        echo "  [$label] no source image, skipping"
        return
    fi

    docker cp "$OUTPUT/$image_file" "$CONTAINER:/opt/ComfyUI/input/$image_file"

    python3 -c "
import json, random, urllib.request
workflow = {
    '1': {'class_type': 'LowVRAMCheckpointLoader', 'inputs': {'ckpt_name': 'ltx-2-19b-dev-fp8.safetensors'}},
    '2': {'class_type': 'LowVRAMAudioVAELoader', 'inputs': {'ckpt_name': 'ltx-2-19b-dev-fp8.safetensors'}},
    '3': {'class_type': 'LTXVGemmaCLIPModelLoader', 'inputs': {
        'gemma_path': 'gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors',
        'ltxv_path': 'ltx-2-19b-dev-fp8.safetensors', 'max_length': 1024}},
    '4': {'class_type': 'CLIPTextEncode', 'inputs': {'text': '''$video_prompt''', 'clip': ['3', 0]}},
    '5': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'low quality, distorted, static, silent', 'clip': ['3', 0]}},
    '6': {'class_type': 'LoadImage', 'inputs': {'image': '$image_file'}},
    '7': {'class_type': 'LTXVImgToVideo', 'inputs': {
        'positive': ['4', 0], 'negative': ['5', 0], 'vae': ['1', 2],
        'image': ['6', 0], 'width': 768, 'height': 512, 'length': $frames,
        'batch_size': 1, 'strength': 0.85}},
    '8': {'class_type': 'LTXVEmptyLatentAudio', 'inputs': {
        'frames_number': $frames, 'frame_rate': 24, 'batch_size': 1, 'audio_vae': ['2', 0]}},
    '9': {'class_type': 'LTXVConcatAVLatent', 'inputs': {
        'video_latent': ['7', 2], 'audio_latent': ['8', 0]}},
    '10': {'class_type': 'STGGuiderAdvanced', 'inputs': {
        'model': ['1', 0], 'positive': ['7', 0], 'negative': ['7', 1],
        'skip_steps_sigma_threshold': 0.998, 'cfg_star_rescale': True,
        'sigmas': '1.0, 0.9933, 0.985, 0.9767, 0.9008, 0.618',
        'cfg_values': '8, 6, 6, 4, 3, 1', 'stg_scale_values': '4, 4, 3, 2, 1, 0',
        'stg_rescale_values': '1, 1, 1, 1, 1, 1',
        'stg_layers_indices': '[29], [29], [29], [29], [29], [29]'}},
    '11': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
    '12': {'class_type': 'BasicScheduler', 'inputs': {'model': ['1', 0], 'scheduler': 'simple', 'steps': 20, 'denoise': 1.0}},
    '13': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': random.randint(1, 10**9)}},
    '14': {'class_type': 'SamplerCustomAdvanced', 'inputs': {
        'noise': ['13', 0], 'guider': ['10', 0], 'sampler': ['11', 0],
        'sigmas': ['12', 0], 'latent_image': ['9', 0]}},
    '15': {'class_type': 'LTXVSeparateAVLatent', 'inputs': {'av_latent': ['14', 0]}},
    '16': {'class_type': 'LTXVSpatioTemporalTiledVAEDecode', 'inputs': {
        'vae': ['1', 2], 'latents': ['15', 0], 'spatial_tiles': 4, 'spatial_overlap': 1,
        'temporal_tile_length': 16, 'temporal_overlap': 2, 'last_frame_fix': False,
        'working_device': 'auto', 'working_dtype': 'auto'}},
    '17': {'class_type': 'LTXVAudioVAEDecode', 'inputs': {'audio_vae': ['2', 0], 'samples': ['15', 1]}},
    '18': {'class_type': 'CreateVideo', 'inputs': {'images': ['16', 0], 'fps': 24.0, 'audio': ['17', 0]}},
    '19': {'class_type': 'SaveVideo', 'inputs': {'video': ['18', 0], 'filename_prefix': 'i2v_${label}', 'format': 'mp4', 'codec': 'h264'}},
}
data = json.dumps({'prompt': workflow}).encode('utf-8')
req = urllib.request.Request('http://$SERVER/prompt', data=data)
req.add_header('Content-Type', 'application/json')
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print(f'  [$label] queued: ' + result.get('prompt_id', 'FAIL'))
" 2>&1
}

submit_i2v "octopus_accountant" \
    "the octopus slowly looks up from the spreadsheets at the camera with existential dread, tentacles stop moving, a phone rings loudly, papers rustle, calculator buttons click" 97

submit_i2v "cats_boardroom" \
    "the CEO cat slams its paw on the table and meows angrily, other cats turn in shock, papers fly everywhere, the fish stock chart crashes on screen, dramatic boardroom tension, gasps and meowing" 97

submit_i2v "angry_ai_user" \
    "the man slams the keyboard on the desk and yells profanity at the screen, veins bulging, a monitor flickers, he grabs his coffee mug and throws it, angry shouting and crashing sounds, keyboard keys flying" 97

echo ""
echo "=============================="
echo "Pipeline complete!"
echo "=============================="
echo "3 videos queued in ComfyUI"
echo "Monitor: http://localhost:8188"
echo "Output:  $OUTPUT/i2v_*.mp4"
echo "Browse:  http://<YOUR_HOST>:9099/videos/"
