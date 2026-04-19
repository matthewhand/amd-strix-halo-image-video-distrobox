#!/bin/bash
# Run both waves: generate ALL images first, then start ComfyUI once for ALL videos
set -e

IMAGE="amd-strix-halo-image-video-toolbox:latest"
CONTAINER="comfyui-test"
SERVER="127.0.0.1:8188"
OUTPUT="/tmp/comfy-outputs"
SCRIPTS="$(cd "$(dirname "$0")/../scripts" && pwd)"

DOCKER_GPU="--device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined"
DOCKER_ENV="-e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib -e LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib"

mkdir -p "$OUTPUT"
docker kill "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true
sleep 2

generate_image() {
    local label="$1"; local prompt="$2"
    local output_file="$OUTPUT/qwen_input_${label}.png"
    [ -f "$output_file" ] && { echo "  [$label] exists, skipping"; return 0; }
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
    steps = 8; num_images = 1; size = '16:9'; ultra_fast = False
    model = 'Qwen/Qwen-Image'; no_mmap = True; lora = None; edit = False
    input_image = None; output_dir = '/tmp'; seed = $RANDOM; guidance_scale = 1.0
    negative_prompt = 'blurry, low quality, distorted, watermark'
    batman = False; fast = False; targets = 'all'
generate_image(Args())
files = glob.glob('/root/.qwen-image-studio/*.png')
if files:
    latest = max(files, key=os.path.getmtime)
    shutil.copy2(latest, '/output/qwen_input_${label}.png')
" 2>&1 | tail -3
    [ -f "$output_file" ] && echo "  [$label] OK" || echo "  [$label] FAILED"
}

echo "======================================="
echo "PHASE 1: Generate ALL 6 Qwen images"
echo "======================================="

# Wave 1
generate_image "octopus_accountant" \
    "hyperrealistic photograph of an octopus wearing tiny reading glasses sitting at a desk covered in spreadsheets and tax forms, each tentacle holding a different pen or calculator, office cubicle background, fluorescent lighting"
generate_image "cats_boardroom" \
    "corporate photograph of a serious business meeting in a luxury boardroom, but all the executives are cats in tiny suits and ties, one cat standing at a whiteboard with a laser pointer, charts showing fish stock prices"
generate_image "angry_ai_user" \
    "photorealistic close up of a frustrated middle aged man in a messy home office, red faced and furious, gripping a keyboard with white knuckles, multiple monitors showing AI chatbot responses, energy drink cans everywhere"

# Wave 2
generate_image "medieval_astronaut" \
    "oil painting in renaissance style of an astronaut in a full NASA spacesuit sitting for a formal portrait in a medieval castle throne room, golden frame, dramatic chiaroscuro lighting, servants in period costume attending"
generate_image "dinosaur_barista" \
    "detailed illustration of a friendly T-Rex working as a barista in a modern hipster coffee shop, tiny apron on its massive body, struggling to hold a tiny espresso cup with its small arms, customers waiting patiently"
generate_image "goldfish_therapist" \
    "photorealistic scene of a goldfish in a tiny bowl sitting on a therapists chair, wearing miniature glasses, across from a stressed great white shark lying on a couch, office setting with diplomas on the wall"

echo ""
echo "======================================="
echo "PHASE 2: Start ComfyUI + submit ALL 6"
echo "======================================="

# Pick best available model
if [ -f "$HOME/comfy-models/checkpoints/ltxv-13b-0.9.8-distilled-fp8.safetensors" ]; then
    LTX_MODEL="ltxv-13b-0.9.8-distilled-fp8.safetensors"
    echo "Using LTX 0.9.8 (latest)"
else
    LTX_MODEL="ltx-2-19b-dev-fp8.safetensors"
    echo "Using LTX-2 19B (old)"
fi

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" $DOCKER_GPU $DOCKER_ENV \
    -p 8188:8188 \
    -v "$HOME/comfy-models:/opt/ComfyUI/models" \
    -v "$OUTPUT:/opt/ComfyUI/output" \
    "$IMAGE" \
    bash -c 'cd /opt/ComfyUI && python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output --disable-mmap'

echo -n "Waiting for ComfyUI"
for i in $(seq 1 90); do
    curl -s "$SERVER/system_stats" >/dev/null 2>&1 && { echo " ready!"; break; }
    echo -n "."; sleep 3
done

submit_i2v() {
    local label="$1"; local video_prompt="$2"; local frames="${3:-97}"
    local image_file="qwen_input_${label}.png"
    [ ! -f "$OUTPUT/$image_file" ] && { echo "  [$label] no image, skip"; return; }
    docker cp "$OUTPUT/$image_file" "$CONTAINER:/opt/ComfyUI/input/$image_file"
    python3 -c "
import json, random, urllib.request
wf = {
    '1': {'class_type': 'LowVRAMCheckpointLoader', 'inputs': {'ckpt_name': '$LTX_MODEL'}},
    '2': {'class_type': 'LowVRAMAudioVAELoader', 'inputs': {'ckpt_name': '$LTX_MODEL'}},
    '3': {'class_type': 'LTXVGemmaCLIPModelLoader', 'inputs': {
        'gemma_path': 'gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors',
        'ltxv_path': '$LTX_MODEL', 'max_length': 1024}},
    '4': {'class_type': 'CLIPTextEncode', 'inputs': {'text': '''$video_prompt''', 'clip': ['3', 0]}},
    '5': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'low quality, distorted, static, silent', 'clip': ['3', 0]}},
    '6': {'class_type': 'LoadImage', 'inputs': {'image': '$image_file'}},
    '7': {'class_type': 'LTXVImgToVideo', 'inputs': {
        'positive': ['4', 0], 'negative': ['5', 0], 'vae': ['1', 2],
        'image': ['6', 0], 'width': 768, 'height': 512, 'length': $frames, 'batch_size': 1, 'strength': 0.85}},
    '8': {'class_type': 'LTXVEmptyLatentAudio', 'inputs': {
        'frames_number': $frames, 'frame_rate': 24, 'batch_size': 1, 'audio_vae': ['2', 0]}},
    '9': {'class_type': 'LTXVConcatAVLatent', 'inputs': {'video_latent': ['7', 2], 'audio_latent': ['8', 0]}},
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
        'noise': ['13', 0], 'guider': ['10', 0], 'sampler': ['11', 0], 'sigmas': ['12', 0], 'latent_image': ['9', 0]}},
    '15': {'class_type': 'LTXVSeparateAVLatent', 'inputs': {'av_latent': ['14', 0]}},
    '16': {'class_type': 'LTXVSpatioTemporalTiledVAEDecode', 'inputs': {
        'vae': ['1', 2], 'latents': ['15', 0], 'spatial_tiles': 4, 'spatial_overlap': 1,
        'temporal_tile_length': 16, 'temporal_overlap': 2, 'last_frame_fix': False,
        'working_device': 'auto', 'working_dtype': 'auto'}},
    '17': {'class_type': 'LTXVAudioVAEDecode', 'inputs': {'audio_vae': ['2', 0], 'samples': ['15', 1]}},
    '18': {'class_type': 'CreateVideo', 'inputs': {'images': ['16', 0], 'fps': 24.0, 'audio': ['17', 0]}},
    '19': {'class_type': 'SaveVideo', 'inputs': {'video': ['18', 0], 'filename_prefix': 'i2v_${label}', 'format': 'mp4', 'codec': 'h264'}},
}
data = json.dumps({'prompt': wf}).encode('utf-8')
req = urllib.request.Request('http://$SERVER/prompt', data=data)
req.add_header('Content-Type', 'application/json')
resp = urllib.request.urlopen(req)
r = json.loads(resp.read())
print(f'  [$label] queued: ' + r.get('prompt_id', 'FAIL'))
" 2>&1
}

# Wave 1 videos
submit_i2v "octopus_accountant" \
    "the octopus slowly looks up from the spreadsheets at the camera with existential dread, tentacles stop moving, a phone rings loudly, papers rustle, calculator buttons click" 97
submit_i2v "cats_boardroom" \
    "the CEO cat slams its paw on the table and meows angrily, other cats turn in shock, papers fly everywhere, the fish stock chart crashes on screen, gasps and meowing" 97
submit_i2v "angry_ai_user" \
    "the man slams the keyboard on the desk and yells profanity at the screen, veins bulging, a monitor flickers, he grabs his coffee mug and throws it, angry shouting and crashing sounds" 97

# Wave 2 videos
submit_i2v "medieval_astronaut" \
    "the astronaut slowly raises the golden visor revealing a confused bearded medieval knight face, servants gasp and drop their trays, candles flicker, dramatic orchestral music, stone hall echoing" 97
submit_i2v "dinosaur_barista" \
    "the T-Rex carefully tries to pour latte art with its tiny trembling arms, milk splashing everywhere, the espresso cup shatters in its claws, customers clapping encouragingly, coffee machine hissing" 145
submit_i2v "goldfish_therapist" \
    "the goldfish adjusts its tiny glasses and nods thoughtfully, the great white shark on the couch starts crying with huge tears, the bowl wobbles, bubbling water sounds, soft emotional piano music" 145

echo ""
echo "======================================="
echo "ALL 6 VIDEOS QUEUED"
echo "======================================="
echo "Wave 1: octopus_accountant, cats_boardroom, angry_ai_user (~4s each)"
echo "Wave 2: medieval_astronaut (~4s), dinosaur_barista (~6s), goldfish_therapist (~6s)"
echo ""
echo "Model: $LTX_MODEL"
echo "Monitor: http://localhost:8188"
echo "Browse:  http://10.0.0.30:9099/videos/"
echo "Est. completion: ~1.5 hours from now"
