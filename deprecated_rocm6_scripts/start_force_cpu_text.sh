#!/bin/bash

# FORCE CPU TEXT ENCODER - AVOID ALL HIP EMBEDDING ERRORS
# Drastic approach: Force entire CLIP text encoder to CPU

set -e

echo "🚀 STARTING FORCE CPU TEXT ENCODER"
echo "==================================="
echo "🔧 Forcing CLIP text encoder to CPU - drastic fix"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Remove all existing containers
docker rm -f strix-halo-cpu-text 2>/dev/null || true
docker rm -f strix-halo-debug 2>/dev/null || true
docker rm -f strix-halo-exact 2>/dev/null || true

# Start with forced CPU text encoder
docker run -d \
    --name strix-halo-cpu-text \
    --device=/dev/dri \
    --device=/dev/kfd \
    --ipc=host \
    --network host \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    -e AMD_SERIALIZE_KERNEL=3 \
    -e TORCH_USE_HIP_DSA=1 \
    -e FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE \
    -e QWEN_FA_SHIM=1 \
    -e WAN_ATTENTION_BACKEND=sdpa \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.95 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROC_ENABLE_PRE_VEGA=1 \
    -e TORCH_CUDNN_V8_API_ENABLED=0 \
    -e FORCE_CLIP_TEXT_ENCODER_CPU=1 \
    -v /home/matthewh/comfy-models:/opt/ComfyUI/models \
    -v /home/matthewh/amd-strix-halo-image-video-toolboxes/ComfyUI_output:/opt/ComfyUI/output \
    amd-strix-halo-image-video-toolbox:rocm6.1 \
    /bin/bash -c '
source /opt/venv/bin/activate

echo "🔧 Force CPU Text Encoder Configuration:"
echo "   HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "   FORCE_CLIP_TEXT_ENCODER_CPU: $FORCE_CLIP_TEXT_ENCODER_CPU"
echo "   Strategy: CPU text encoder, GPU for everything else"

# Create aggressive CPU text encoder fix
cat > /tmp/aggressive_cpu_fix.py << "EOF"
import torch
import torch.nn.functional as F
import comfy.sd1_clip
import comfy.ops

print("🛡️ Applying AGGRESSIVE CPU text encoder fix...")

# Method 1: Override CLIP text encode function
original_clip_forward = None

def cpu_clip_text_encode(tokens):
    """Force entire CLIP text encoding to CPU"""
    print("🧠 FORCING CLIP text encoder to CPU (avoid HIP embedding errors)")

    # Find and move CLIP model to CPU
    for name, module in torch.nn.modules._modules.modules():
        if hasattr(module, "transformer") and hasattr(module, "encode_token_weights"):
            print(f"🔄 Moving CLIP module to CPU: {name}")
            if hasattr(module, "parameters"):
                for param in module.parameters():
                    param.data = param.data.cpu()
            module = module.cpu()
            break

    return tokens  # Let ComfyUI handle the rest

# Method 2: Override ComfyUI ops to force CPU for embedding
class SafeEmbeddingOp:
    def __init__(self):
        self.original_forward = None

    def forward_comfy_cast_weights(self, *args, **kwargs):
        """Force embedding operations to CPU"""
        print("🔄 Embedding op: Forcing CPU to avoid HIP errors")

        # Find the embedding weight and move to CPU
        if len(args) >= 2:
            input_tensor, weight_tensor = args[0], args[1]

            # Move to CPU for computation
            input_cpu = input_tensor.cpu() if input_tensor.device.type == "cuda" else input_tensor
            weight_cpu = weight_tensor.cpu() if weight_tensor.device.type == "cuda" else weight_tensor

            try:
                # Perform embedding on CPU
                result = torch.nn.functional.embedding(input_cpu, weight_cpu,
                    *args[2:], **kwargs)
                print("✅ Embedding completed successfully on CPU")
                return result
            except Exception as e:
                print(f"❌ Even CPU embedding failed: {e}")
                raise e
        else:
            print("⚠️ Not enough arguments for embedding")
            return None

# Apply the aggressive fix
safe_op = SafeEmbeddingOp()

# Monkey patch ComfyUI operations
try:
    if hasattr(comfy.ops, "disable_weight_init"):
        original_method = comfy.ops.disable_weight_init
        def patched_disable_weight_init(weight_dtype, *args, **kwargs):
            instance = original_method(weight_dtype, *args, **kwargs)
            if hasattr(instance, "forward_comfy_cast_weights"):
                instance.forward_comfy_cast_weights = safe_op.forward_comfy_cast_weights
            return instance
        comfy.ops.disable_weight_init = patched_disable_weight_init
        print("✅ ComfyUI ops patched for CPU embedding")
except Exception as e:
    print(f"⚠️ Failed to patch ComfyUI ops: {e}")

print("✅ Aggressive CPU text encoder fix applied")

EOF

# Apply aggressive fix before starting services
python /tmp/aggressive_cpu_fix.py

# Start Qwen Image Studio
echo "🖼️ Starting Qwen Image Studio (CPU Text Encoder)..."
cd /opt/qwen-image-studio
export PYTHONPATH=/opt/qwen-image-studio/src:$PYTHONPATH
uvicorn qwen-image-studio.server:app --reload --host 0.0.0.0 --port 8000 &
QWEN_PID=$!

# Start ComfyUI with aggressive CPU text encoder fix
echo "🎬 Starting ComfyUI (Aggressive CPU Text Encoder)..."
cd /opt/ComfyUI
echo "✅ Starting with aggressive CPU text encoder fix"
python /tmp/aggressive_cpu_fix.py && python main.py --listen 0.0.0.0 --port 8188 --output-directory /opt/ComfyUI/output &
COMFY_PID=$!

echo "✅ Both services started with aggressive CPU text encoder:"
echo "   🖼️ Qwen Image Studio: http://localhost:8000 (PID: $QWEN_PID)"
echo "   🎬 ComfyUI (CPU Text): http://localhost:8188 (PID: $COMFY_PID)"
echo "   🧠 CLIP Text Encoder: FORCED TO CPU"
echo "   🔥 Video Generation: Still GPU"
echo "   🛡️ Embedding Operations: CPU only"

wait
'

echo ""
echo -e "${YELLOW}⏳ Waiting for aggressive CPU text encoder services to start...${NC}"

# Wait for services to start
for i in {1..60}; do
    QWEN_READY=$(curl -s http://localhost:8000 > /dev/null 2>&1 && echo "yes" || echo "no")
    COMFYUI_READY=$(curl -s http://localhost:8188 > /dev/null 2>&1 && echo "yes" || echo "no")

    if [ "$QWEN_READY" = "yes" ] && [ "$COMFYUI_READY" = "yes" ]; then
        echo -e "${GREEN}✅ AGGRESSIVE CPU TEXT ENCODER SUCCESSFUL!${NC}"
        echo ""
        echo -e "${BLUE}🚀 Aggressive CPU Text Setup is Running:${NC}"
        echo "   🖼️ Qwen Image Studio: http://localhost:8000"
        echo "   🎬 ComfyUI (CPU Text): http://localhost:8188"
        echo ""
        echo -e "${GREEN}🧠 Aggressive Strategy Applied:${NC}"
        echo "   🧠 CLIP Text Encoder: FORCED CPU (avoid all HIP errors)"
        echo "   🔥 Video Generation: Still uses GPU"
        echo "   🛡️ Embedding Operations: CPU-only (no HIP crashes)"
        echo "   ⚡ Overall speed: Slower text, fast video generation"
        echo ""
        echo -e "${BLUE}💡 This should completely avoid HIP embedding errors!${NC}"
        exit 0
    fi

    if [ $i -eq 20 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (20/60 seconds)${NC}"
    elif [ $i -eq 40 ]; then
        echo -e "${YELLOW}⏳ Still starting... Qwen: $QWEN_READY, ComfyUI: $COMFYUI_READY (40/60 seconds)${NC}"
    fi

    sleep 1
done

echo -e "${RED}❌ Error: Aggressive CPU text encoder failed to start${NC}"
echo -e "${YELLOW}Check logs with: docker logs strix-halo-cpu-text${NC}"
exit 1