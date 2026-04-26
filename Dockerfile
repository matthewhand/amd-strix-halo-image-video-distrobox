FROM registry.fedoraproject.org/fedora:rawhide

# Build args
ARG INSTALL_GUI=false

# Base packages (keep compilers/headers for Triton JIT at runtime)
RUN dnf -y install --setopt=install_weak_deps=False --nodocs \
      libdrm-devel python3.13 python3.13-devel git rsync libatomic bash ca-certificates curl \
      gcc gcc-c++ binutils make git ffmpeg-free openh264 \
  && dnf clean all && rm -rf /var/cache/dnf/*

# Optional: GUI tools for distrobox (X11/Wayland image viewers)
RUN if [ "$INSTALL_GUI" = "true" ]; then \
      dnf -y install --setopt=install_weak_deps=False --nodocs feh imv && \
      dnf clean all && rm -rf /var/cache/dnf/*; \
    fi

# Python venv
RUN /usr/bin/python3.13 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
RUN printf 'source /opt/venv/bin/activate\n' > /etc/profile.d/venv.sh
RUN python -m pip install --upgrade pip setuptools wheel

# Helper scripts (ComfyUI-only)
COPY scripts/get_wan22.sh /opt/
COPY scripts/set_extra_paths.sh /opt/
COPY scripts/get_qwen_image.sh /opt/
COPY scripts/apply_qwen_patches.py /opt/
COPY scripts/start_docker.sh /opt/
COPY scripts/qwen_launcher.py /opt/
COPY scripts/qwen_tts_launcher.py /opt/
COPY scripts/qwen_tts_serve.py /opt/
COPY scripts/wan_launcher.py /opt/
COPY scripts/test_wan_permutations.py /opt/
COPY scripts/download_wan_cli.sh /opt/
COPY scripts/ernie_launcher.py /opt/
COPY scripts/get_ernie_image.sh /opt/

# ROCm + PyTorch (TheRock, include torchaudio for resolver; remove later)
ARG ROCM_INDEX=https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/
RUN python -m pip install --index-url ${ROCM_INDEX} 'rocm[libraries,devel]' && \
    python -m pip install --index-url ${ROCM_INDEX} \
      torch torchvision torchaudio pytorch-triton-rocm numpy

WORKDIR /opt

# ComfyUI
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI 
WORKDIR /opt/ComfyUI
RUN python -m pip install -r requirements.txt && \
    python -m pip install --prefer-binary \
      pillow opencv-python-headless imageio imageio-ffmpeg scipy "huggingface_hub[hf_transfer]" pyyaml

# ComfyUI plugins
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth=1 https://github.com/cubiq/ComfyUI_essentials /opt/ComfyUI/custom_nodes/ComfyUI_essentials 
RUN git clone --depth=1 https://github.com/kyuz0/ComfyUI-AMDGPUMonitor /opt/ComfyUI/custom_nodes/ComfyUI-AMDGPUMonitor
RUN git clone --depth=1 https://github.com/Lightricks/ComfyUI-LTXVideo /opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo && \
    python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt
# ComfyUI audio tensor AMD fix (Lightricks/ComfyUI-LTXVideo#361)
RUN sed -i 's/\.float()\.numpy()/.float().cpu().numpy()/g' /opt/ComfyUI/comfy_api/latest/_input_impl/video_types.py

# Qwen Image Studio
WORKDIR /opt
RUN git clone --depth=1 https://github.com/kyuz0/qwen-image-studio /opt/qwen-image-studio && \
    python -m pip install -r /opt/qwen-image-studio/requirements.txt

# Flash-Attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
    cd flash-attention &&\
    git checkout main_perf &&\
    python setup.py install && \
    cd /opt && rm -rf /opt/flash-attention

# Wan Video Studio
RUN git clone --depth=1 https://github.com/kyuz0/wan-video-studio /opt/wan-video-studio && \
    python -m pip install --prefer-binary \
      opencv-python-headless diffusers tokenizers accelerate \
      imageio[ffmpeg] easydict ftfy dashscope imageio-ffmpeg decord librosa

# heartlib (local Python package — HeartMuLa generation pipeline)
# IMPORTANT: install with --no-deps to preserve TheRock ROCm torch + ComfyUI-tracked
# transformers/numpy/tokenizers/accelerate. bitsandbytes intentionally skipped
# (heartlib doesn't import it; ROCm-incompatible).
COPY .heartlib /opt/heartlib
RUN python -m pip install --no-deps -e /opt/heartlib && \
    pip install --no-deps torchtune==0.4.0 torchao==0.9.0 vector_quantize_pytorch omegaconf 'antlr4-python3-runtime==4.9.*' && \
    pip install pyarrow dill multiprocess xxhash tiktoken sentencepiece blobfile einx && \
    pip install --no-deps 'datasets>=2.16.0' fsspec aiohttp 'huggingface-hub>=0.20.0' filelock packaging requests tqdm pyyaml typing-extensions && \
    python -c "from heartlib import HeartMuLaGenPipeline" || echo "WARN: heartlib import failed"

# Permissions & trims (keep compilers/headers)
RUN chmod -R a+rwX /opt && chmod +x /opt/*.sh /opt/*.py || true && \
    find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    python -m pip cache purge || true && rm -rf /root/.cache/pip || true && \
    dnf clean all && rm -rf /var/cache/dnf/*

# ROCm/Triton env (exports TRITON_HIP_* and LD_LIBRARY_PATH; also FA enable)
COPY scripts/01-rocm-env-for-triton.sh /etc/profile.d/01-rocm-env-for-triton.sh

# ROCm environment for Strix Halo (gfx1151)
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV QWEN_FA_SHIM=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,garbage_collection_threshold:0.8
# LTX-2 performance tweaks (from ROCm/TheRock#2845 benchmarks on gfx1151)
ENV HSA_ENABLE_SDMA=0
ENV HSA_USE_SVM=0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# Make ROCm SDK libraries available for JIT compilation (aiter) and runtime linking
ENV LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel/lib
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib

# Banner script (runs on login). Use a high sort key so it runs after venv.sh and 01-rocm-env...
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
RUN chmod 0644 /etc/profile.d/99-toolbox-banner.sh

# Keep /opt/venv/bin first after user dotfiles
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
RUN chmod 0644 /etc/profile.d/zz-venv-last.sh

# Disable core dumps in interactive shells (helps with recovering faster from ROCm crashes)
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

CMD ["/bin/bash"]

