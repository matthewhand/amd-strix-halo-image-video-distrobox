FROM registry.fedoraproject.org/fedora:rawhide

# Base packages (keep compilers/headers for Triton JIT at runtime)
RUN dnf -y install --setopt=install_weak_deps=False --nodocs \
      libdrm-devel python3.13 python3.13-devel git rsync libatomic bash ca-certificates curl \
      gcc gcc-c++ binutils make \
  && dnf clean all && rm -rf /var/cache/dnf/*

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

# ROCm + PyTorch (TheRock, include torchaudio for resolver; remove later)
ARG ROCM_INDEX=https://d2awnip2yjpvqn.cloudfront.net/v2/gfx1151/
RUN python -m pip install --index-url ${ROCM_INDEX} 'rocm[libraries,devel]' && \
    python -m pip install --index-url ${ROCM_INDEX} \
      torch torchvision torchaudio pytorch-triton-rocm numpy

WORKDIR /opt

# ComfyUI
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    rm -rf /opt/ComfyUI/.git
WORKDIR /opt/ComfyUI
RUN python -m pip install -r requirements.txt && \
    python -m pip install --prefer-binary \
      pillow opencv-python-headless imageio imageio-ffmpeg scipy "huggingface_hub[hf_transfer]" pyyaml
RUN python -m pip uninstall -y torchaudio

# ComfyUI plugins
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth=1 https://github.com/cubiq/ComfyUI_essentials /opt/ComfyUI/custom_nodes/ComfyUI_essentials && \
    rm -rf /opt/ComfyUI/custom_nodes/ComfyUI_essentials/.git
RUN git clone --depth=1 https://github.com/kyuz0/ComfyUI-AMDGPUMonitor /opt/ComfyUI/custom_nodes/ComfyUI-AMDGPUMonitor && \
    rm -rf /opt/ComfyUI/custom_nodes/ComfyUI-AMDGPUMonitor/.git

# Qwen Image Studio
WORKDIR /opt
RUN git clone --depth=1 https://github.com/kyuz0/qwen-image-studio /opt/qwen-image-studio && \
    python -m pip install -r /opt/qwen-image-studio/requirements.txt && \
    rm -rf /opt/qwen-image-studio/.git

# Flash-Attention
RUN python -m pip install "triton==3.2.0"
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
      imageio[ffmpeg] easydict ftfy dashscope imageio-ffmpeg && \
    rm -rf /opt/wan-video-studio/.git

# Permissions & trims (keep compilers/headers)
RUN chmod -R a+rwX /opt && chmod +x /opt/*.sh || true && \
    find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    python -m pip cache purge || true && rm -rf /root/.cache/pip || true && \
    dnf clean all && rm -rf /var/cache/dnf/*

# ROCm/Triton env (exports TRITON_HIP_* and LD_LIBRARY_PATH; also FA enable)
COPY scripts/01-rocm-env-for-triton.sh /etc/profile.d/01-rocm-env-for-triton.sh

# Banner script (runs on login). Use a high sort key so it runs after venv.sh and 01-rocm-env...
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
RUN chmod 0644 /etc/profile.d/99-toolbox-banner.sh

# Ensure /opt/venv/bin wins over ~/.local/bin, and prefer the venv rocm-smi
RUN cat >/etc/profile.d/zz-venv-first.sh <<'EOF'
# /etc/profile.d/zz-venv-first.sh
case ":$PATH:" in
  *":/opt/venv/bin:"*) : ;;             # already present
  *) PATH="/opt/venv/bin:$PATH" ;;      # prepend if missing
esac
alias rocm-smi='/opt/venv/bin/rocm-smi'
EOF
RUN chmod 0644 /etc/profile.d/zz-venv-first.sh

CMD ["/bin/bash"]

