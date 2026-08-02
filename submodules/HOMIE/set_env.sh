pip install -r requirements.txt
pip install xformers
pip install xfuser
# flash attention installation
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install transformers==4.57.0
pip uninstall -y opencv-python
pip install opencv-python-headless==4.5.5.64
# mllm feature extraction
pip install qwen_vl_utils
