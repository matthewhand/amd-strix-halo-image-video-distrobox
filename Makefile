# WAN and ComfyUI Test Makefile
.PHONY: help test-wan test-comfyui test-all setup-wan setup-comfyui clean

# Default target
help:
	@echo "WAN and ComfyUI Test Suite"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@echo "  help          - Show this help message"
	@echo "  setup-wan     - Setup WAN models and environment"
	@echo "  setup-comfyui - Setup ComfyUI environment"
	@echo "  test-wan      - Test WAN video generation"
	@echo "  test-comfyui  - Test ComfyUI image generation"
	@echo "  test-all      - Run all tests"
	@echo "  clean         - Clean temporary files"
	@echo ""

# Setup WAN environment
setup-wan:
	@echo "🔧 Setting up WAN environment..."
	@distrobox enter strix-halo-image-video -- /bin/bash -c "source /opt/venv/bin/activate && cd /opt/wan-video-studio && python -c 'import torch; print(\"PyTorch:\", torch.__version__)' && echo '✅ WAN environment ready'"
	@echo ""

# Setup ComfyUI environment
setup-comfyui:
	@echo "🔧 Setting up ComfyUI environment..."
	@distrobox enter strix-halo-image-video -- /bin/bash -c "source /opt/venv/bin/activate && cd /opt/ComfyUI && python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())' && echo '✅ ComfyUI environment ready'"
	@echo ""

# Test WAN video generation
test-wan:
	@echo "🎬 Testing WAN video generation..."
	@mkdir -p /tmp/wan_test_output
	@distrobox enter strix-halo-image-video -- /bin/bash -c "source /opt/venv/bin/activate && cd /opt/wan-video-studio && timeout 300 python generate.py --task ti2v-5B --prompt 'a simple test video' --size 1280*704 --frame_num 9 --ckpt_dir ~/comfy-models/diffusion_models --save_dir /tmp/wan_test_output --t5_cpu --sample_steps 2 --offload_model True" || echo "⚠️ WAN test completed with timeout or errors"
	@if [ -d "/tmp/wan_test_output" ] && [ "$$(ls -A /tmp/wan_test_output)" ]; then \
		echo "✅ WAN test files created:"; \
		ls -la /tmp/wan_test_output/ | head -10; \
		for f in /tmp/wan_test_output/*; do \
			if [ -f "$$f" ]; then \
				size=$$(du -h "$$f" | cut -f1); \
				echo "  $$(basename "$$f"): $$size"; \
			fi; \
		done; \
	else \
		echo "❌ No WAN test files created"; \
	fi
	@echo ""

# Test ComfyUI image generation
test-comfyui:
	@echo "🎨 Testing ComfyUI image generation..."
	@mkdir -p /tmp/comfyui_test_output
	@echo "Checking ComfyUI status..."
	@curl -s http://localhost:8188/ > /dev/null && echo "✅ ComfyUI is running" || echo "❌ ComfyUI not accessible"
	@echo "Testing simple image generation..."
	@python3 generators/simple-images.py || echo "⚠️ ComfyUI test completed with errors"
	@if [ -d "/tmp/comfyui_test_output" ] && [ "$$(ls -A /tmp/comfyui_test_output)" ]; then \
		echo "✅ ComfyUI test files created:"; \
		ls -la /tmp/comfyui_test_output/ | head -10; \
		for f in /tmp/comfyui_test_output/*; do \
			if [ -f "$$f" ]; then \
				size=$$(du -h "$$f" | cut -f1); \
				echo "  $$(basename "$$f"): $$size"; \
			fi; \
		done; \
	else \
		echo "❌ No ComfyUI test files created"; \
	fi
	@echo ""

# Run all tests
test-all: setup-wan setup-comfyui test-wan test-comfyui
	@echo "🎉 All tests completed!"
	@echo ""

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	@rm -rf /tmp/wan_test_output
	@rm -rf /tmp/comfyui_test_output
	@rm -f *.pyc
	@echo "✅ Clean completed"
	@echo ""

# Quick status check
status:
	@echo "📊 System Status"
	@echo "==============="
	@echo "WAN Models:"
	@ls -la ~/comfy-models/diffusion_models/ | grep -E "(wan|t5)" | head -5 || echo "  No WAN models found"
	@echo ""
	@echo "ComfyUI Status:"
	@curl -s http://localhost:8188/ > /dev/null && echo "  ✅ Running on port 8188" || echo "  ❌ Not accessible"
	@echo ""
	@echo "GPU Status:"
	@distrobox enter strix-halo-image-video -- /bin/bash -c "source /opt/venv/bin/activate && python -c 'import torch; print(\"  CUDA:\", torch.cuda.is_available()); print(\"  GPU Count:\", torch.cuda.device_count())'" 2>/dev/null || echo "  GPU status unavailable"
	@echo ""