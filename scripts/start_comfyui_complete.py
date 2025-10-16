#!/usr/bin/env python3
"""
Complete ComfyUI Launcher for AMD Strix Halo
Includes model setup and proper environment configuration
"""
import os
import sys
import subprocess
import signal
from pathlib import Path

# Add current directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

def check_rocm_environment():
    """Check if ROCm/HIP environment is properly configured"""
    required_vars = ['HSA_OVERRIDE_GFX_VERSION']
    missing_vars = []

    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        print("⚠️  Missing ROCm environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("💡 Setting default ROCm environment for Strix Halo...")
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
        print("✅ Set HSA_OVERRIDE_GFX_VERSION=11.0.0")

def setup_comfyui_environment():
    """Setup environment variables for ComfyUI"""
    # Set ROCm environment
    check_rocm_environment()

    # Python path setup
    if '/opt/venv/bin/python' not in sys.executable:
        venv_python = '/opt/venv/bin/python3'
        if Path(venv_python).exists():
            print(f"🐍 Using virtual environment: {venv_python}")
            return venv_python

    return sys.executable

def ensure_models_available():
    """Ensure essential models are available before starting ComfyUI"""
    print("🔍 Checking model availability...")

    try:
        from comfy_model_manager import ComfyModelManager
        manager = ComfyModelManager()

        if not manager.check_comfyui_installation():
            print("❌ ComfyUI installation not found")
            return False

        # Check if essential models exist
        checkpoint_dir = Path("/opt/ComfyUI/models/checkpoints")
        if not any(checkpoint_dir.glob("*.safetensors")) and not any(checkpoint_dir.glob("*.ckpt")):
            print("⚠️  No checkpoint models found!")
            print("🚀 Auto-installing essential models...")

            if manager.install_essential_models():
                print("✅ Essential models installed successfully")
            else:
                print("❌ Failed to install essential models")
                return False
        else:
            print("✅ Checkpoint models found")

        return True

    except ImportError:
        print("⚠️  Model manager not available, continuing anyway...")
        return True
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return True  # Continue anyway

def start_comfyui():
    """Start ComfyUI with proper configuration"""
    comfy_dir = Path("/opt/ComfyUI")

    if not comfy_dir.exists():
        print(f"❌ ComfyUI directory not found: {comfy_dir}")
        return False

    # Change to ComfyUI directory
    os.chdir(comfy_dir)

    # Setup environment
    python_exe = setup_comfyui_environment()

    # Ensure models are available
    if not ensure_models_available():
        print("⚠️  Continuing without model verification...")

    # ComfyUI startup arguments
    cmd = [
        python_exe,
        "main.py",
        "--listen", "0.0.0.0",
        "--port", "8188",
        "--disable-auto-launch",
        "--disable-metadata",
        "--cpu",  # Remove this if you want GPU acceleration
    ]

    # Add GPU support if ROCm is available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.remove("--cpu")
            print("🚀 GPU acceleration enabled (ROCm)")
        else:
            print("⚠️  GPU not available, using CPU")
    except ImportError:
        print("⚠️  PyTorch not available, using CPU")

    print("=" * 60)
    print("🎨 Starting ComfyUI")
    print(f"📡 Web interface: http://localhost:8188")
    print(f"🐍 Python: {python_exe}")
    print(f"📁 Directory: {comfy_dir}")
    print("=" * 60)

    try:
        # Start ComfyUI
        process = subprocess.Popen(cmd, env=os.environ.copy())

        # Handle shutdown gracefully
        def signal_handler(sig, frame):
            print("\n🛑 Shutting down ComfyUI...")
            process.terminate()
            process.wait()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Wait for process to complete
        process.wait()
        return process.returncode == 0

    except Exception as e:
        print(f"❌ Failed to start ComfyUI: {e}")
        return False

def main():
    """Main launcher function"""
    print("🚀 AMD Strix Halo ComfyUI Launcher")
    print("=" * 40)

    # Check if running in distrobox
    if os.path.exists('/.dockerenv') or os.environ.get('DISTROBOX_ENTER_DIRECTORY'):
        print("📦 Detected distrobox environment")

    # Start ComfyUI
    success = start_comfyui()

    if not success:
        print("❌ ComfyUI failed to start")
        sys.exit(1)

    print("✅ ComfyUI stopped successfully")

if __name__ == "__main__":
    main()