#!/usr/bin/env python3
"""
Patched wrapper for qwen-image-mps.py CLI

This script applies ROCm compatibility patches before running the Qwen Image CLI.
Use this instead of the original qwen-image-mps.py for AMD GPU support.
"""

# Enable FA shim before any imports
import os
os.environ["QWEN_FA_SHIM"] = "1"

import sys

# Add src to path
sys.path.insert(0, "src")
sys.path.insert(0, "/opt/qwen-image-studio/src")

# Apply patches BEFORE importing CLI
# Import from the patches script
sys.path.insert(0, "/opt")
from apply_qwen_patches import apply_comprehensive_patches

patches_ok = apply_comprehensive_patches()
if not patches_ok:
    print("⚠️  Warning: Failed to apply some patches. CLI may not work correctly.")

# Now import and run the original CLI
from qwen_image_mps import cli

if __name__ == "__main__":
    cli.main()
