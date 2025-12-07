#!/usr/bin/env python3
"""
Quick WAN Model Status Checker
Shows what WAN 2.2 models you have and what's missing
"""

import os
from pathlib import Path

def check_wan_models():
    """Check WAN 2.2 model status"""
    print("🔍 WAN 2.2 Model Status Check")
    print("=" * 40)

    base_dir = Path("/home/matthewh")
    comfyui_models = base_dir / "comfy-models"
    wan_lightning = base_dir / "Wan2.2-Lightning"
    cache_dir = base_dir / ".cache/huggingface/hub"

    # Expected models
    expected_models = {
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors": "T2V High Noise (Working)",
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": "T2V Low Noise (Working)",
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors": "I2V High Noise (Channel Issue)",
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors": "I2V Low Noise (Channel Issue)",
        "wan2.2_ti2v_5B_fp16.safetensors": "TI2V 5B Model",
        "wan2.2_vae.safetensors": "WAN VAE (Required)"
    }

    print("📋 Model Status:")
    print("-" * 40)

    found_models = []
    missing_models = []
    total_size = 0

    for model_file, description in expected_models.items():
        # Check in ComfyUI models
        model_path = comfyui_models / model_file
        if model_path.exists():
            size = model_path.stat().st_size
            size_gb = size / (1024**3)
            status = "✅" if "T2V" in description else "⚠️" if "I2V" in description else "✅"
            print(f"{status} {model_file[:30]:<30} ({size_gb:.1f} GB) - {description}")
            found_models.append((model_file, size))
            total_size += size
        else:
            print(f"❌ {model_file[:30]:<30} - {description}")
            missing_models.append(model_file)

    print("-" * 40)
    print(f"📊 Found: {len(found_models)}/{len(expected_models)} models")
    print(f"💾 Total size: {total_size:.1f} GB")

    # Check additional models in directories
    print("\n📁 Additional Models Found:")
    print("-" * 40)

    all_wan_files = []
    for pattern in ["*wan*.safetensors", "*WAN*.safetensors"]:
        all_wan_files.extend(comfyui_models.glob(pattern))

    # Remove duplicates and expected models
    additional_files = []
    for file_path in all_wan_files:
        if file_path.name not in expected_models:
            size = file_path.stat().st_size / (1024**3)
            additional_files.append((file_path.name, size))

    for name, size in additional_files:
        print(f"📄 {name[:40]:<40} ({size:.1f} GB)")

    # Check Lightning LoRAs
    print("\n⚡ WAN Lightning LoRAs:")
    print("-" * 40)
    if wan_lightning.exists():
        lora_files = list(wan_lightning.rglob("*Lightning*.safetensors"))
        for lora in lora_files:
            size = lora.stat().st_size / (1024**2)
            print(f"📄 {lora.name[:40]:<40} ({size:.1f} MB)")
    else:
        print("❌ WAN Lightning directory not found")

    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 40)

    if any("T2V" in desc for desc in expected_models.values() if any(found in [f[0] for f in found_models] for found in [desc.split()[0] for desc in expected_models.values()])):
        print("✅ T2V models available - Use Text-to-Video for reliable generation")

    if any("I2V" in desc and any(found in model for found, model in found_models) for desc in expected_models.values()):
        print("⚠️  I2V models present but may have channel compatibility issues")
        print("   Consider using the Comfy-Org/Wan_2.2_ComfyUI_Repackaged models")

    if missing_models:
        print(f"📥 Missing {len(missing_models)} models - Run download script")

    if total_size < 20:  # Less than 20GB suggests missing core models
        print("🔄 Consider downloading full ComfyUI repackaged models")

if __name__ == "__main__":
    check_wan_models()