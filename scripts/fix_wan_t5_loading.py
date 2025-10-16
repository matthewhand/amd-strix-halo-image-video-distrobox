#!/usr/bin/env python3
"""
Fix WAN T5 model loading by converting safetensors to the expected format
"""
import os
import sys
import torch
from safetensors.torch import load_file
from pathlib import Path

def convert_t5_model():
    """Convert T5 safetensors to the expected format for WAN"""
    
    # Paths
    source_path = Path("/home/matthewh/comfy-models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")
    target_path = Path("/home/matthewh/comfy-models/diffusion_models/models_t5_umt5-xxl-enc-bf16.pth")
    
    if not source_path.exists():
        print(f"Source file not found: {source_path}")
        return False
    
    print(f"Loading T5 model from: {source_path}")
    try:
        # Load the safetensors file
        state_dict = load_file(source_path)
        print(f"Loaded {len(state_dict)} tensors")
        
        # Create a converted state dict with the expected key names
        converted_dict = {}
        
        # Map from HuggingFace T5 format to WAN format
        key_mapping = {
            "shared.weight": "token_embedding.weight",
            "encoder.final_layer_norm.weight": "norm.weight",
        }
        
        # Map encoder blocks
        for key, value in state_dict.items():
            if key.startswith("encoder.block."):
                # Convert "encoder.block.0.layer.0.SelfAttention.q.weight" to "blocks.0.attn.q.weight"
                new_key = key
                new_key = new_key.replace("encoder.block.", "blocks.")
                new_key = new_key.replace(".layer.0.SelfAttention.", ".attn.")
                new_key = new_key.replace(".layer.1.DenseReluDense.", ".ffn.")
                new_key = new_key.replace(".wi_0.", ".gate.0.")
                new_key = new_key.replace(".wi_1.", ".fc1.")
                new_key = new_key.replace(".wo.", ".fc2.")
                new_key = new_key.replace(".layer_norm.", ".norm.")
                
                # Add position embedding if needed
                if "relative_attention_bias" in new_key:
                    continue  # Skip relative attention bias for now
                    
                converted_dict[new_key] = value
            elif key in key_mapping:
                converted_dict[key_mapping[key]] = value
            elif key == "shared.weight":
                converted_dict["token_embedding.weight"] = value
            elif key == "encoder.final_layer_norm.weight":
                converted_dict["norm.weight"] = value
                
        # Add missing position embeddings (initialize as zeros)
        print("Adding position embeddings...")
        for i in range(24):  # T5-XXL has 24 layers
            converted_dict[f"blocks.{i}.pos_embedding.embedding.weight"] = torch.zeros(512, 3072)
        
        print(f"Converted to {len(converted_dict)} tensors")
        
        # Save the converted model
        print(f"Saving converted model to: {target_path}")
        torch.save(converted_dict, target_path)
        print("Conversion completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    success = convert_t5_model()
    sys.exit(0 if success else 1)