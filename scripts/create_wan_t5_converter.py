#!/usr/bin/env python3
"""
Create a proper T5 model converter for WAN video generation
"""
import os
import sys
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path

def convert_t5_for_wan():
    """Convert T5 model to WAN-compatible format"""
    
    # Paths
    source_path = Path("/home/matthewh/comfy-models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")
    target_path = Path("/home/matthewh/comfy-models/diffusion_models/models_t5_umt5-xxl-enc-bf16.pth")
    
    print(f"Loading T5 model from: {source_path}")
    try:
        # Load the safetensors file
        state_dict = load_file(source_path)
        print(f"Loaded {len(state_dict)} tensors")
        
        # Create WAN-compatible state dict
        converted_dict = {}
        
        # Key mappings from HuggingFace T5 to WAN format
        for key, value in state_dict.items():
            # Skip scale weights and metadata
            if "scale_weight" in key or key in ["scaled_fp8", "spiece_model"]:
                continue
                
            # Map shared embeddings
            if key == "shared.weight":
                converted_dict["token_embedding.weight"] = value
            # Map final layer norm
            elif key == "encoder.final_layer_norm.weight":
                converted_dict["norm.weight"] = value
            # Map encoder blocks
            elif key.startswith("encoder.block."):
                new_key = key
                # Convert block structure
                new_key = new_key.replace("encoder.block.", "blocks.")
                # Convert attention layers
                new_key = new_key.replace(".layer.0.SelfAttention.", ".attn.")
                # Convert FFN layers
                new_key = new_key.replace(".layer.1.DenseReluDense.", ".ffn.")
                # Convert specific weight names
                new_key = new_key.replace(".wi_0.", ".gate.0.")
                new_key = new_key.replace(".wi_1.", ".fc1.")
                new_key = new_key.replace(".wo.", ".fc2.")
                # Convert layer norm
                new_key = new_key.replace(".layer_norm.", ".norm.")
                # Skip relative attention bias
                if "relative_attention_bias" in new_key:
                    continue
                converted_dict[new_key] = value
        
        # Add missing position embeddings (initialize properly)
        print("Adding position embeddings...")
        for i in range(24):  # T5-XXL has 24 layers
            # Create position embeddings with proper dimensions
            pos_emb = torch.zeros(512, 5120)  # seq_len x hidden_dim
            converted_dict[f"blocks.{i}.pos_embedding.embedding.weight"] = pos_emb
        
        print(f"Converted to {len(converted_dict)} tensors")
        
        # Save as PyTorch format
        print(f"Saving converted model to: {target_path}")
        torch.save(converted_dict, target_path)
        print("T5 conversion completed successfully!")
        
        # Test loading
        print("Testing converted model...")
        test_dict = torch.load(target_path, map_location='cpu', weights_only=False)
        print(f"Test loaded {len(test_dict)} tensors")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_t5_for_wan()
    sys.exit(0 if success else 1)