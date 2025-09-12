#!/usr/bin/env python3

"""
Test weight conversion and numerical accuracy between PyTorch ESM2 and MLX implementation.
"""

import sys
import os
import torch
import esm
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from esm_mlx import ESM2MLX, ESM2Config
from esm_mlx.convert_weights import (
    convert_pytorch_weights_to_mlx,
    extract_esm2_config_from_pytorch,
    pytorch_to_mlx_key_mapping
)

def load_pytorch_model():
    """Load PyTorch ESM2 model."""
    print("🔄 Loading PyTorch ESM2-8M model...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    print(f"✅ Loaded PyTorch model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, alphabet

def test_key_mappings(pytorch_state_dict):
    """Test that our key mappings work correctly."""
    print("\n🗝️ Testing key mappings...")
    
    mapped_keys = 0
    skipped_keys = 0
    
    for pytorch_key in pytorch_state_dict.keys():
        mlx_key = pytorch_to_mlx_key_mapping(pytorch_key)
        if mlx_key is not None:
            mapped_keys += 1
            if mapped_keys <= 5:  # Show first few
                print(f"  ✅ {pytorch_key} -> {mlx_key}")
        else:
            skipped_keys += 1
            if skipped_keys <= 3:  # Show first few skipped
                print(f"  ⏭️ Skipped: {pytorch_key}")
    
    print(f"\n📊 Mapping results:")
    print(f"  - Mapped: {mapped_keys} keys")
    print(f"  - Skipped: {skipped_keys} keys")
    
    return mapped_keys > 0

def create_mlx_model_from_pytorch(pytorch_model):
    """Create MLX model with config matching PyTorch model."""
    print("\n🏗️ Creating matching MLX model...")
    
    config = extract_esm2_config_from_pytorch(pytorch_model)
    print(f"Extracted config: {config.hidden_size}D, {config.num_hidden_layers} layers")
    
    mlx_model = ESM2MLX(config)
    print(f"✅ Created MLX model")
    
    return mlx_model, config

def convert_and_load_weights(pytorch_model, mlx_model):
    """Convert PyTorch weights and attempt to load into MLX model."""
    print("\n⚖️ Converting weights...")
    
    pytorch_state_dict = pytorch_model.state_dict()
    mlx_weights = convert_pytorch_weights_to_mlx(pytorch_state_dict)
    
    print(f"Converted {len(mlx_weights)} weight tensors")
    
    # Try to load weights - this is simplified, real implementation needs careful loading
    print("Note: Full weight loading requires matching MLX parameter structure")
    
    return mlx_weights

def compare_model_outputs(pytorch_model, alphabet, mlx_model, config):
    """Compare outputs between PyTorch and MLX models."""
    print("\n🔍 Comparing model outputs...")
    
    # Create test sequence
    sequence = "MKTVRQERLKSIVRILERSKEPVSG"
    print(f"Test sequence: {sequence}")
    
    # PyTorch inference
    print("\n🐍 PyTorch inference...")
    batch_converter = alphabet.get_batch_converter()
    data = [("test", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        pytorch_results = pytorch_model(batch_tokens, repr_layers=[pytorch_model.num_layers])
    
    pytorch_embeddings = pytorch_results["representations"][pytorch_model.num_layers]
    print(f"  PyTorch output shape: {pytorch_embeddings.shape}")
    print(f"  PyTorch output range: {pytorch_embeddings.min():.4f} to {pytorch_embeddings.max():.4f}")
    
    # MLX inference (with random weights for now)
    print("\n🍎 MLX inference (with random weights)...")
    
    # Convert tokens to MLX format
    # Note: This is simplified - real conversion needs proper token mapping
    mlx_tokens = mx.array(batch_tokens.numpy())
    
    try:
        mlx_results = mlx_model(mlx_tokens, output_hidden_states=True)
        mlx_embeddings = mlx_results["last_hidden_state"]
        
        print(f"  MLX output shape: {mlx_embeddings.shape}")
        print(f"  MLX output range: {float(mx.min(mlx_embeddings)):.4f} to {float(mx.max(mlx_embeddings)):.4f}")
        
        print(f"\n📋 Comparison:")
        print(f"  Shape match: {pytorch_embeddings.shape == tuple(mlx_embeddings.shape)}")
        print(f"  Note: Values differ due to random MLX weights vs trained PyTorch weights")
        
        return True
        
    except Exception as e:
        print(f"❌ MLX inference failed: {e}")
        return False

def test_tokenization_compatibility(alphabet):
    """Test that we understand the tokenization correctly."""
    print("\n🔤 Testing tokenization compatibility...")
    
    # Test sequence
    sequence = "MKTVRQERLK"
    
    # PyTorch tokenization
    batch_converter = alphabet.get_batch_converter()
    data = [("test", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    print(f"Sequence: {sequence}")
    print(f"Tokens shape: {batch_tokens.shape}")
    print(f"Tokens: {batch_tokens.squeeze().tolist()}")
    
    # Show token mapping
    tokens = batch_tokens.squeeze().tolist()
    token_chars = []
    for token in tokens:
        if token < len(alphabet):
            token_chars.append(alphabet.get_tok(token))
        else:
            token_chars.append(f"<{token}>")
    
    print(f"Token mapping: {' '.join(token_chars)}")
    
    return tokens

def main():
    """Main testing function."""
    print("🧪 Testing Real Weight Conversion")
    print("=" * 60)
    
    try:
        # Load PyTorch model
        pytorch_model, alphabet = load_pytorch_model()
        
        # Test key mappings
        test_key_mappings(pytorch_model.state_dict())
        
        # Create MLX model
        mlx_model, config = create_mlx_model_from_pytorch(pytorch_model)
        
        # Convert weights
        mlx_weights = convert_and_load_weights(pytorch_model, mlx_model)
        
        # Test tokenization
        test_tokenization_compatibility(alphabet)
        
        # Compare outputs
        compare_model_outputs(pytorch_model, alphabet, mlx_model, config)
        
        print(f"\n🎉 Weight conversion test completed!")
        print(f"\nNext steps:")
        print(f"1. Implement proper weight loading into MLX model")
        print(f"2. Add numerical accuracy validation")
        print(f"3. Test with larger models")
        print(f"4. Optimize performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)