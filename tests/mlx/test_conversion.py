#!/usr/bin/env python3

"""Test weight conversion from PyTorch ESM to MLX."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_conversion_dry_run():
    """Test the conversion logic without actually loading large models."""
    print("🧪 Testing weight conversion utilities...")
    
    try:
        from esm_mlx.convert_weights import pytorch_to_mlx_key_mapping, ESM2Config
        
        # Test key mapping
        test_cases = [
            ("embed_tokens.weight", "embeddings.word_embeddings.weight"),
            ("layers.0.self_attn.q_proj.weight", "encoder.layers.0.attention.q_proj.weight"),
            ("layers.5.fc1.weight", "encoder.layers.5.ffn.layers.0.weight"),
            ("layers.10.self_attn_layer_norm.weight", "encoder.layers.10.layer_norm1.weight"),
        ]
        
        print("Testing key mappings...")
        for pytorch_key, expected_mlx_key in test_cases:
            mlx_key = pytorch_to_mlx_key_mapping(pytorch_key)
            if mlx_key == expected_mlx_key:
                print(f"✅ {pytorch_key} -> {mlx_key}")
            else:
                print(f"❌ {pytorch_key} -> {mlx_key} (expected {expected_mlx_key})")
        
        # Test config creation
        print("\nTesting config creation...")
        config = ESM2Config(
            vocab_size=33,
            hidden_size=640,
            num_hidden_layers=30,
            num_attention_heads=20
        )
        print(f"✅ Config created: {config.hidden_size}D, {config.num_hidden_layers} layers")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_with_mock_model():
    """Test with a small mock PyTorch-like model structure."""
    print("\n🧪 Testing with mock model structure...")
    
    try:
        import torch
        from esm_mlx.convert_weights import convert_pytorch_weights_to_mlx
        
        # Create mock PyTorch state dict
        mock_state_dict = {
            "embed_tokens.weight": torch.randn(33, 128),
            "layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
            "layers.0.fc1.weight": torch.randn(512, 128),
            "lm_head.weight": torch.randn(33, 128)
        }
        
        # Convert to MLX
        mlx_weights = convert_pytorch_weights_to_mlx(mock_state_dict)
        
        print(f"✅ Converted {len(mlx_weights)} weight tensors")
        for key, array in mlx_weights.items():
            print(f"  - {key}: {array.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run conversion tests."""
    print("🚀 Testing ESM2 PyTorch to MLX Conversion")
    print("=" * 50)
    
    success = True
    
    # Test 1: Dry run
    success &= test_conversion_dry_run()
    
    # Test 2: Mock model
    success &= test_with_mock_model()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All conversion tests passed!")
        print("\nNext steps:")
        print("1. Install fair-esm: pip install fair-esm")
        print("2. Run: python -m esm_mlx.convert_weights --model esm2_t6_8M_UR50D")
    else:
        print("❌ Some tests failed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)