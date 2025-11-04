#!/usr/bin/env python3

"""Test script for MLX ESM2 implementation."""

import sys
import os
import numpy as np

# Add the current directory to path so we can import our MLX modules
sys.path.insert(0, os.path.dirname(__file__))

try:
    import mlx.core as mx
    from esm_mlx import ESM2MLX, ESM2Config
    print("✅ MLX and ESM2MLX imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_test_config():
    """Create a small test configuration for quick testing."""
    return ESM2Config(
        vocab_size=33,
        hidden_size=128,  # Small for testing
        num_hidden_layers=2,  # Just 2 layers for testing
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=256,
        attention_head_dim=16  # 128 // 8
    )


def test_model_creation():
    """Test creating the model."""
    print("\n🧪 Testing model creation...")
    
    config = create_test_config()
    model = ESM2MLX(config)
    
    print(f"✅ Model created with {config.num_hidden_layers} layers")
    return model, config


def test_forward_pass():
    """Test a forward pass through the model."""
    print("\n🧪 Testing forward pass...")
    
    model, config = test_model_creation()
    
    # Create test input
    batch_size = 2
    seq_len = 10
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
    
    print(f"✅ Forward pass successful")
    print(f"  - Prediction scores shape: {outputs['prediction_scores'].shape}")
    print(f"  - Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"  - Number of hidden states: {len(outputs['hidden_states']) if outputs['hidden_states'] else 0}")
    print(f"  - Number of attention layers: {len(outputs['attentions']) if outputs['attentions'] else 0}")
    
    return model, config, outputs


def test_sequence_embeddings():
    """Test sequence embedding extraction."""
    print("\n🧪 Testing sequence embeddings...")
    
    model, config, _ = test_forward_pass()
    
    # Create test input with attention mask
    batch_size = 2
    seq_len = 10
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = mx.ones((batch_size, seq_len))
    attention_mask = mx.where(mx.arange(seq_len)[None, :] < mx.array([8, 6])[:, None], 
                             attention_mask, mx.zeros_like(attention_mask))
    
    embeddings = model.get_sequence_embeddings(input_ids, attention_mask)
    
    print(f"✅ Sequence embeddings extracted")
    print(f"  - Embeddings shape: {embeddings.shape}")
    
    return embeddings


def test_protein_sequence():
    """Test with a real protein sequence."""
    print("\n🧪 Testing with protein sequence...")
    
    # Simple protein sequence to amino acid mapping (simplified)
    protein_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    # For testing, we'll just use random token IDs 
    # In real implementation, we'd use proper ESM tokenization
    model, config = test_model_creation()
    
    seq_len = min(len(protein_seq), 50)  # Limit length for testing
    input_ids = mx.random.randint(0, config.vocab_size, (1, seq_len))
    
    outputs = model(input_ids, return_contacts=True, output_attentions=True)
    
    print(f"✅ Protein sequence processed")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Output shape: {outputs['prediction_scores'].shape}")
    if outputs['contacts'] is not None:
        print(f"  - Contacts shape: {outputs['contacts'].shape}")
    
    return outputs


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting ESM2 MLX Tests")
    print("=" * 50)
    
    try:
        # Test 1: Model creation
        test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass()
        
        # Test 3: Sequence embeddings
        test_sequence_embeddings()
        
        # Test 4: Protein sequence
        test_protein_sequence()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)