#!/usr/bin/env python3

"""
Debug the layer normalization issue.
"""

import torch
import esm
import numpy as np
import mlx.core as mx
import mlx.nn as nn

def test_layernorm_directly():
    """Test layer norm implementation directly."""
    print("🔍 Testing LayerNorm Implementation")
    print("=" * 50)
    
    # Create test data
    test_data = np.random.randn(1, 5, 320).astype(np.float32)
    
    # PyTorch LayerNorm
    pt_input = torch.tensor(test_data)
    pt_ln = torch.nn.LayerNorm(320, eps=1e-5)
    
    # Set specific weights for testing
    pt_ln.weight.data.fill_(1.0)
    pt_ln.bias.data.fill_(0.0)
    
    with torch.no_grad():
        pt_output = pt_ln(pt_input)
    
    # MLX LayerNorm
    mlx_input = mx.array(test_data)
    mlx_ln = nn.LayerNorm(320, eps=1e-5)
    mlx_ln.weight = mx.ones((320,))
    mlx_ln.bias = mx.zeros((320,))
    
    mlx_output = mlx_ln(mlx_input)
    
    # Compare
    pt_np = pt_output.numpy()
    mlx_np = np.array(mlx_output)
    
    max_diff = np.max(np.abs(pt_np - mlx_np))
    
    print(f"LayerNorm test with same weights:")
    print(f"  Input range: {test_data.min():.4f} to {test_data.max():.4f}")
    print(f"  PyTorch output range: {pt_np.min():.4f} to {pt_np.max():.4f}")
    print(f"  MLX output range: {mlx_np.min():.4f} to {mlx_np.max():.4f}")
    print(f"  Max diff: {max_diff:.8f}")
    
    return max_diff < 1e-6

def test_position_embeddings():
    """Check if we're missing position embeddings."""
    print("\n🔍 Checking Position Embeddings")
    print("=" * 50)
    
    # Load PyTorch model
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    
    # Check if PyTorch model has position embeddings
    print("PyTorch model components:")
    for name, module in pytorch_model.named_modules():
        if 'pos' in name.lower() or 'position' in name.lower():
            print(f"  {name}: {type(module)}")
    
    # Check state dict for position-related parameters
    print("\nPosition-related parameters:")
    for name in pytorch_model.state_dict().keys():
        if 'pos' in name.lower() or 'position' in name.lower():
            print(f"  {name}: {pytorch_model.state_dict()[name].shape}")
    
    # Check if there are any parameters we haven't mapped
    print(f"\nAll parameters in PyTorch model:")
    total_params = 0
    for name, param in pytorch_model.named_parameters():
        if total_params < 10:  # Show first 10
            print(f"  {name}: {param.shape}")
        total_params += 1
    
    print(f"Total parameters: {total_params}")

def test_esm_embedding_flow():
    """Test the exact ESM embedding flow."""
    print("\n🔍 Testing ESM Embedding Flow")
    print("=" * 50)
    
    # Load PyTorch model
    pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    # Test tokens
    test_tokens = torch.tensor([[0, 4, 5, 6, 2]])  # CLS, A, C, D, EOS
    
    print(f"Test tokens: {test_tokens}")
    print(f"Token meanings: {[alphabet.get_tok(t.item()) for t in test_tokens.squeeze()]}")
    
    # Step by step through PyTorch
    with torch.no_grad():
        # Step 1: Token embedding
        embeddings = pytorch_model.embed_tokens(test_tokens)
        print(f"\n1. Token embeddings:")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Range: {embeddings.min():.4f} to {embeddings.max():.4f}")
        print(f"   Sample: {embeddings[0, 0, :3]}")
        
        # Check if there's any scaling
        embed_scale = getattr(pytorch_model, 'embed_scale', 1.0)
        print(f"   Embed scale: {embed_scale}")
        
        # Step 2: Layer norm after embeddings
        if hasattr(pytorch_model, 'emb_layer_norm_after'):
            normed_embeddings = pytorch_model.emb_layer_norm_after(embeddings)
            print(f"\n2. After layer norm:")
            print(f"   Shape: {normed_embeddings.shape}")
            print(f"   Range: {normed_embeddings.min():.4f} to {normed_embeddings.max():.4f}")
            print(f"   Sample: {normed_embeddings[0, 0, :3]}")
            
            # Check layer norm parameters
            ln = pytorch_model.emb_layer_norm_after
            print(f"   LayerNorm weight sample: {ln.weight[:3]}")
            print(f"   LayerNorm bias sample: {ln.bias[:3]}")
            print(f"   LayerNorm eps: {ln.eps}")

def main():
    """Debug layer normalization."""
    try:
        test_layernorm_directly()
        test_position_embeddings()
        test_esm_embedding_flow()
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()