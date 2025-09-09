#!/usr/bin/env python3

"""
Debug the attention mechanism step by step to achieve perfect numerical accuracy.
"""

import sys
import os
import torch
import esm
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
import mlx.nn as nn
from esm_mlx import ESM2MLX, ESM2Config
from esm_mlx.convert_weights import extract_esm2_config_from_pytorch


def test_attention_projections():
    """Test Q, K, V projections individually."""
    print("🔍 Testing Attention Projections")
    print("=" * 50)
    
    # Load PyTorch model
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    pt_layer = pytorch_model.layers[0]
    pt_attn = pt_layer.self_attn
    
    # Create MLX attention
    config = extract_esm2_config_from_pytorch(pytorch_model)
    from esm_mlx.esm2_mlx import ESM2AttentionLayer
    mlx_attn = ESM2AttentionLayer(config)
    
    # Transfer weights
    pytorch_state = pytorch_model.state_dict()
    layer_idx = 0
    
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        pt_weight = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.weight"]
        pt_bias = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.bias"]
        
        mlx_proj = getattr(mlx_attn, proj_name)
        mlx_proj.weight = mx.array(pt_weight.detach().cpu().numpy())
        mlx_proj.bias = mx.array(pt_bias.detach().cpu().numpy())
    
    # Test input - use normalized embeddings
    with torch.no_grad():
        test_tokens = torch.tensor([[0, 4, 5, 6, 2]])
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
        pt_normed_embeddings = pytorch_model.emb_layer_norm_after(pt_embeddings)
    
    mlx_input = mx.array(pt_normed_embeddings.detach().cpu().numpy())
    
    print(f"Input shape: {pt_normed_embeddings.shape}")
    
    # Test each projection
    projections = ['q_proj', 'k_proj', 'v_proj']
    
    for proj_name in projections:
        print(f"\n  Testing {proj_name}:")
        
        # PyTorch projection
        pt_proj = getattr(pt_attn, proj_name)
        with torch.no_grad():
            pt_output = pt_proj(pt_normed_embeddings)
        
        # MLX projection
        mlx_proj = getattr(mlx_attn, proj_name)
        mlx_output = mlx_proj(mlx_input)
        
        # Compare
        pt_np = pt_output.detach().cpu().numpy()
        mlx_np = np.array(mlx_output)
        
        max_diff = np.max(np.abs(pt_np - mlx_np))
        cosine_sim = np.dot(pt_np.flatten(), mlx_np.flatten()) / (
            np.linalg.norm(pt_np.flatten()) * np.linalg.norm(mlx_np.flatten())
        )
        
        print(f"    Shape: PT {pt_output.shape} vs MLX {mlx_output.shape}")
        print(f"    Max diff: {max_diff:.8f}")
        print(f"    Cosine sim: {cosine_sim:.8f}")
        
        if max_diff < 1e-6:
            print(f"    ✅ {proj_name} perfect!")
        else:
            print(f"    ❌ {proj_name} mismatch")
    
    return mlx_attn, mlx_input, pt_normed_embeddings


def test_attention_reshape_and_scores():
    """Test attention score computation step by step."""
    print("\n🔍 Testing Attention Reshape and Scores")
    print("=" * 50)
    
    mlx_attn, mlx_input, pt_input = test_attention_projections()
    
    # Load PyTorch model for comparison
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    pt_attn = pytorch_model.layers[0].self_attn
    
    # Get Q, K, V from both models
    with torch.no_grad():
        pt_q = pt_attn.q_proj(pt_input)
        pt_k = pt_attn.k_proj(pt_input)
        pt_v = pt_attn.v_proj(pt_input)
    
    mlx_q = mlx_attn.q_proj(mlx_input)
    mlx_k = mlx_attn.k_proj(mlx_input)
    mlx_v = mlx_attn.v_proj(mlx_input)
    
    print(f"\nQ, K, V shapes:")
    print(f"  PyTorch Q: {pt_q.shape}")
    print(f"  MLX Q: {mlx_q.shape}")
    
    # Now test reshaping for multi-head attention
    batch_size, seq_len, hidden_size = pt_input.shape
    num_heads = mlx_attn.num_heads
    head_dim = mlx_attn.head_dim
    
    print(f"\nReshaping parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Seq len: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Expected: {num_heads * head_dim} == {hidden_size}")
    
    # PyTorch typically reshapes to (batch, seq, num_heads, head_dim) then transposes
    # Let's see what the PyTorch ESM attention actually does by examining the source
    
    # For now, let's check our reshaping logic
    mlx_q_reshaped = mlx_q.reshape(batch_size, seq_len, num_heads, head_dim)
    mlx_q_transposed = mlx_q_reshaped.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
    
    print(f"\nMLX reshaping:")
    print(f"  Original Q: {mlx_q.shape}")
    print(f"  Reshaped Q: {mlx_q_reshaped.shape}")
    print(f"  Transposed Q: {mlx_q_transposed.shape}")
    
    # Test if we can compute attention scores
    mlx_k_reshaped = mlx_k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Compute attention scores: Q @ K^T
    scores = mx.matmul(mlx_q_transposed, mlx_k_reshaped.transpose(0, 1, 3, 2))
    scale = 1.0 / np.sqrt(head_dim)
    scaled_scores = scores * scale
    
    print(f"\nAttention scores:")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Scale factor: {scale}")
    print(f"  Scaled scores range: {float(mx.min(scaled_scores)):.4f} to {float(mx.max(scaled_scores)):.4f}")
    
    return mlx_attn, mlx_input, pt_input


def compare_with_pytorch_attention():
    """Compare with actual PyTorch attention computation."""
    print("\n🔍 Comparing with PyTorch Attention")
    print("=" * 50)
    
    # Load PyTorch model
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    pt_attn = pytorch_model.layers[0].self_attn
    
    # Create test input
    with torch.no_grad():
        test_tokens = torch.tensor([[0, 4, 5, 6, 2]])
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
        pt_normed_embeddings = pytorch_model.emb_layer_norm_after(pt_embeddings)
    
    print(f"Input shape: {pt_normed_embeddings.shape}")
    
    # PyTorch attention forward
    with torch.no_grad():
        # Note: PyTorch ESM attention might expect different tensor format
        # Let's check what the attention function signature is
        
        print(f"\nPyTorch attention parameters:")
        print(f"  embed_dim: {pt_attn.embed_dim}")
        print(f"  num_heads: {pt_attn.num_heads}")
        print(f"  head_dim: {pt_attn.head_dim if hasattr(pt_attn, 'head_dim') else 'N/A'}")
        print(f"  dropout: {pt_attn.dropout}")
        
        # Try to call the attention
        try:
            pt_attn_output, pt_attn_weights = pt_attn(
                query=pt_normed_embeddings,
                key=pt_normed_embeddings,
                value=pt_normed_embeddings,
                need_weights=True
            )
            
            print(f"\nPyTorch attention outputs:")
            print(f"  Output shape: {pt_attn_output.shape}")
            print(f"  Weights shape: {pt_attn_weights.shape if pt_attn_weights is not None else None}")
            print(f"  Output range: {pt_attn_output.min():.4f} to {pt_attn_output.max():.4f}")
            
        except Exception as e:
            print(f"  ❌ PyTorch attention call failed: {e}")
            
            # Try different format - maybe it expects (seq, batch, dim)
            pt_input_transposed = pt_normed_embeddings.transpose(0, 1)
            print(f"  Trying transposed input: {pt_input_transposed.shape}")
            
            try:
                pt_attn_output, pt_attn_weights = pt_attn(
                    query=pt_input_transposed,
                    key=pt_input_transposed,
                    value=pt_input_transposed,
                    need_weights=True
                )
                
                print(f"  ✅ Success with transposed input!")
                print(f"  Output shape: {pt_attn_output.shape}")
                print(f"  Weights shape: {pt_attn_weights.shape if pt_attn_weights is not None else None}")
                
                # Transpose back
                pt_attn_output = pt_attn_output.transpose(0, 1)
                print(f"  Final output shape: {pt_attn_output.shape}")
                
            except Exception as e2:
                print(f"  ❌ Still failed: {e2}")
                return None
    
    # Now test our MLX attention
    config = extract_esm2_config_from_pytorch(pytorch_model)
    from esm_mlx.esm2_mlx import ESM2AttentionLayer
    mlx_attn = ESM2AttentionLayer(config)
    
    # Transfer weights
    pytorch_state = pytorch_model.state_dict()
    layer_idx = 0
    
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        pt_weight = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.weight"]
        pt_bias = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.bias"]
        
        mlx_proj = getattr(mlx_attn, proj_name)
        mlx_proj.weight = mx.array(pt_weight.detach().cpu().numpy())
        mlx_proj.bias = mx.array(pt_bias.detach().cpu().numpy())
    
    # Disable dropout
    mlx_attn.dropout = lambda x: x
    
    # MLX attention forward
    mlx_input = mx.array(pt_normed_embeddings.detach().cpu().numpy())
    mlx_attn_output, mlx_attn_weights = mlx_attn(mlx_input, output_attentions=True)
    
    print(f"\nMLX attention outputs:")
    print(f"  Output shape: {mlx_attn_output.shape}")
    print(f"  Weights shape: {mlx_attn_weights.shape if mlx_attn_weights is not None else None}")
    print(f"  Output range: {float(mx.min(mlx_attn_output)):.4f} to {float(mx.max(mlx_attn_output)):.4f}")
    
    # Compare outputs
    if pt_attn_output is not None:
        pt_np = pt_attn_output.detach().cpu().numpy()
        mlx_np = np.array(mlx_attn_output)
        
        max_diff = np.max(np.abs(pt_np - mlx_np))
        cosine_sim = np.dot(pt_np.flatten(), mlx_np.flatten()) / (
            np.linalg.norm(pt_np.flatten()) * np.linalg.norm(mlx_np.flatten())
        )
        
        print(f"\nAttention comparison:")
        print(f"  Max diff: {max_diff:.8f}")
        print(f"  Cosine similarity: {cosine_sim:.8f}")
        
        if cosine_sim > 0.99:
            print("  ✅ Attention outputs match!")
            return True
        else:
            print("  ❌ Attention outputs don't match")
            
            # Debug specific differences
            print(f"\n  Debugging differences:")
            print(f"    PyTorch first few values: {pt_np.flatten()[:5]}")
            print(f"    MLX first few values: {mlx_np.flatten()[:5]}")
            
            return False
    
    return False


def main():
    """Debug attention mechanism systematically."""
    print("🔬 Debugging Attention Mechanism")
    print("=" * 60)
    
    try:
        # Test projections first
        test_attention_projections()
        
        # Test reshaping and score computation
        test_attention_reshape_and_scores()
        
        # Compare with actual PyTorch attention
        success = compare_with_pytorch_attention()
        
        if success:
            print(f"\n🎉 Attention mechanism debugging complete!")
            print(f"Ready to test full model numerical accuracy.")
        else:
            print(f"\n🔍 More debugging needed for attention mechanism.")
            print(f"Focus areas:")
            print(f"1. Tensor format differences (batch_first vs seq_first)")
            print(f"2. Attention mask handling")
            print(f"3. Dropout behavior")
            print(f"4. Head dimension calculations")
        
        return success
        
    except Exception as e:
        print(f"❌ Attention debugging failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)