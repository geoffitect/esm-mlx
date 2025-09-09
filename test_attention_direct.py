#!/usr/bin/env python3

"""
Direct test of attention mechanism using the correct PyTorch seq_first format.
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
from esm_mlx.convert_weights import extract_esm2_config_from_pytorch


def test_attention_with_correct_format():
    """Test attention using the correct seq_first format for PyTorch."""
    print("🧪 Testing Attention with Correct Format")
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
    
    print(f"Original input shape: {pt_normed_embeddings.shape}")
    
    # Convert to seq_first format for PyTorch
    pt_input_seq_first = pt_normed_embeddings.transpose(0, 1)  # (seq, batch, dim)
    print(f"PyTorch seq_first input: {pt_input_seq_first.shape}")
    
    # PyTorch attention (seq_first)
    with torch.no_grad():
        pt_output, pt_weights = pt_attn(
            query=pt_input_seq_first,
            key=pt_input_seq_first,
            value=pt_input_seq_first,
            need_weights=True
        )
    
    # Convert PyTorch output back to batch_first
    pt_output_batch_first = pt_output.transpose(0, 1)
    
    print(f"\nPyTorch attention (seq_first):")
    print(f"  Output: {pt_output.shape} -> {pt_output_batch_first.shape}")
    print(f"  Weights: {pt_weights.shape}")
    print(f"  Output range: {pt_output_batch_first.min():.4f} to {pt_output_batch_first.max():.4f}")
    
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
    
    # Disable dropout
    mlx_attn.dropout = lambda x: x
    
    # MLX attention (batch_first input)
    mlx_input = mx.array(pt_normed_embeddings.detach().cpu().numpy())
    mlx_output, mlx_weights = mlx_attn(mlx_input, output_attentions=True)
    
    print(f"\nMLX attention (batch_first):")
    print(f"  Output: {mlx_output.shape}")
    print(f"  Weights: {mlx_weights.shape if mlx_weights is not None else None}")
    print(f"  Output range: {float(mx.min(mlx_output)):.4f} to {float(mx.max(mlx_output)):.4f}")
    
    # Compare outputs
    pt_np = pt_output_batch_first.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    max_diff = np.max(np.abs(pt_np - mlx_np))
    cosine_sim = np.dot(pt_np.flatten(), mlx_np.flatten()) / (
        np.linalg.norm(pt_np.flatten()) * np.linalg.norm(mlx_np.flatten())
    )
    
    print(f"\nComparison:")
    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Cosine similarity: {cosine_sim:.8f}")
    
    # Compare attention weights
    if pt_weights is not None and mlx_weights is not None:
        pt_weights_np = pt_weights.detach().cpu().numpy()
        mlx_weights_np = np.array(mlx_weights)
        
        print(f"\nAttention weights comparison:")
        print(f"  PyTorch weights: {pt_weights.shape}")
        print(f"  MLX weights: {mlx_weights.shape}")
        print(f"  PyTorch weights range: {pt_weights_np.min():.6f} to {pt_weights_np.max():.6f}")
        print(f"  MLX weights range: {mlx_weights_np.min():.6f} to {mlx_weights_np.max():.6f}")
        
        # Check if they're both attention matrices (sum to 1)
        pt_sums = pt_weights_np.sum(axis=-1)
        mlx_sums = mlx_weights_np.sum(axis=-1)
        print(f"  PyTorch row sums: {pt_sums.flatten()[:3]}...")
        print(f"  MLX row sums: {mlx_sums.flatten()[:3]}...")
    
    if cosine_sim > 0.99:
        print("  ✅ Attention outputs match!")
        return True
    else:
        print("  ❌ Attention outputs don't match")
        
        # Debug the first few values
        print(f"\n  Debugging:")
        print(f"    PyTorch first 5 values: {pt_np.flatten()[:5]}")
        print(f"    MLX first 5 values: {mlx_np.flatten()[:5]}")
        
        return False


def test_step_by_step_attention():
    """Test attention computation step by step."""
    print("\n🔍 Step-by-Step Attention Testing")
    print("=" * 50)
    
    # Load PyTorch model
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    pt_attn = pytorch_model.layers[0].self_attn
    
    # Create test input
    with torch.no_grad():
        test_tokens = torch.tensor([[0, 4, 5, 6, 2]])
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
        pt_normed_embeddings = pytorch_model.emb_layer_norm_after(pt_embeddings)
    
    # Get dimensions
    batch_size, seq_len, hidden_size = pt_normed_embeddings.shape
    num_heads = pt_attn.num_heads
    head_dim = hidden_size // num_heads
    
    print(f"Dimensions:")
    print(f"  Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_size}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")
    
    # Step 1: Get Q, K, V projections
    with torch.no_grad():
        pt_q = pt_attn.q_proj(pt_normed_embeddings)
        pt_k = pt_attn.k_proj(pt_normed_embeddings)
        pt_v = pt_attn.v_proj(pt_normed_embeddings)
    
    print(f"\nStep 1 - Projections:")
    print(f"  Q: {pt_q.shape}")
    print(f"  K: {pt_k.shape}")
    print(f"  V: {pt_v.shape}")
    
    # Now manually compute attention to understand PyTorch ESM's approach
    # Convert to seq_first format
    pt_q_seq = pt_q.transpose(0, 1)  # (seq, batch, dim)
    pt_k_seq = pt_k.transpose(0, 1)  # (seq, batch, dim)
    pt_v_seq = pt_v.transpose(0, 1)  # (seq, batch, dim)
    
    # Reshape for multi-head: (seq, batch, dim) -> (seq, batch*heads, head_dim)
    pt_q_multi = pt_q_seq.contiguous().view(seq_len, batch_size * num_heads, head_dim)
    pt_k_multi = pt_k_seq.contiguous().view(seq_len, batch_size * num_heads, head_dim)
    pt_v_multi = pt_v_seq.contiguous().view(seq_len, batch_size * num_heads, head_dim)
    
    print(f"\nStep 2 - Multi-head reshape:")
    print(f"  Q multi: {pt_q_multi.shape}")
    print(f"  K multi: {pt_k_multi.shape}")
    print(f"  V multi: {pt_v_multi.shape}")
    
    # Compute attention scores
    # PyTorch formula: q @ k^T / sqrt(head_dim)
    scale = 1.0 / np.sqrt(head_dim)
    
    # q: (seq, batch*heads, head_dim)
    # k: (seq, batch*heads, head_dim) -> (batch*heads, head_dim, seq)
    pt_scores = torch.matmul(pt_q_multi.transpose(0, 1), pt_k_multi.permute(1, 2, 0)) * scale
    
    print(f"\nStep 3 - Attention scores:")
    print(f"  Scores: {pt_scores.shape}")
    print(f"  Score range: {pt_scores.min():.4f} to {pt_scores.max():.4f}")
    
    # Apply softmax
    pt_attn_probs = torch.softmax(pt_scores, dim=-1)
    
    print(f"\nStep 4 - Attention probabilities:")
    print(f"  Probs: {pt_attn_probs.shape}")
    print(f"  Prob range: {pt_attn_probs.min():.6f} to {pt_attn_probs.max():.6f}")
    print(f"  Row sums: {pt_attn_probs.sum(dim=-1)[:3, 0]}")  # Should be ~1.0
    
    # Apply to values
    # attn_probs: (batch*heads, seq, seq)
    # v: (seq, batch*heads, head_dim) -> (batch*heads, seq, head_dim)
    pt_v_transposed = pt_v_multi.transpose(0, 1)
    pt_attn_output = torch.matmul(pt_attn_probs, pt_v_transposed)
    
    print(f"\nStep 5 - Apply attention to values:")
    print(f"  V transposed: {pt_v_transposed.shape}")
    print(f"  Attention output: {pt_attn_output.shape}")
    print(f"  Output range: {pt_attn_output.min():.4f} to {pt_attn_output.max():.4f}")
    
    # Reshape back
    pt_attn_output_reshaped = pt_attn_output.view(batch_size, num_heads, seq_len, head_dim)
    pt_attn_output_final = pt_attn_output_reshaped.transpose(2, 1).contiguous().view(batch_size, seq_len, hidden_size)
    
    print(f"\nStep 6 - Reshape back:")
    print(f"  Reshaped: {pt_attn_output_reshaped.shape}")
    print(f"  Final: {pt_attn_output_final.shape}")
    
    # Apply output projection
    with torch.no_grad():
        pt_final_output = pt_attn.out_proj(pt_attn_output_final)
    
    print(f"\nStep 7 - Output projection:")
    print(f"  Final output: {pt_final_output.shape}")
    print(f"  Final range: {pt_final_output.min():.4f} to {pt_final_output.max():.4f}")
    
    # Compare with actual PyTorch attention call
    pt_input_seq_first = pt_normed_embeddings.transpose(0, 1)
    with torch.no_grad():
        pt_ref_output, _ = pt_attn(pt_input_seq_first, pt_input_seq_first, pt_input_seq_first)
        pt_ref_output_batch = pt_ref_output.transpose(0, 1)
    
    print(f"\nComparison with reference:")
    print(f"  Manual output: {pt_final_output.shape}")
    print(f"  Reference output: {pt_ref_output_batch.shape}")
    
    diff = torch.abs(pt_final_output - pt_ref_output_batch).max()
    print(f"  Max difference: {diff:.8f}")
    
    if diff < 1e-5:
        print("  ✅ Manual computation matches PyTorch!")
    else:
        print("  ❌ Manual computation doesn't match PyTorch")
    
    return pt_final_output, pt_ref_output_batch


def main():
    """Main testing function."""
    print("🔬 Direct Attention Testing")
    print("=" * 60)
    
    try:
        # Test attention with correct format
        success1 = test_attention_with_correct_format()
        
        # Test step by step
        test_step_by_step_attention()
        
        if success1:
            print(f"\n🎉 Attention mechanism fixed!")
        else:
            print(f"\n🔍 More work needed on attention mechanism")
        
        return success1
        
    except Exception as e:
        print(f"❌ Direct attention testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)