#!/usr/bin/env python3

"""
Test exact PyTorch matching.
"""

import sys
import os
import torch
import esm
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from esm_mlx import ESM2MLX, ESM2Config
from esm_mlx.convert_weights import extract_esm2_config_from_pytorch


def compare_tensors(name, pt_tensor, mlx_tensor, tolerance=1e-5):
    """Compare tensors."""
    if hasattr(pt_tensor, 'detach'):
        pt_np = pt_tensor.detach().cpu().numpy()
    else:
        pt_np = np.array(pt_tensor)
    mlx_np = np.array(mlx_tensor)
    
    max_diff = np.max(np.abs(pt_np - mlx_np))
    cosine_sim = np.dot(pt_np.flatten(), mlx_np.flatten()) / (np.linalg.norm(pt_np.flatten()) * np.linalg.norm(mlx_np.flatten()))
    passed = max_diff < tolerance and cosine_sim > 0.99999
    
    print(f"{name}: max_diff={max_diff:.2e}, cosine={cosine_sim:.8f}, {'✅' if passed else '❌'}")
    if not passed:
        print(f"  First 5 PyTorch: {pt_np.flatten()[:5]}")
        print(f"  First 5 MLX:     {mlx_np.flatten()[:5]}")
    
    return passed


def main():
    """Test exact PyTorch matching."""
    print("🎯 Testing Exact PyTorch Matching")
    print("=" * 50)
    
    # Setup models
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    # Disable dropout
    def disable_dropout_recursive(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = 0.0
                child.eval()
            else:
                disable_dropout_recursive(child)
    disable_dropout_recursive(pytorch_model)
    
    config = extract_esm2_config_from_pytorch(pytorch_model)
    mlx_model = ESM2MLX(config)
    
    # Transfer weights
    pytorch_state = pytorch_model.state_dict()
    
    # Transfer embeddings
    mlx_model.embeddings.word_embeddings.weight = mx.array(pytorch_state["embed_tokens.weight"].detach().cpu().numpy())
    mlx_model.embeddings.layer_norm.weight = mx.array(pytorch_state["emb_layer_norm_after.weight"].detach().cpu().numpy())
    mlx_model.embeddings.layer_norm.bias = mx.array(pytorch_state["emb_layer_norm_after.bias"].detach().cpu().numpy())
    
    # Transfer first layer attention
    layer_idx = 0
    mlx_layer = mlx_model.encoder.layers[layer_idx]
    
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        pt_weight = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.weight"]
        pt_bias = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.bias"]
        
        mlx_proj = getattr(mlx_layer.attention, proj_name)
        mlx_proj.weight = mx.array(pt_weight.detach().cpu().numpy())
        mlx_proj.bias = mx.array(pt_bias.detach().cpu().numpy())
    
    # Test input
    test_tokens = torch.tensor([[0, 4, 5, 6, 2]])
    mlx_tokens = mx.array(test_tokens.numpy())
    
    with torch.no_grad():
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
        pt_normed = pytorch_model.emb_layer_norm_after(pt_embeddings)
    
    mlx_embeddings_raw = mlx_model.embeddings.word_embeddings(mlx_tokens)
    mlx_embeddings_scaled = mlx_embeddings_raw * mlx_model.embeddings.embed_scale
    mlx_normed = mlx_model.embeddings.layer_norm(mlx_embeddings_scaled)
    
    print("Testing exact PyTorch seq_first attention...")
    
    # PyTorch reference (seq_first call)
    pt_attn = pytorch_model.layers[0].self_attn
    pt_input_seq_first = pt_normed.transpose(0, 1)  # (seq, batch, dim)
    
    with torch.no_grad():
        pt_ref_output, pt_ref_weights = pt_attn(pt_input_seq_first, pt_input_seq_first, pt_input_seq_first, need_weights=True)
        pt_ref_output_batch = pt_ref_output.transpose(0, 1)  # Back to batch_first
    
    print(f"PyTorch reference:")
    print(f"  Input shape: {pt_input_seq_first.shape}")
    print(f"  Output shape: {pt_ref_output.shape} -> {pt_ref_output_batch.shape}")
    print(f"  Output range: {pt_ref_output_batch.min():.4f} to {pt_ref_output_batch.max():.4f}")
    
    # Our MLX implementation
    mlx_attn = mlx_model.encoder.layers[0].attention
    mlx_output, mlx_weights = mlx_attn(mlx_normed, output_attentions=True)
    
    print(f"\nMLX implementation:")
    print(f"  Input shape: {mlx_normed.shape}")
    print(f"  Output shape: {mlx_output.shape}")
    print(f"  Output range: {float(mx.min(mlx_output)):.4f} to {float(mx.max(mlx_output)):.4f}")
    
    # Compare
    match = compare_tensors("Attention outputs", pt_ref_output_batch, mlx_output)
    
    if not match:
        print(f"\n❌ Still not matching. Let's debug the implementation step by step...")
        
        # Let's manually implement the EXACT PyTorch computation
        print(f"\nManual PyTorch-style computation in MLX:")
        
        # Convert to seq_first
        query_seq = mlx_normed.transpose(1, 0, 2)  # (seq, batch, dim)
        print(f"Query seq_first: {query_seq.shape}")
        
        # Project
        q = mlx_attn.q_proj(query_seq)
        k = mlx_attn.k_proj(query_seq) 
        v = mlx_attn.v_proj(query_seq)
        print(f"After projection: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # Scale Q
        q_scaled = q * mlx_attn.scale
        print(f"After scaling Q: {q_scaled.shape}")
        
        # Get dimensions
        seq_len, batch_size, embed_dim = q.shape
        num_heads = mlx_attn.num_heads
        head_dim = mlx_attn.head_dim
        
        # Reshape for multi-head
        q_reshaped = q_scaled.reshape(seq_len, batch_size * num_heads, head_dim)
        k_reshaped = k.reshape(seq_len, batch_size * num_heads, head_dim)
        v_reshaped = v.reshape(seq_len, batch_size * num_heads, head_dim)
        print(f"After reshape: q={q_reshaped.shape}, k={k_reshaped.shape}, v={v_reshaped.shape}")
        
        # Transpose for bmm
        q_t = q_reshaped.transpose(1, 0, 2)  # (batch*heads, seq, head_dim)
        k_t = k_reshaped.transpose(1, 0, 2)
        v_t = v_reshaped.transpose(1, 0, 2)
        print(f"After transpose: q={q_t.shape}, k={k_t.shape}, v={v_t.shape}")
        
        # Attention
        attn_weights = mx.matmul(q_t, k_t.transpose(0, 2, 1))
        attn_probs = mx.softmax(attn_weights, axis=-1)
        attn_out = mx.matmul(attn_probs, v_t)
        print(f"Attention output: {attn_out.shape}")
        
        # Reshape back
        attn_out_reshaped = attn_out.transpose(1, 0, 2).reshape(seq_len, batch_size, embed_dim)
        print(f"Reshaped: {attn_out_reshaped.shape}")
        
        # Output projection
        final_out = mlx_attn.out_proj(attn_out_reshaped)
        print(f"After out_proj: {final_out.shape}")
        
        # Convert to batch_first
        final_out_batch = final_out.transpose(1, 0, 2)
        print(f"Final batch_first: {final_out_batch.shape}")
        
        # Compare manual computation
        compare_tensors("Manual computation", pt_ref_output_batch, final_out_batch)
        
        # Let's also check if the projections match PyTorch exactly
        print(f"\nChecking projections on seq_first input:")
        
        with torch.no_grad():
            pt_q = pt_attn.q_proj(pt_input_seq_first)
            pt_k = pt_attn.k_proj(pt_input_seq_first)
            pt_v = pt_attn.v_proj(pt_input_seq_first)
        
        compare_tensors("Q projection (seq_first)", pt_q, q)
        compare_tensors("K projection (seq_first)", pt_k, k)
        compare_tensors("V projection (seq_first)", pt_v, v)
        
        # Check scaling
        pt_scaling = getattr(pt_attn, 'scaling', None)
        print(f"PyTorch scaling: {pt_scaling}")
        print(f"MLX scale: {mlx_attn.scale}")
        
        if pt_scaling:
            pt_q_scaled = pt_q * pt_scaling
            compare_tensors("Q after scaling", pt_q_scaled, q_scaled)
    
    return match


if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)