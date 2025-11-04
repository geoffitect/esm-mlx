#!/usr/bin/env python3

"""
Comprehensive numerical accuracy testing framework.
Designed to achieve 1:1 parity with PyTorch ESM2.
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


def test_component_accuracy(component_name: str, pt_output: torch.Tensor, mlx_output: mx.array, tolerance: float = 1e-5):
    """Test accuracy of a specific component."""
    pt_np = pt_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    # Shape check
    if pt_np.shape != mlx_np.shape:
        print(f"  {component_name}: ❌ FAIL - Shape mismatch: PyTorch {pt_np.shape} vs MLX {mlx_np.shape}")
        return False
    
    # Numerical comparisons
    max_diff = np.max(np.abs(pt_np - mlx_np))
    
    # Cosine similarity
    pt_flat = pt_np.flatten()
    mlx_flat = mlx_np.flatten()
    
    if np.linalg.norm(pt_flat) == 0 or np.linalg.norm(mlx_flat) == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = np.dot(pt_flat, mlx_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(mlx_flat))
    
    # Pass/fail determination - realistic for float32 arithmetic
    passed = (max_diff < tolerance and cosine_sim > 0.99999)
    
    print(f"  {component_name}:")
    print(f"    Max diff: {max_diff:.2e}")
    print(f"    Cosine sim: {cosine_sim:.8f}")
    print(f"    Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if not passed:
        print(f"    First 5 PyTorch values: {pt_flat[:5]}")
        print(f"    First 5 MLX values:     {mlx_flat[:5]}")
    
    return passed


def main():
    """Main accuracy testing function."""
    print("🧪 Comprehensive Numerical Accuracy Test Suite")
    print("=" * 60)
    
    try:
        # Load PyTorch model
        print("Loading PyTorch ESM2-8M model...")
        pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        pytorch_model.eval()
        
        # Disable dropout completely
        def disable_dropout_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Dropout):
                    child.p = 0.0
                    child.eval()
                else:
                    disable_dropout_recursive(child)
        disable_dropout_recursive(pytorch_model)
        
        # Create MLX model
        print("Creating MLX model...")
        config = extract_esm2_config_from_pytorch(pytorch_model)
        mlx_model = ESM2MLX(config)
        
        # Transfer weights
        print("Transferring weights...")
        pytorch_state = pytorch_model.state_dict()
        
        # Transfer embeddings
        mlx_model.embeddings.word_embeddings.weight = mx.array(pytorch_state["embed_tokens.weight"].detach().cpu().numpy())
        mlx_model.embeddings.layer_norm.weight = mx.array(pytorch_state["emb_layer_norm_after.weight"].detach().cpu().numpy())
        mlx_model.embeddings.layer_norm.bias = mx.array(pytorch_state["emb_layer_norm_after.bias"].detach().cpu().numpy())
        
        # Transfer transformer layers
        for layer_idx in range(config.num_hidden_layers):
            mlx_layer = mlx_model.encoder.layers[layer_idx]
            
            # Attention weights
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                pt_weight = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.weight"]
                pt_bias = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.bias"]
                
                mlx_proj = getattr(mlx_layer.attention, proj_name)
                mlx_proj.weight = mx.array(pt_weight.detach().cpu().numpy())
                mlx_proj.bias = mx.array(pt_bias.detach().cpu().numpy())
            
            # Layer norms
            mlx_layer.self_attn_layer_norm.weight = mx.array(pytorch_state[f"layers.{layer_idx}.self_attn_layer_norm.weight"].detach().cpu().numpy())
            mlx_layer.self_attn_layer_norm.bias = mx.array(pytorch_state[f"layers.{layer_idx}.self_attn_layer_norm.bias"].detach().cpu().numpy())
            mlx_layer.final_layer_norm.weight = mx.array(pytorch_state[f"layers.{layer_idx}.final_layer_norm.weight"].detach().cpu().numpy())
            mlx_layer.final_layer_norm.bias = mx.array(pytorch_state[f"layers.{layer_idx}.final_layer_norm.bias"].detach().cpu().numpy())
            
            # FC layers
            mlx_layer.fc1.weight = mx.array(pytorch_state[f"layers.{layer_idx}.fc1.weight"].detach().cpu().numpy())
            mlx_layer.fc1.bias = mx.array(pytorch_state[f"layers.{layer_idx}.fc1.bias"].detach().cpu().numpy())
            mlx_layer.fc2.weight = mx.array(pytorch_state[f"layers.{layer_idx}.fc2.weight"].detach().cpu().numpy())
            mlx_layer.fc2.bias = mx.array(pytorch_state[f"layers.{layer_idx}.fc2.bias"].detach().cpu().numpy())
        
        # Transfer output head
        mlx_model.lm_head.weight = mx.array(pytorch_state["lm_head.weight"].detach().cpu().numpy())
        mlx_model.lm_head.bias = mx.array(pytorch_state["lm_head.bias"].detach().cpu().numpy())
        
        print("✅ Weight transfer complete")
        
        # Test embeddings
        print(f"\n🔤 Testing Embeddings")
        print("=" * 30)
        
        test_tokens = torch.tensor([[0, 4, 5, 6, 2]])
        mlx_tokens = mx.array(test_tokens.numpy())
        
        with torch.no_grad():
            pt_embeddings = pytorch_model.embed_tokens(test_tokens)
            pt_normed = pytorch_model.emb_layer_norm_after(pt_embeddings)
        
        mlx_embeddings_raw = mlx_model.embeddings.word_embeddings(mlx_tokens)
        mlx_embeddings_scaled = mlx_embeddings_raw * mlx_model.embeddings.embed_scale
        mlx_normed = mlx_model.embeddings.layer_norm(mlx_embeddings_scaled)
        
        embed_passed = test_component_accuracy("Raw embeddings", pt_embeddings, mlx_embeddings_raw)
        norm_passed = test_component_accuracy("Normed embeddings", pt_normed, mlx_normed)
        
        if not embed_passed or not norm_passed:
            print("❌ Embedding tests failed!")
            return False
        
        # Test attention step by step
        print(f"\n🧠 Testing Attention (Step by Step)")
        print("=" * 30)
        
        pt_attn = pytorch_model.layers[0].self_attn
        mlx_attn = mlx_model.encoder.layers[0].attention
        
        batch_size, seq_len, hidden_size = pt_normed.shape
        num_heads = pt_attn.num_heads
        head_dim = hidden_size // num_heads
        
        # Step 1: Q, K, V projections
        with torch.no_grad():
            pt_q = pt_attn.q_proj(pt_normed)
            pt_k = pt_attn.k_proj(pt_normed)
            pt_v = pt_attn.v_proj(pt_normed)
        
        mlx_q = mlx_attn.q_proj(mlx_normed)
        mlx_k = mlx_attn.k_proj(mlx_normed)
        mlx_v = mlx_attn.v_proj(mlx_normed)
        
        q_passed = test_component_accuracy("Q projection", pt_q, mlx_q)
        k_passed = test_component_accuracy("K projection", pt_k, mlx_k)
        v_passed = test_component_accuracy("V projection", pt_v, mlx_v)
        
        if not all([q_passed, k_passed, v_passed]):
            print("❌ Projection tests failed!")
            return False
        
        # Step 2: Test full attention call
        print(f"\nStep 2: Full Attention Call")
        
        pt_input_seq_first = pt_normed.transpose(0, 1)
        with torch.no_grad():
            pt_full_output, _ = pt_attn(pt_input_seq_first, pt_input_seq_first, pt_input_seq_first)
            pt_full_output = pt_full_output.transpose(0, 1)  # Back to batch_first
        
        mlx_full_output, _ = mlx_attn(mlx_normed)
        
        attn_passed = test_component_accuracy("Full attention", pt_full_output, mlx_full_output)
        
        if not attn_passed:
            print("❌ Attention test failed!")
            # Let's debug this more carefully
            print("\nDebugging attention...")
            
            # Manual computation
            scale = 1.0 / np.sqrt(head_dim)
            
            # PyTorch manual
            pt_q_seq = pt_q.transpose(0, 1)
            pt_k_seq = pt_k.transpose(0, 1)
            pt_v_seq = pt_v.transpose(0, 1)
            
            pt_q_multi = pt_q_seq.contiguous().view(seq_len, batch_size * num_heads, head_dim)
            pt_k_multi = pt_k_seq.contiguous().view(seq_len, batch_size * num_heads, head_dim)
            pt_v_multi = pt_v_seq.contiguous().view(seq_len, batch_size * num_heads, head_dim)
            
            with torch.no_grad():
                pt_scores = torch.matmul(pt_q_multi.transpose(0, 1), pt_k_multi.permute(1, 2, 0)) * scale
                pt_probs = torch.softmax(pt_scores, dim=-1)
                pt_v_trans = pt_v_multi.transpose(0, 1)
                pt_output = torch.matmul(pt_probs, pt_v_trans)
            
            # MLX manual
            mlx_q_seq = mlx_q.transpose(1, 0, 2)
            mlx_k_seq = mlx_k.transpose(1, 0, 2)
            mlx_v_seq = mlx_v.transpose(1, 0, 2)
            
            mlx_q_multi = mlx_q_seq.reshape(seq_len, batch_size * num_heads, head_dim)
            mlx_k_multi = mlx_k_seq.reshape(seq_len, batch_size * num_heads, head_dim)
            mlx_v_multi = mlx_v_seq.reshape(seq_len, batch_size * num_heads, head_dim)
            
            mlx_scores = mx.matmul(mlx_q_multi.transpose(1, 0, 2), mlx_k_multi.transpose(1, 2, 0)) * scale
            mlx_probs = mx.softmax(mlx_scores, axis=-1)
            mlx_v_trans = mlx_v_multi.transpose(1, 0, 2)
            mlx_output = mx.matmul(mlx_probs, mlx_v_trans)
            
            test_component_accuracy("Manual attention scores", pt_scores, mlx_scores)
            test_component_accuracy("Manual attention probs", pt_probs, mlx_probs)
            test_component_accuracy("Manual attention output", pt_output, mlx_output)
            
            return False
        
        # Test full model
        print(f"\n🏗️ Testing Full Model")
        print("=" * 30)
        
        with torch.no_grad():
            pt_outputs = pytorch_model(test_tokens, repr_layers=[pytorch_model.num_layers])
            pt_last_hidden = pt_outputs["representations"][pytorch_model.num_layers]
        
        mlx_outputs = mlx_model(mlx_tokens)
        mlx_last_hidden = mlx_outputs["last_hidden_state"]
        
        full_passed = test_component_accuracy("Full model", pt_last_hidden, mlx_last_hidden)
        
        if full_passed:
            print(f"\n🎉 ALL TESTS PASSED! Perfect 1:1 accuracy achieved!")
            return True
        else:
            print(f"\n❌ Full model test failed!")
            return False
        
    except Exception as e:
        print(f"❌ Accuracy testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
