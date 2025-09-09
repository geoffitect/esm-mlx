#!/usr/bin/env python3

"""
Test each component individually to isolate the numerical accuracy issues.
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


def test_embeddings_only():
    """Test embeddings in isolation."""
    print("🧪 Testing Embeddings Only")
    print("=" * 40)
    
    # Load PyTorch model
    pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    # Create MLX model
    config = extract_esm2_config_from_pytorch(pytorch_model)
    mlx_model = ESM2MLX(config)
    
    # Transfer embedding weights
    pt_embed_weight = pytorch_model.embed_tokens.weight
    mlx_model.embeddings.word_embeddings.weight = mx.array(pt_embed_weight.detach().cpu().numpy())
    
    # Test tokens
    test_tokens = torch.tensor([[0, 4, 5, 6, 2]])  # CLS, A, C, D, EOS
    mlx_tokens = mx.array(test_tokens.numpy())
    
    # PyTorch embedding
    with torch.no_grad():
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
    
    # MLX embedding
    mlx_embeddings = mlx_model.embeddings.word_embeddings(mlx_tokens)
    
    # Compare
    pt_np = pt_embeddings.detach().cpu().numpy()
    mlx_np = np.array(mlx_embeddings)
    
    max_diff = np.max(np.abs(pt_np - mlx_np))
    cosine_sim = np.dot(pt_np.flatten(), mlx_np.flatten()) / (
        np.linalg.norm(pt_np.flatten()) * np.linalg.norm(mlx_np.flatten())
    )
    
    print(f"  Embedding test:")
    print(f"    PyTorch shape: {pt_embeddings.shape}")
    print(f"    MLX shape: {mlx_embeddings.shape}")
    print(f"    Max diff: {max_diff:.8f}")
    print(f"    Cosine similarity: {cosine_sim:.8f}")
    
    if max_diff < 1e-6:
        print("  ✅ Embeddings match perfectly!")
        return True
    else:
        print("  ❌ Embeddings don't match")
        return False


def test_embedding_layer_norm():
    """Test embedding layer normalization."""
    print("\n🧪 Testing Embedding Layer Norm")
    print("=" * 40)
    
    # Load PyTorch model
    pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    # Create MLX model
    config = extract_esm2_config_from_pytorch(pytorch_model)
    mlx_model = ESM2MLX(config)
    
    # Transfer weights
    pt_embed_weight = pytorch_model.embed_tokens.weight
    pt_ln_weight = pytorch_model.emb_layer_norm_after.weight
    pt_ln_bias = pytorch_model.emb_layer_norm_after.bias
    
    mlx_model.embeddings.word_embeddings.weight = mx.array(pt_embed_weight.detach().cpu().numpy())
    mlx_model.embeddings.layer_norm.weight = mx.array(pt_ln_weight.detach().cpu().numpy())
    mlx_model.embeddings.layer_norm.bias = mx.array(pt_ln_bias.detach().cpu().numpy())
    
    # Test tokens
    test_tokens = torch.tensor([[0, 4, 5, 6, 2]])  # CLS, A, C, D, EOS
    mlx_tokens = mx.array(test_tokens.numpy())
    
    # PyTorch forward
    with torch.no_grad():
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
        pt_normed = pytorch_model.emb_layer_norm_after(pt_embeddings)
    
    # MLX forward - disable dropout for testing
    mlx_model.embeddings.dropout = lambda x: x  # Disable dropout
    mlx_normed = mlx_model.embeddings(mlx_tokens)
    
    # Compare
    pt_np = pt_normed.detach().cpu().numpy()
    mlx_np = np.array(mlx_normed)
    
    max_diff = np.max(np.abs(pt_np - mlx_np))
    cosine_sim = np.dot(pt_np.flatten(), mlx_np.flatten()) / (
        np.linalg.norm(pt_np.flatten()) * np.linalg.norm(mlx_np.flatten())
    )
    
    print(f"  Embedding + LayerNorm test:")
    print(f"    PyTorch shape: {pt_normed.shape}")
    print(f"    MLX shape: {mlx_normed.shape}")
    print(f"    Max diff: {max_diff:.8f}")
    print(f"    Cosine similarity: {cosine_sim:.8f}")
    
    if max_diff < 1e-5:
        print("  ✅ Embedding + LayerNorm match!")
        return True
    else:
        print("  ❌ Embedding + LayerNorm don't match")
        return False


def test_single_attention_layer():
    """Test a single attention layer."""
    print("\n🧪 Testing Single Attention Layer")
    print("=" * 40)
    
    # Load PyTorch model
    pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    # Get first layer
    pt_layer = pytorch_model.layers[0]
    
    # Create MLX layer with same config
    config = extract_esm2_config_from_pytorch(pytorch_model)
    from esm_mlx.esm2_mlx import ESM2TransformerLayer
    mlx_layer = ESM2TransformerLayer(config)
    
    # Transfer ALL weights for this layer
    layer_idx = 0
    pytorch_state = pytorch_model.state_dict()
    
    # Attention weights
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        pt_weight = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.weight"]
        pt_bias = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.bias"]
        
        mlx_proj = getattr(mlx_layer.attention, proj_name)
        mlx_proj.weight = mx.array(pt_weight.detach().cpu().numpy())
        mlx_proj.bias = mx.array(pt_bias.detach().cpu().numpy())
    
    # Layer norms
    mlx_layer.self_attn_layer_norm.weight = mx.array(
        pytorch_state[f"layers.{layer_idx}.self_attn_layer_norm.weight"].detach().cpu().numpy()
    )
    mlx_layer.self_attn_layer_norm.bias = mx.array(
        pytorch_state[f"layers.{layer_idx}.self_attn_layer_norm.bias"].detach().cpu().numpy()
    )
    mlx_layer.final_layer_norm.weight = mx.array(
        pytorch_state[f"layers.{layer_idx}.final_layer_norm.weight"].detach().cpu().numpy()
    )
    mlx_layer.final_layer_norm.bias = mx.array(
        pytorch_state[f"layers.{layer_idx}.final_layer_norm.bias"].detach().cpu().numpy()
    )
    
    # FC layers
    mlx_layer.fc1.weight = mx.array(pytorch_state[f"layers.{layer_idx}.fc1.weight"].detach().cpu().numpy())
    mlx_layer.fc1.bias = mx.array(pytorch_state[f"layers.{layer_idx}.fc1.bias"].detach().cpu().numpy())
    mlx_layer.fc2.weight = mx.array(pytorch_state[f"layers.{layer_idx}.fc2.weight"].detach().cpu().numpy())
    mlx_layer.fc2.bias = mx.array(pytorch_state[f"layers.{layer_idx}.fc2.bias"].detach().cpu().numpy())
    
    # Test input - use same embeddings from PyTorch
    test_tokens = torch.tensor([[0, 4, 5, 6, 2]])
    with torch.no_grad():
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
        pt_normed_embeddings = pytorch_model.emb_layer_norm_after(pt_embeddings)
    
    mlx_input = mx.array(pt_normed_embeddings.detach().cpu().numpy())
    
    # PyTorch layer forward
    with torch.no_grad():
        pt_output, pt_attn = pt_layer(pt_normed_embeddings)
    
    # MLX layer forward - disable dropout for testing
    mlx_layer.dropout = lambda x: x  # Disable dropout
    mlx_output, mlx_attn = mlx_layer(mlx_input, output_attentions=True)
    
    # Compare outputs
    pt_np = pt_output.detach().cpu().numpy()
    mlx_np = np.array(mlx_output)
    
    max_diff = np.max(np.abs(pt_np - mlx_np))
    cosine_sim = np.dot(pt_np.flatten(), mlx_np.flatten()) / (
        np.linalg.norm(pt_np.flatten()) * np.linalg.norm(mlx_np.flatten())
    )
    
    print(f"  Single layer test:")
    print(f"    PyTorch output: {pt_output.shape}")
    print(f"    MLX output: {mlx_output.shape}")
    print(f"    Max diff: {max_diff:.8f}")
    print(f"    Cosine similarity: {cosine_sim:.8f}")
    
    if cosine_sim > 0.99:
        print("  ✅ Single layer matches!")
        return True
    else:
        print("  ❌ Single layer doesn't match")
        return False


def test_gelu_activation():
    """Test GELU activation function."""
    print("\n🧪 Testing GELU Activation")
    print("=" * 40)
    
    # Test values
    test_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # PyTorch GELU
    pt_input = torch.tensor(test_input, dtype=torch.float32)
    pt_gelu = torch.nn.functional.gelu(pt_input)
    pt_result = pt_gelu.numpy()
    
    # MLX GELU (our updated implementation)
    mlx_input = mx.array(test_input)
    mlx_gelu = mlx_input * 0.5 * (1.0 + mx.erf(mlx_input / mx.sqrt(2.0)))
    mlx_result = np.array(mlx_gelu)
    
    max_diff = np.max(np.abs(pt_result - mlx_result))
    
    print(f"  GELU test:")
    print(f"    Input: {test_input}")
    print(f"    PyTorch: {pt_result}")
    print(f"    MLX: {mlx_result}")
    print(f"    Max diff: {max_diff:.8f}")
    
    if max_diff < 1e-6:
        print("  ✅ GELU matches!")
        return True
    else:
        print("  ❌ GELU doesn't match")
        return False


def main():
    """Run component-by-component tests."""
    print("🔬 Component-by-Component Testing")
    print("=" * 60)
    
    results = []
    
    try:
        # Test each component
        results.append(("Embeddings", test_embeddings_only()))
        results.append(("Embedding LayerNorm", test_embedding_layer_norm()))
        results.append(("GELU Activation", test_gelu_activation()))
        results.append(("Single Layer", test_single_attention_layer()))
        
        # Summary
        print(f"\n🎯 Component Test Results:")
        print("=" * 30)
        
        for name, passed in results:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
        
        all_passed = all(result[1] for result in results)
        if all_passed:
            print(f"\n🎉 All components pass! Architecture should be correct.")
        else:
            print(f"\n🔍 Some components fail. Focus on fixing these first.")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)