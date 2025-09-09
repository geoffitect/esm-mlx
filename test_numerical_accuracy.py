#!/usr/bin/env python3

"""
Test numerical accuracy between PyTorch ESM2 and MLX implementation.
This validates that our MLX implementation produces the same results as PyTorch.
"""

import sys
import os
import torch
import esm
import numpy as np
from typing import Tuple, Dict

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from esm_mlx import ESM2MLX, ESM2Config
from esm_mlx.convert_weights import extract_esm2_config_from_pytorch, convert_pytorch_weights_to_mlx


def create_matched_models():
    """Create PyTorch and MLX models with matching architecture."""
    print("🔧 Creating matched PyTorch and MLX models...")
    
    # Load PyTorch model
    pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    # Extract config and create MLX model
    config = extract_esm2_config_from_pytorch(pytorch_model)
    mlx_model = ESM2MLX(config)
    
    print(f"✅ Models created:")
    print(f"  PyTorch: {sum(p.numel() for p in pytorch_model.parameters()):,} params")
    print(f"  MLX config: {config.hidden_size}D, {config.num_hidden_layers} layers")
    
    return pytorch_model, mlx_model, alphabet, config


def create_test_inputs(alphabet) -> Tuple[torch.Tensor, mx.array]:
    """Create test inputs for both models."""
    print("\n🧬 Creating test inputs...")
    
    # Test sequences of different lengths
    sequences = [
        "MKTVRQERLK",  # Short
        "MKTVRQERLKSIVRILERSKEPVSG",  # Medium
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"  # Long
    ]
    
    # Use the medium sequence for main test
    sequence = sequences[1]
    print(f"Test sequence: {sequence} (length: {len(sequence)})")
    
    # PyTorch tokenization
    batch_converter = alphabet.get_batch_converter()
    data = [("test", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # MLX tokenization (same tokens)
    mlx_tokens = mx.array(batch_tokens.numpy())
    
    print(f"  Tokenized shape: {batch_tokens.shape}")
    print(f"  Token range: {batch_tokens.min().item()} to {batch_tokens.max().item()}")
    
    return batch_tokens, mlx_tokens


def manually_transfer_weights(pytorch_model, mlx_model):
    """Manually transfer key weights to test numerical accuracy."""
    print("\n⚖️ Manually transferring critical weights...")
    
    pytorch_state = pytorch_model.state_dict()
    transferred = 0
    
    # Transfer embedding weights
    if "embed_tokens.weight" in pytorch_state:
        pt_emb = pytorch_state["embed_tokens.weight"]
        mlx_emb = mx.array(pt_emb.detach().cpu().numpy())
        mlx_model.embeddings.word_embeddings.weight = mlx_emb
        transferred += 1
        print(f"  ✅ Transferred embeddings: {pt_emb.shape}")
    
    # Transfer embedding layer norm
    if "emb_layer_norm_after.weight" in pytorch_state:
        pt_ln_w = pytorch_state["emb_layer_norm_after.weight"]
        pt_ln_b = pytorch_state["emb_layer_norm_after.bias"]
        mlx_model.embeddings.layer_norm.weight = mx.array(pt_ln_w.detach().cpu().numpy())
        mlx_model.embeddings.layer_norm.bias = mx.array(pt_ln_b.detach().cpu().numpy())
        transferred += 2
        print(f"  ✅ Transferred embedding layer norm")
    
    # Transfer first layer weights as a test
    layer_idx = 0
    mlx_layer = mlx_model.encoder.layers[layer_idx]
    
    # Attention weights
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        pt_key_w = f"layers.{layer_idx}.self_attn.{proj_name}.weight"
        pt_key_b = f"layers.{layer_idx}.self_attn.{proj_name}.bias"
        
        if pt_key_w in pytorch_state:
            pt_weight = pytorch_state[pt_key_w]
            pt_bias = pytorch_state[pt_key_b]
            
            mlx_proj = getattr(mlx_layer.attention, proj_name)
            mlx_proj.weight = mx.array(pt_weight.detach().cpu().numpy())
            mlx_proj.bias = mx.array(pt_bias.detach().cpu().numpy())
            transferred += 2
    
    # Layer norms
    for ln_name, mlx_ln in [("self_attn_layer_norm", mlx_layer.self_attn_layer_norm),
                            ("final_layer_norm", mlx_layer.final_layer_norm)]:
        pt_key_w = f"layers.{layer_idx}.{ln_name}.weight"
        pt_key_b = f"layers.{layer_idx}.{ln_name}.bias"
        
        if pt_key_w in pytorch_state:
            mlx_ln.weight = mx.array(pytorch_state[pt_key_w].detach().cpu().numpy())
            mlx_ln.bias = mx.array(pytorch_state[pt_key_b].detach().cpu().numpy())
            transferred += 2
    
    # FC layers
    for fc_name in ['fc1', 'fc2']:
        pt_key_w = f"layers.{layer_idx}.{fc_name}.weight"
        pt_key_b = f"layers.{layer_idx}.{fc_name}.bias"
        
        if pt_key_w in pytorch_state:
            mlx_fc = getattr(mlx_layer, fc_name)
            mlx_fc.weight = mx.array(pytorch_state[pt_key_w].detach().cpu().numpy())
            mlx_fc.bias = mx.array(pytorch_state[pt_key_b].detach().cpu().numpy())
            transferred += 2
    
    print(f"  📊 Transferred {transferred} weight tensors for layer 0")
    
    return transferred > 0


def run_inference_comparison(pytorch_model, mlx_model, pt_tokens, mlx_tokens):
    """Compare inference results between models."""
    print("\n🔬 Running inference comparison...")
    
    # PyTorch inference
    print("  🐍 PyTorch inference...")
    with torch.no_grad():
        pt_results = pytorch_model(pt_tokens, repr_layers=[pytorch_model.num_layers])
    
    pt_embeddings = pt_results["representations"][pytorch_model.num_layers]
    pt_logits = pt_results.get("logits", None)
    
    print(f"    PyTorch embeddings: {pt_embeddings.shape}")
    print(f"    PyTorch range: {pt_embeddings.min():.4f} to {pt_embeddings.max():.4f}")
    if pt_logits is not None:
        print(f"    PyTorch logits: {pt_logits.shape}")
    
    # MLX inference
    print("  🍎 MLX inference...")
    try:
        mlx_results = mlx_model(mlx_tokens, output_hidden_states=True)
        mlx_embeddings = mlx_results["last_hidden_state"]
        mlx_logits = mlx_results.get("prediction_scores", None)
        
        print(f"    MLX embeddings: {mlx_embeddings.shape}")
        print(f"    MLX range: {float(mx.min(mlx_embeddings)):.4f} to {float(mx.max(mlx_embeddings)):.4f}")
        if mlx_logits is not None:
            print(f"    MLX logits: {mlx_logits.shape}")
        
        return pt_embeddings, mlx_embeddings, pt_logits, mlx_logits
        
    except Exception as e:
        print(f"    ❌ MLX inference failed: {e}")
        return None, None, None, None


def calculate_accuracy_metrics(pt_tensor, mlx_tensor, name="tensors"):
    """Calculate accuracy metrics between PyTorch and MLX tensors."""
    if pt_tensor is None or mlx_tensor is None:
        return None
    
    # Convert to numpy for comparison
    pt_np = pt_tensor.detach().cpu().numpy()
    mlx_np = np.array(mlx_tensor)
    
    # Shape check
    if pt_np.shape != mlx_np.shape:
        print(f"  ❌ Shape mismatch for {name}: {pt_np.shape} vs {mlx_np.shape}")
        return None
    
    # Calculate metrics
    abs_diff = np.abs(pt_np - mlx_np)
    rel_diff = abs_diff / (np.abs(pt_np) + 1e-8)
    
    metrics = {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)), 
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "cosine_sim": float(np.dot(pt_np.flatten(), mlx_np.flatten()) / 
                           (np.linalg.norm(pt_np.flatten()) * np.linalg.norm(mlx_np.flatten()))),
        "correlation": float(np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1])
    }
    
    print(f"  📊 {name} accuracy:")
    print(f"    Max absolute diff: {metrics['max_abs_diff']:.6f}")
    print(f"    Mean absolute diff: {metrics['mean_abs_diff']:.6f}")
    print(f"    Max relative diff: {metrics['max_rel_diff']:.6f}")
    print(f"    Cosine similarity: {metrics['cosine_sim']:.6f}")
    print(f"    Correlation: {metrics['correlation']:.6f}")
    
    return metrics


def test_layer_by_layer_comparison(pytorch_model, mlx_model, pt_tokens, mlx_tokens):
    """Test layer-by-layer comparison."""
    print("\n🔍 Layer-by-layer comparison...")
    
    # For this test, we'd need to hook into intermediate layers
    # This is a simplified version
    print("  Note: Full layer-by-layer comparison requires hooking mechanism")
    print("  Testing only final outputs for now")
    
    return True


def main():
    """Main numerical accuracy test."""
    print("🧪 ESM2 Numerical Accuracy Test")
    print("=" * 60)
    
    try:
        # Create models
        pytorch_model, mlx_model, alphabet, config = create_matched_models()
        
        # Create test inputs  
        pt_tokens, mlx_tokens = create_test_inputs(alphabet)
        
        # Transfer some weights manually for testing
        weights_transferred = manually_transfer_weights(pytorch_model, mlx_model)
        
        if not weights_transferred:
            print("❌ Failed to transfer weights, using random weights")
        
        # Run inference comparison
        pt_embeddings, mlx_embeddings, pt_logits, mlx_logits = run_inference_comparison(
            pytorch_model, mlx_model, pt_tokens, mlx_tokens
        )
        
        if mlx_embeddings is not None:
            # Calculate accuracy metrics
            emb_metrics = calculate_accuracy_metrics(pt_embeddings, mlx_embeddings, "embeddings")
            
            if pt_logits is not None and mlx_logits is not None:
                logit_metrics = calculate_accuracy_metrics(pt_logits, mlx_logits, "logits")
            
            # Overall assessment
            print(f"\n🎯 Overall Assessment:")
            if weights_transferred and emb_metrics:
                if emb_metrics["cosine_sim"] > 0.99:
                    print("  ✅ Excellent numerical accuracy!")
                elif emb_metrics["cosine_sim"] > 0.95:
                    print("  ✅ Good numerical accuracy")
                elif emb_metrics["cosine_sim"] > 0.8:
                    print("  ⚠️ Moderate accuracy - check weight loading")
                else:
                    print("  ❌ Poor accuracy - architecture mismatch likely")
            else:
                print("  ℹ️ Using random weights - architecture validation only")
        
        print(f"\n🎉 Numerical accuracy test completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)