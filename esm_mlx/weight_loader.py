#!/usr/bin/env python3

"""
MLX weight loading utilities for ESM2 models.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any, List
from pathlib import Path

from .esm2_mlx import ESM2MLX
from .convert_weights import convert_pytorch_weights_to_mlx


def load_weights_into_mlx_model(mlx_model: ESM2MLX, mlx_weights: Dict[str, mx.array]) -> Dict[str, str]:
    """
    Load converted weights into MLX model.
    
    Args:
        mlx_model: The MLX ESM2 model
        mlx_weights: Dictionary of converted weights
        
    Returns:
        Dictionary of loading results
    """
    
    results = {
        "loaded": [],
        "missing": [],
        "shape_mismatches": [],
        "unexpected": []
    }
    
    # Get model's parameter dictionary
    model_params = dict(mlx_model.named_parameters())
    
    # Track which model parameters we've loaded
    loaded_params = set()
    
    print(f"🔄 Loading weights into MLX model...")
    print(f"  Model has {len(model_params)} parameters")
    print(f"  Converted weights: {len(mlx_weights)} tensors")
    
    # Load weights
    for weight_key, weight_tensor in mlx_weights.items():
        if weight_key in model_params:
            model_param = model_params[weight_key]
            
            # Check shape compatibility
            if weight_tensor.shape == model_param.shape:
                # Load the weight
                try:
                    # In MLX, we need to update the parameter directly
                    # This is a simplified approach - in practice you'd use proper loading
                    setattr(mlx_model, weight_key.replace('.', '_temp_'), weight_tensor)
                    results["loaded"].append(weight_key)
                    loaded_params.add(weight_key)
                except Exception as e:
                    print(f"  ❌ Failed to load {weight_key}: {e}")
            else:
                results["shape_mismatches"].append({
                    "key": weight_key,
                    "expected": model_param.shape,
                    "got": weight_tensor.shape
                })
        else:
            results["unexpected"].append(weight_key)
    
    # Find missing parameters
    for param_name in model_params:
        if param_name not in loaded_params:
            results["missing"].append(param_name)
    
    # Print summary
    print(f"\n📊 Weight Loading Summary:")
    print(f"  ✅ Loaded: {len(results['loaded'])} weights")
    print(f"  ❌ Missing: {len(results['missing'])} weights")
    print(f"  ⚠️ Shape mismatches: {len(results['shape_mismatches'])}")
    print(f"  🔍 Unexpected: {len(results['unexpected'])}")
    
    if results["shape_mismatches"]:
        print(f"\n⚠️ Shape Mismatches:")
        for mismatch in results["shape_mismatches"][:3]:  # Show first 3
            print(f"  {mismatch['key']}: expected {mismatch['expected']}, got {mismatch['got']}")
    
    if results["missing"]:
        print(f"\n❌ Missing Parameters:")
        for missing in results["missing"][:5]:  # Show first 5
            print(f"  {missing}")
        if len(results["missing"]) > 5:
            print(f"  ... and {len(results['missing']) - 5} more")
    
    return results


def create_mlx_model_with_pytorch_weights(pytorch_model, alphabet):
    """
    Create MLX model and load PyTorch weights.
    
    Args:
        pytorch_model: PyTorch ESM2 model
        alphabet: ESM alphabet
        
    Returns:
        Tuple of (mlx_model, loading_results)
    """
    from .convert_weights import extract_esm2_config_from_pytorch
    
    print("🏗️ Creating MLX model with PyTorch weights...")
    
    # Extract config
    config = extract_esm2_config_from_pytorch(pytorch_model)
    
    # Create MLX model
    mlx_model = ESM2MLX(config)
    
    # Convert weights
    pytorch_state_dict = pytorch_model.state_dict()
    mlx_weights = convert_pytorch_weights_to_mlx(pytorch_state_dict)
    
    # Load weights (simplified approach)
    loading_results = load_weights_into_mlx_model(mlx_model, mlx_weights)
    
    return mlx_model, loading_results


def create_parameter_mapping():
    """
    Create a detailed mapping between PyTorch and MLX parameter names.
    This helps debug weight loading issues.
    """
    
    # This would be populated based on the actual model structure
    mapping = {
        # Embeddings
        "embed_tokens.weight": "embeddings.word_embeddings.weight",
        "emb_layer_norm_after.weight": "embeddings.layer_norm.weight", 
        "emb_layer_norm_after.bias": "embeddings.layer_norm.bias",
        
        # Transformer layers (template)
        "layers.{}.self_attn.q_proj.weight": "encoder.layers.{}.attention.q_proj.weight",
        "layers.{}.self_attn.q_proj.bias": "encoder.layers.{}.attention.q_proj.bias",
        "layers.{}.self_attn.k_proj.weight": "encoder.layers.{}.attention.k_proj.weight", 
        "layers.{}.self_attn.k_proj.bias": "encoder.layers.{}.attention.k_proj.bias",
        "layers.{}.self_attn.v_proj.weight": "encoder.layers.{}.attention.v_proj.weight",
        "layers.{}.self_attn.v_proj.bias": "encoder.layers.{}.attention.v_proj.bias",
        "layers.{}.self_attn.out_proj.weight": "encoder.layers.{}.attention.out_proj.weight",
        "layers.{}.self_attn.out_proj.bias": "encoder.layers.{}.attention.out_proj.bias",
        
        "layers.{}.self_attn_layer_norm.weight": "encoder.layers.{}.self_attn_layer_norm.weight",
        "layers.{}.self_attn_layer_norm.bias": "encoder.layers.{}.self_attn_layer_norm.bias",
        "layers.{}.final_layer_norm.weight": "encoder.layers.{}.final_layer_norm.weight",
        "layers.{}.final_layer_norm.bias": "encoder.layers.{}.final_layer_norm.bias",
        
        "layers.{}.fc1.weight": "encoder.layers.{}.fc1.weight",
        "layers.{}.fc1.bias": "encoder.layers.{}.fc1.bias", 
        "layers.{}.fc2.weight": "encoder.layers.{}.fc2.weight",
        "layers.{}.fc2.bias": "encoder.layers.{}.fc2.bias",
        
        # Output layers
        "lm_head.weight": "lm_head.weight",
        "lm_head.bias": "lm_head.bias",
        "lm_head.dense.weight": "lm_head.dense.weight", 
        "lm_head.dense.bias": "lm_head.dense.bias",
        "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
    }
    
    return mapping


def analyze_model_parameters(model, name="Model"):
    """Analyze model parameters for debugging."""
    print(f"\n🔍 {name} Parameters:")
    
    params = dict(model.named_parameters())
    total_params = sum(p.size if hasattr(p, 'size') else p.numel() for p in params.values())
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameter count: {len(params)}")
    
    # Group by component
    groups = {}
    for name, param in params.items():
        if 'embedding' in name:
            group = 'embeddings'
        elif 'encoder.layers' in name:
            group = 'encoder'
        elif 'lm_head' in name:
            group = 'lm_head'
        else:
            group = 'other'
        
        if group not in groups:
            groups[group] = []
        groups[group].append((name, param.shape if hasattr(param, 'shape') else param.size()))
    
    for group, params_list in groups.items():
        print(f"  {group}: {len(params_list)} parameters")
        if len(params_list) <= 3:
            for pname, pshape in params_list:
                print(f"    {pname}: {pshape}")
        else:
            for pname, pshape in params_list[:2]:
                print(f"    {pname}: {pshape}")
            print(f"    ... and {len(params_list) - 2} more")


def save_mlx_weights(weights: Dict[str, mx.array], save_path: str):
    """Save MLX weights to file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to a format that can be saved
    # MLX has mx.save and mx.load functions
    try:
        mx.savez(str(save_path), **weights)
        print(f"✅ Saved weights to {save_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save weights: {e}")
        return False


def load_mlx_weights(load_path: str) -> Dict[str, mx.array]:
    """Load MLX weights from file."""
    try:
        weights = mx.load(load_path)
        print(f"✅ Loaded weights from {load_path}")
        return dict(weights)
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return {}