# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MLX quantization utilities for ESMFold models."""

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import quantized as qnn
from typing import Optional, Dict, Any

from .esmfold_mlx import ESMFoldMLX, ESMFoldConfig
from .esm2_mlx import ESM2MLX


def quantize_esmfold_model(
    model: ESMFoldMLX,
    bits: int = 4,
    group_size: int = 64,
    quantize_backbone: bool = True,
    quantize_structure: bool = True,
    quantize_confidence: bool = True
) -> ESMFoldMLX:
    """
    Quantize ESMFold model for faster inference and reduced memory usage.
    
    Args:
        model: ESMFold model to quantize
        bits: Number of bits for quantization (4, 8, 16)
        group_size: Group size for quantization (32, 64, 128)
        quantize_backbone: Whether to quantize ESM-2 backbone
        quantize_structure: Whether to quantize structure prediction components
        quantize_confidence: Whether to quantize confidence prediction components
        
    Returns:
        Quantized ESMFold model
    """
    
    print(f"🔥 Quantizing ESMFold model to {bits}-bit with group_size={group_size}")
    
    # Quantize ESM-2 backbone
    if quantize_backbone:
        print("  Quantizing ESM-2 backbone...")
        model.esm = quantize_esm2_model(model.esm, bits, group_size)
    
    # Quantize folding trunk
    if quantize_structure:
        print("  Quantizing folding trunk...")
        model.folding_trunk = quantize_folding_trunk(model.folding_trunk, bits, group_size)
        
        print("  Quantizing structure prediction head...")
        model.structure_head = quantize_structure_head(model.structure_head, bits, group_size)
    
    # Quantize confidence head
    if quantize_confidence:
        print("  Quantizing confidence head...")
        model.confidence_head = quantize_confidence_head(model.confidence_head, bits, group_size)
    
    print("✅ Quantization complete!")
    return model


def quantize_esm2_model(esm_model: ESM2MLX, bits: int = 4, group_size: int = 64) -> ESM2MLX:
    """Quantize ESM-2 transformer model."""
    
    # Quantize embeddings
    if hasattr(esm_model.embeddings, 'word_embeddings'):
        esm_model.embeddings.word_embeddings = qnn.QuantizedEmbedding.from_embedding(
            esm_model.embeddings.word_embeddings,
            bits=bits,
            group_size=group_size
        )
    
    # Quantize transformer layers
    for i, layer in enumerate(esm_model.encoder.layers):
        # Quantize attention projections
        if hasattr(layer.attention, 'q_proj'):
            layer.attention.q_proj = qnn.QuantizedLinear.from_linear(
                layer.attention.q_proj, bits=bits, group_size=group_size
            )
            layer.attention.k_proj = qnn.QuantizedLinear.from_linear(
                layer.attention.k_proj, bits=bits, group_size=group_size
            )
            layer.attention.v_proj = qnn.QuantizedLinear.from_linear(
                layer.attention.v_proj, bits=bits, group_size=group_size
            )
            layer.attention.out_proj = qnn.QuantizedLinear.from_linear(
                layer.attention.out_proj, bits=bits, group_size=group_size
            )
        
        # Quantize feed-forward layers
        if hasattr(layer, 'fc1'):
            layer.fc1 = qnn.QuantizedLinear.from_linear(
                layer.fc1, bits=bits, group_size=group_size
            )
            layer.fc2 = qnn.QuantizedLinear.from_linear(
                layer.fc2, bits=bits, group_size=group_size
            )
    
    # Quantize language modeling head
    if hasattr(esm_model, 'lm_head'):
        esm_model.lm_head = qnn.QuantizedLinear.from_linear(
            esm_model.lm_head, bits=bits, group_size=group_size
        )
    
    return esm_model


def quantize_ipa_block(ipa_block, bits: int = 4, group_size: int = 64):
    """Quantize Invariant Point Attention block."""
    
    ipa = ipa_block.ipa
    
    # Quantize scalar projections
    ipa.linear_q = qnn.QuantizedLinear.from_linear(ipa.linear_q, bits=bits, group_size=group_size)
    ipa.linear_k = qnn.QuantizedLinear.from_linear(ipa.linear_k, bits=bits, group_size=group_size)
    ipa.linear_v = qnn.QuantizedLinear.from_linear(ipa.linear_v, bits=bits, group_size=group_size)
    ipa.linear_b = qnn.QuantizedLinear.from_linear(ipa.linear_b, bits=bits, group_size=group_size)
    
    # Quantize point projections
    ipa.linear_q_points = qnn.QuantizedLinear.from_linear(ipa.linear_q_points, bits=bits, group_size=group_size)
    ipa.linear_k_points = qnn.QuantizedLinear.from_linear(ipa.linear_k_points, bits=bits, group_size=group_size)
    ipa.linear_v_points = qnn.QuantizedLinear.from_linear(ipa.linear_v_points, bits=bits, group_size=group_size)
    
    # Quantize output projection
    ipa.linear_out = qnn.QuantizedLinear.from_linear(ipa.linear_out, bits=bits, group_size=group_size)
    
    return ipa_block


def quantize_sequential_layers(module, bits: int = 4, group_size: int = 64):
    """Helper to quantize layers within Sequential modules."""
    for i, layer in enumerate(module):
        if isinstance(layer, nn.Linear):
            module[i] = qnn.QuantizedLinear.from_linear(layer, bits=bits, group_size=group_size)


def quantize_folding_trunk(folding_trunk, bits: int = 4, group_size: int = 64):
    """Quantize folding trunk components."""
    
    # Quantize input projections
    folding_trunk.single_proj = qnn.QuantizedLinear.from_linear(
        folding_trunk.single_proj, bits=bits, group_size=group_size
    )
    
    # Quantize pair embeddings
    folding_trunk.pair_embed = qnn.QuantizedEmbedding.from_embedding(
        folding_trunk.pair_embed, bits=bits, group_size=group_size
    )
    
    # Quantize folding blocks
    for block in folding_trunk.folding_blocks:
        # Quantize triangular attention (it's a Sequential)
        if hasattr(block.triangular_attention, 'transform'):
            quantize_sequential_layers(block.triangular_attention.transform, bits, group_size)
        
        # Quantize IPA blocks
        for ipa_block in block.ipa_blocks:
            quantize_ipa_block(ipa_block, bits, group_size)
        
        # Quantize backbone update (it's a Sequential)
        if hasattr(block, 'backbone_update'):
            quantize_sequential_layers(block.backbone_update, bits, group_size)
    
    return folding_trunk


def quantize_structure_head(structure_head, bits: int = 4, group_size: int = 64):
    """Quantize structure prediction head."""
    
    # Quantize backbone update components
    backbone_update = structure_head.backbone_update
    
    # Quantize angle ResNet
    angle_resnet = backbone_update.angle_resnet
    
    # Quantize input projection
    angle_resnet.input_projection = qnn.QuantizedLinear.from_linear(
        angle_resnet.input_projection, bits=bits, group_size=group_size
    )
    
    # Quantize residual blocks (they are Sequential modules)
    for block in angle_resnet.blocks:
        quantize_sequential_layers(block, bits, group_size)
    
    # Quantize angle output
    angle_resnet.angle_output = qnn.QuantizedLinear.from_linear(
        angle_resnet.angle_output, bits=bits, group_size=group_size
    )
    
    # Quantize side chain prediction
    structure_head.sidechain_angles = qnn.QuantizedLinear.from_linear(
        structure_head.sidechain_angles, bits=bits, group_size=group_size
    )
    
    return structure_head


def quantize_confidence_head(confidence_head, bits: int = 4, group_size: int = 64):
    """Quantize confidence prediction head."""
    
    # Quantize LDDT head
    lddt_head = confidence_head.lddt_head
    quantize_sequential_layers(lddt_head.confidence_net, bits, group_size)
    
    # Quantize distogram head
    distogram_head = confidence_head.distogram_head
    quantize_sequential_layers(distogram_head.distance_net, bits, group_size)
    
    # Quantize TM predictor
    tm_predictor = confidence_head.tm_predictor
    quantize_sequential_layers(tm_predictor.tm_net, bits, group_size)
    
    # Quantize calibration network
    quantize_sequential_layers(confidence_head.calibration_net, bits, group_size)
    
    return confidence_head


def get_quantization_stats(model: ESMFoldMLX) -> Dict[str, Any]:
    """Get quantization statistics for the model."""
    
    stats = {
        "total_params": 0,
        "quantized_params": 0,
        "quantized_layers": 0,
        "total_layers": 0,
        "memory_reduction_ratio": 0.0,
        "layer_breakdown": {}
    }
    
    def count_params_recursive(module, prefix=""):
        nonlocal stats
        
        # For MLX modules, we need to iterate through attributes
        for name in dir(module):
            if name.startswith('_'):
                continue
                
            try:
                child = getattr(module, name)
                if not hasattr(child, '__class__'):
                    continue
                    
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, (qnn.QuantizedLinear, qnn.QuantizedEmbedding)):
                    stats["quantized_layers"] += 1
                    if hasattr(child, 'weight'):
                        params = child.weight.size
                        stats["quantized_params"] += params
                        stats["total_params"] += params
                        stats["layer_breakdown"][full_name] = {"type": "quantized", "params": params}
                elif isinstance(child, (nn.Linear, nn.Embedding)):
                    stats["total_layers"] += 1
                    if hasattr(child, 'weight'):
                        params = child.weight.size
                        stats["total_params"] += params
                        stats["layer_breakdown"][full_name] = {"type": "float32", "params": params}
                elif hasattr(child, '__dict__') and hasattr(child, '__class__'):
                    # Recursively check submodules
                    count_params_recursive(child, full_name)
            except:
                continue
    
    count_params_recursive(model)
    
    # Calculate memory reduction (rough estimate)
    if stats["total_params"] > 0:
        # Assume 4-bit quantization saves ~75% memory vs fp32
        quantized_ratio = stats["quantized_params"] / stats["total_params"]
        stats["memory_reduction_ratio"] = quantized_ratio * 0.75
    
    return stats


def benchmark_quantized_model(
    model: ESMFoldMLX,
    input_ids: mx.array,
    attention_mask: Optional[mx.array] = None,
    warmup_runs: int = 3,
    benchmark_runs: int = 10
) -> Dict[str, float]:
    """Benchmark quantized model performance."""
    
    import time
    
    print("🏁 Benchmarking quantized model...")
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(input_ids, attention_mask)
    
    # Benchmark
    times = []
    for i in range(benchmark_runs):
        start = time.time()
        output = model(input_ids, attention_mask)
        mx.eval(output["coordinates"])  # Ensure computation completes
        end = time.time()
        times.append(end - start)
        print(f"  Run {i+1}/{benchmark_runs}: {times[-1]:.4f}s")
    
    import numpy as np
    stats = {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "median_time": float(np.median(times))
    }
    
    print(f"📊 Benchmark Results:")
    print(f"  Mean: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s")
    print(f"  Range: [{stats['min_time']:.4f}s, {stats['max_time']:.4f}s]")
    
    return stats


def save_quantized_model(model: ESMFoldMLX, save_path: str):
    """Save quantized model to disk."""
    
    print(f"💾 Saving quantized model to {save_path}")
    
    # Save model weights
    model.save_weights(save_path)
    
    # Save quantization metadata
    import json
    metadata = {
        "model_type": "ESMFoldMLX",
        "quantization": "enabled",
        "bits": 4,  # Could be made configurable
        "group_size": 64,
        "stats": get_quantization_stats(model)
    }
    
    metadata_path = save_path.replace(".npz", "_quant_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved with quantization metadata")


def load_quantized_model(model_path: str, config: ESMFoldConfig) -> ESMFoldMLX:
    """Load quantized model from disk."""
    
    print(f"📂 Loading quantized model from {model_path}")
    
    # Create model and quantize it
    model = ESMFoldMLX(config)
    model = quantize_esmfold_model(model)
    
    # Load weights
    model.load_weights(model_path)
    
    print(f"✅ Quantized model loaded")
    return model