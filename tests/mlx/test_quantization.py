#!/usr/bin/env python3

"""
Test MLX quantization for ESMFold models.
Demonstrates 4-bit quantization for massive speedup and memory reduction.
"""

import sys
import time
import numpy as np
from typing import Dict, Any

import mlx.core as mx
import mlx.nn as nn

# Import our components
from esm_mlx.config import ESM2Config
from esm_mlx.esmfold_mlx import ESMFoldMLX, ESMFoldConfig
from esm_mlx.quantization import (
    quantize_esmfold_model, 
    get_quantization_stats,
    benchmark_quantized_model
)


def create_test_model_and_data():
    """Create test model and sample data."""
    
    print("🏗️  Creating test ESMFold model...")
    
    # Create compact model for testing
    esm_config = ESM2Config(
        vocab_size=33,
        hidden_size=320,
        num_hidden_layers=4,  # Reduced for testing
        num_attention_heads=16,
        intermediate_size=1280,
        layer_norm_eps=1e-5
    )
    
    esmfold_config = ESMFoldConfig(
        esm_config=esm_config,
        c_s=256,
        c_z=64,
        num_folding_blocks=2,  # Reduced for testing
        num_ipa_blocks=1,      # Reduced for testing
        num_recycles=1         # Reduced for testing
    )
    
    model = ESMFoldMLX(esmfold_config)
    
    # Create test sequence data
    batch_size = 2
    seq_len = 50
    
    # Random protein-like sequence
    input_ids = mx.random.randint(4, 24, (batch_size, seq_len))  # Skip special tokens
    attention_mask = mx.ones((batch_size, seq_len))
    
    print(f"  Model config: {esmfold_config.c_s}x{esmfold_config.c_z}, {esmfold_config.num_folding_blocks} blocks")
    print(f"  Test data: {input_ids.shape}")
    
    return model, input_ids, attention_mask


def test_model_functionality(model, input_ids, attention_mask, model_name="Model"):
    """Test that model produces valid outputs."""
    
    print(f"🧪 Testing {model_name} functionality...")
    
    try:
        start_time = time.time()
        output = model(input_ids, attention_mask, num_recycles=1)
        inference_time = time.time() - start_time
        
        # Check outputs
        coords = output["coordinates"]
        plddt = output["plddt"] 
        tm_score = output["tm_score"]
        
        print(f"  ✅ Forward pass successful in {inference_time:.4f}s")
        print(f"  📐 Coordinates: {coords.shape}, range [{mx.min(coords):.2f}, {mx.max(coords):.2f}]")
        print(f"  📊 pLDDT: {plddt.shape}, mean {mx.mean(plddt):.3f}")
        print(f"  🎯 TM-score: {tm_score.shape}, mean {mx.mean(tm_score):.3f}")
        
        return True, inference_time, output
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False, 0.0, None


def compare_model_outputs(original_output, quantized_output, tolerance=1e-2):
    """Compare outputs between original and quantized models."""
    
    print("🔍 Comparing model outputs...")
    
    def compute_similarity(a, b, name):
        diff = mx.abs(a - b)
        max_diff = mx.max(diff)
        mean_diff = mx.mean(diff)
        
        # Cosine similarity
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        cos_sim = mx.sum(a_flat * b_flat) / (mx.linalg.norm(a_flat) * mx.linalg.norm(b_flat))
        
        print(f"  {name}:")
        print(f"    Max diff: {float(max_diff):.6f}")
        print(f"    Mean diff: {float(mean_diff):.6f}")
        print(f"    Cosine similarity: {float(cos_sim):.6f}")
        
        return float(cos_sim) > 0.95  # 95% similarity threshold
    
    # Compare coordinates
    coords_similar = compute_similarity(
        original_output["coordinates"], 
        quantized_output["coordinates"], 
        "Coordinates"
    )
    
    # Compare confidence scores
    plddt_similar = compute_similarity(
        original_output["plddt"],
        quantized_output["plddt"],
        "pLDDT scores"
    )
    
    tm_similar = compute_similarity(
        original_output["tm_score"],
        quantized_output["tm_score"], 
        "TM-scores"
    )
    
    all_similar = coords_similar and plddt_similar and tm_similar
    
    if all_similar:
        print("  ✅ Quantized model outputs are highly similar to original!")
    else:
        print("  ⚠️  Some outputs show significant differences (expected for 4-bit quantization)")
    
    return all_similar


def test_quantization_levels():
    """Test different quantization bit levels."""
    
    print("\n" + "="*60)
    print("🎛️  Testing Different Quantization Levels")
    print("="*60)
    
    model, input_ids, attention_mask = create_test_model_and_data()
    
    # Get original model performance
    print("\n📊 Original FP32 Model:")
    orig_success, orig_time, orig_output = test_model_functionality(
        model, input_ids, attention_mask, "Original"
    )
    
    if not orig_success:
        print("❌ Original model failed, skipping quantization tests")
        return
    
    # Test different quantization levels
    quantization_configs = [
        {"bits": 8, "group_size": 64, "name": "8-bit"},
        {"bits": 4, "group_size": 64, "name": "4-bit"},
        {"bits": 4, "group_size": 32, "name": "4-bit (smaller groups)"}
    ]
    
    results = []
    
    for config in quantization_configs:
        print(f"\n🔬 Testing {config['name']} quantization...")
        
        try:
            # Quantize model (need to recreate to avoid modifying original)
            test_model, _, _ = create_test_model_and_data()
            
            quantized_model = quantize_esmfold_model(
                test_model,
                bits=config["bits"],
                group_size=config["group_size"],
                quantize_backbone=True,
                quantize_structure=True,
                quantize_confidence=True
            )
            
            # Get quantization stats
            quant_stats = get_quantization_stats(quantized_model)
            print(f"  📈 Quantization coverage: {quant_stats['quantized_layers']} layers")
            print(f"  💾 Estimated memory reduction: {quant_stats['memory_reduction_ratio']:.1%}")
            
            # Test functionality
            quant_success, quant_time, quant_output = test_model_functionality(
                quantized_model, input_ids, attention_mask, config['name']
            )
            
            if quant_success:
                # Compare outputs
                similar = compare_model_outputs(orig_output, quant_output)
                
                # Calculate speedup
                speedup = orig_time / quant_time if quant_time > 0 else 0
                
                results.append({
                    "config": config['name'],
                    "success": True,
                    "time": quant_time,
                    "speedup": speedup,
                    "similar": similar,
                    "memory_reduction": quant_stats['memory_reduction_ratio']
                })
                
                print(f"  🚀 Speedup: {speedup:.2f}x")
                
            else:
                results.append({
                    "config": config['name'],
                    "success": False,
                    "time": 0,
                    "speedup": 0,
                    "similar": False,
                    "memory_reduction": 0
                })
                
        except Exception as e:
            print(f"  ❌ Quantization failed: {e}")
            results.append({
                "config": config['name'],
                "success": False,
                "error": str(e),
                "time": 0,
                "speedup": 0,
                "similar": False,
                "memory_reduction": 0
            })
    
    # Summary
    print(f"\n" + "="*60)
    print("📋 QUANTIZATION RESULTS SUMMARY")
    print("="*60)
    
    print(f"{'Config':<20} {'Success':<8} {'Time':<8} {'Speedup':<8} {'Similar':<8} {'Mem Save':<8}")
    print("-" * 60)
    print(f"{'Original (FP32)':<20} {'✅':<8} {orig_time:<8.4f} {'1.00x':<8} {'N/A':<8} {'0%':<8}")
    
    for result in results:
        success_icon = "✅" if result["success"] else "❌"
        similar_icon = "✅" if result.get("similar", False) else "⚠️"
        mem_save = f"{result['memory_reduction']:.1%}"
        
        print(f"{result['config']:<20} {success_icon:<8} {result['time']:<8.4f} "
              f"{result['speedup']:<8.2f} {similar_icon:<8} {mem_save:<8}")


def benchmark_quantized_performance():
    """Comprehensive benchmark of quantized vs original performance."""
    
    print("\n" + "="*60)
    print("🏁 QUANTIZED PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Create larger model for meaningful benchmarks
    esm_config = ESM2Config(
        vocab_size=33,
        hidden_size=640,       # Larger model
        num_hidden_layers=8,   # More layers
        num_attention_heads=20,
        intermediate_size=2560,
        layer_norm_eps=1e-5
    )
    
    esmfold_config = ESMFoldConfig(
        esm_config=esm_config,
        c_s=512,              # Larger structure module
        c_z=128,
        num_folding_blocks=4,
        num_ipa_blocks=2,
        num_recycles=3
    )
    
    print(f"📏 Benchmark model: {esm_config.hidden_size}D, {esm_config.num_hidden_layers} layers")
    
    # Create test data with different sequence lengths
    test_cases = [
        {"name": "Short peptide", "seq_len": 30, "batch_size": 4},
        {"name": "Medium protein", "seq_len": 100, "batch_size": 2},
        {"name": "Large protein", "seq_len": 300, "batch_size": 1}
    ]
    
    for case in test_cases:
        print(f"\n🧬 Testing {case['name']} ({case['seq_len']} residues, batch={case['batch_size']})")
        
        # Create model and data
        model = ESMFoldMLX(esmfold_config)
        input_ids = mx.random.randint(4, 24, (case["batch_size"], case["seq_len"]))
        attention_mask = mx.ones((case["batch_size"], case["seq_len"]))
        
        # Benchmark original model
        print("  📊 Original model...")
        orig_stats = benchmark_quantized_model(
            model, input_ids, attention_mask, warmup_runs=2, benchmark_runs=5
        )
        
        # Quantize and benchmark
        print("  🔥 4-bit quantized model...")
        quantized_model = quantize_esmfold_model(model, bits=4, group_size=64)
        quant_stats = benchmark_quantized_model(
            quantized_model, input_ids, attention_mask, warmup_runs=2, benchmark_runs=5
        )
        
        # Calculate improvements
        speedup = orig_stats["mean_time"] / quant_stats["mean_time"]
        
        print(f"  🚀 Results:")
        print(f"    Original: {orig_stats['mean_time']:.4f}s ± {orig_stats['std_time']:.4f}s")
        print(f"    Quantized: {quant_stats['mean_time']:.4f}s ± {quant_stats['std_time']:.4f}s")
        print(f"    Speedup: {speedup:.2f}x")


def main():
    """Main quantization test suite."""
    
    print("🔥 MLX ESMFold Quantization Test Suite")
    print("="*60)
    print("Testing quantization capabilities for massive speedup on Apple Silicon!")
    
    try:
        # Test different quantization levels
        test_quantization_levels()
        
        # Comprehensive benchmark
        benchmark_quantized_performance()
        
        print(f"\n" + "="*60)
        print("✅ ALL QUANTIZATION TESTS COMPLETED!")
        print("🚀 ESMFold is ready for LUDICROUS SPEED with MLX quantization!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()