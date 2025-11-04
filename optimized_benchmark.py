#!/usr/bin/env python3

"""
Optimized performance benchmark with proper warmup and methodology.
"""

import sys
import os
import time
import torch
import esm
import numpy as np
import gc
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from esm_mlx import ESM2MLX, ESM2Config
from esm_mlx.convert_weights import extract_esm2_config_from_pytorch


def create_models_once():
    """Create both models once and transfer weights."""
    print("🏗️ Creating Models (One-Time Setup)")
    print("=" * 50)
    
    # Load PyTorch model
    print("Loading PyTorch ESM2-8M model...")
    pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    # Disable dropout
    def disable_dropout_recursive(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = 0.0
            else:
                disable_dropout_recursive(child)
    disable_dropout_recursive(pytorch_model)
    
    print(f"✅ PyTorch model loaded: {sum(p.numel() for p in pytorch_model.parameters()):,} parameters")
    
    # Create MLX model
    print("Creating MLX model...")
    config = extract_esm2_config_from_pytorch(pytorch_model)
    mlx_model = ESM2MLX(config)
    
    # Transfer weights systematically
    print("Transferring weights...")
    pytorch_state = pytorch_model.state_dict()
    transferred_count = 0
    
    # Transfer embeddings
    if "embed_tokens.weight" in pytorch_state:
        mlx_model.embeddings.word_embeddings.weight = mx.array(
            pytorch_state["embed_tokens.weight"].detach().cpu().numpy()
        )
        transferred_count += 1
    
    if "emb_layer_norm_after.weight" in pytorch_state:
        mlx_model.embeddings.layer_norm.weight = mx.array(
            pytorch_state["emb_layer_norm_after.weight"].detach().cpu().numpy()
        )
        mlx_model.embeddings.layer_norm.bias = mx.array(
            pytorch_state["emb_layer_norm_after.bias"].detach().cpu().numpy()
        )
        transferred_count += 2
    
    # Transfer all transformer layers
    for layer_idx in range(config.num_hidden_layers):
        mlx_layer = mlx_model.encoder.layers[layer_idx]
        
        # Attention weights
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            pt_weight = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.weight"]
            pt_bias = pytorch_state[f"layers.{layer_idx}.self_attn.{proj_name}.bias"]
            
            mlx_proj = getattr(mlx_layer.attention, proj_name)
            mlx_proj.weight = mx.array(pt_weight.detach().cpu().numpy())
            mlx_proj.bias = mx.array(pt_bias.detach().cpu().numpy())
            transferred_count += 2
        
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
        transferred_count += 4
        
        # FC layers
        mlx_layer.fc1.weight = mx.array(pytorch_state[f"layers.{layer_idx}.fc1.weight"].detach().cpu().numpy())
        mlx_layer.fc1.bias = mx.array(pytorch_state[f"layers.{layer_idx}.fc1.bias"].detach().cpu().numpy())
        mlx_layer.fc2.weight = mx.array(pytorch_state[f"layers.{layer_idx}.fc2.weight"].detach().cpu().numpy())
        mlx_layer.fc2.bias = mx.array(pytorch_state[f"layers.{layer_idx}.fc2.bias"].detach().cpu().numpy())
        transferred_count += 4
    
    # Transfer output head
    if "lm_head.weight" in pytorch_state:
        mlx_model.lm_head.weight = mx.array(pytorch_state["lm_head.weight"].detach().cpu().numpy())
        mlx_model.lm_head.bias = mx.array(pytorch_state["lm_head.bias"].detach().cpu().numpy())
        transferred_count += 2
    
    print(f"✅ Transferred {transferred_count} weight tensors")
    
    return pytorch_model, mlx_model, alphabet


def create_test_sequences(alphabet) -> List[Tuple[str, torch.Tensor, mx.array]]:
    """Create test sequences of varying lengths."""
    
    sequences = [
        ("Short peptide", "MKTVRQERLK"),
        ("Medium protein", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("Long protein", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ]
    
    test_data = []
    batch_converter = alphabet.get_batch_converter()
    
    for name, sequence in sequences:
        # PyTorch tokenization
        data = [(name, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        # MLX tokenization
        mlx_tokens = mx.array(batch_tokens.numpy())
        
        test_data.append((name, batch_tokens, mlx_tokens))
        print(f"  {name}: {len(sequence)} residues -> {batch_tokens.shape[1]} tokens")
    
    return test_data


def proper_warmup(model, tokens, framework_name: str, is_pytorch: bool = True, num_warmup: int = 5):
    """Proper warmup for both frameworks."""
    print(f"    Warming up {framework_name} ({num_warmup} runs)...")
    
    for i in range(num_warmup):
        try:
            if is_pytorch:
                with torch.no_grad():
                    outputs = model(tokens, repr_layers=[model.num_layers])
                    # Force computation to complete
                    _ = outputs["representations"][model.num_layers].cpu()
            else:
                # MLX model
                outputs = model(tokens)
                # Force computation to complete
                mx.eval(outputs["last_hidden_state"])
        except Exception as e:
            print(f"      Warmup {i+1} failed: {e}")
            return False
    
    return True


def benchmark_with_proper_methodology(model, tokens, framework_name: str, is_pytorch: bool = True, num_runs: int = 10) -> Dict[str, float]:
    """Benchmark with proper methodology."""
    
    # Proper warmup first
    if not proper_warmup(model, tokens, framework_name, is_pytorch):
        return {"success": False, "error": "Warmup failed"}
    
    # Clean up memory
    gc.collect()
    
    # Measure multiple runs
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        try:
            if is_pytorch:
                with torch.no_grad():
                    outputs = model(tokens, repr_layers=[model.num_layers])
                    # Force computation to complete
                    result = outputs["representations"][model.num_layers].cpu()
            else:
                # MLX model  
                outputs = model(tokens)
                # Force computation to complete
                result = np.array(outputs["last_hidden_state"])
            
            end_time = time.time()
            times.append(end_time - start_time)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Statistical analysis
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    # Get output info
    if is_pytorch:
        output_shape = outputs["representations"][model.num_layers].shape
    else:
        output_shape = outputs["last_hidden_state"].shape
    
    return {
        "success": True,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "times": times.tolist(),
        "output_shape": output_shape
    }


def run_optimized_benchmark():
    """Run optimized performance benchmark."""
    print("\n🚀 Optimized Performance Benchmark")
    print("=" * 60)
    
    # Create models once
    pytorch_model, mlx_model, alphabet = create_models_once()
    
    # Create test sequences
    print(f"\n📊 Creating test sequences...")
    test_sequences = create_test_sequences(alphabet)
    
    # Results storage
    results = {}
    
    print(f"\n⏱️ Running benchmarks...")
    
    for seq_name, pt_tokens, mlx_tokens in test_sequences:
        print(f"\n  Testing: {seq_name}")
        print(f"    Token shape: {pt_tokens.shape}")
        
        results[seq_name] = {}
        
        # PyTorch benchmark
        print(f"    🐍 PyTorch...")
        pt_results = benchmark_with_proper_methodology(pytorch_model, pt_tokens, "PyTorch", is_pytorch=True)
        results[seq_name]["pytorch"] = pt_results
        
        if pt_results["success"]:
            print(f"      Mean time: {pt_results['mean_time']*1000:.2f}ms ± {pt_results['std_time']*1000:.2f}ms")
            print(f"      Min time: {pt_results['min_time']*1000:.2f}ms")
            print(f"      Output: {pt_results['output_shape']}")
        else:
            print(f"      ❌ Failed: {pt_results.get('error', 'Unknown error')}")
        
        # MLX benchmark
        print(f"    🍎 MLX...")
        mlx_results = benchmark_with_proper_methodology(mlx_model, mlx_tokens, "MLX", is_pytorch=False)
        results[seq_name]["mlx"] = mlx_results
        
        if mlx_results["success"]:
            print(f"      Mean time: {mlx_results['mean_time']*1000:.2f}ms ± {mlx_results['std_time']*1000:.2f}ms")
            print(f"      Min time: {mlx_results['min_time']*1000:.2f}ms")
            print(f"      Output: {mlx_results['output_shape']}")
        else:
            print(f"      ❌ Failed: {mlx_results.get('error', 'Unknown error')}")
        
        # Compare if both succeeded
        if pt_results["success"] and mlx_results["success"]:
            speedup_mean = pt_results["mean_time"] / mlx_results["mean_time"]
            speedup_best = pt_results["min_time"] / mlx_results["min_time"]
            
            print(f"    📈 Comparison:")
            print(f"      Mean speedup: {speedup_mean:.2f}x {'✅' if speedup_mean > 1.0 else '❌'}")
            print(f"      Best speedup: {speedup_best:.2f}x {'✅' if speedup_best > 1.0 else '❌'}")
    
    return results


def analyze_optimized_results(results: Dict) -> Dict[str, float]:
    """Analyze optimized benchmark results."""
    print(f"\n📋 Optimized Performance Analysis")
    print("=" * 40)
    
    analysis = {
        "successful_tests": 0,
        "total_tests": 0,
        "mean_speedups": [],
        "best_speedups": []
    }
    
    for seq_name, seq_results in results.items():
        analysis["total_tests"] += 1
        
        pt_result = seq_results["pytorch"]
        mlx_result = seq_results["mlx"]
        
        if pt_result["success"] and mlx_result["success"]:
            analysis["successful_tests"] += 1
            
            mean_speedup = pt_result["mean_time"] / mlx_result["mean_time"]
            best_speedup = pt_result["min_time"] / mlx_result["min_time"]
            
            analysis["mean_speedups"].append(mean_speedup)
            analysis["best_speedups"].append(best_speedup)
            
            print(f"  {seq_name}:")
            print(f"    Mean speedup: {mean_speedup:.2f}x")
            print(f"    Best speedup: {best_speedup:.2f}x")
            print(f"    PyTorch: {pt_result['mean_time']*1000:.2f}ms")
            print(f"    MLX: {mlx_result['mean_time']*1000:.2f}ms")
    
    if analysis["mean_speedups"]:
        avg_mean_speedup = np.mean(analysis["mean_speedups"])
        avg_best_speedup = np.mean(analysis["best_speedups"])
        
        print(f"\n  📊 Summary:")
        print(f"    Successful tests: {analysis['successful_tests']}/{analysis['total_tests']}")
        print(f"    Average mean speedup: {avg_mean_speedup:.2f}x")
        print(f"    Average best speedup: {avg_best_speedup:.2f}x")
        
        # Value proposition assessment
        meaningful_speedup = avg_mean_speedup > 1.2  # At least 20% faster
        
        print(f"\n  🎯 Value Proposition:")
        print(f"    Meaningful speedup (>1.2x): {'✅' if meaningful_speedup else '❌'} ({avg_mean_speedup:.2f}x)")
        
        analysis["avg_mean_speedup"] = avg_mean_speedup
        analysis["avg_best_speedup"] = avg_best_speedup
        analysis["has_meaningful_benefits"] = meaningful_speedup
        
        if analysis["has_meaningful_benefits"]:
            print(f"    🚀 MLX shows clear performance benefits!")
        else:
            print(f"    ⚠️ MLX benefits are marginal.")
    else:
        print(f"    ❌ No successful comparisons to analyze")
        analysis["has_meaningful_benefits"] = False
    
    return analysis


def main():
    """Main optimized benchmarking function."""
    print("🧪 Optimized MLX vs PyTorch Benchmark")
    print("=" * 60)
    print("Using proper warmup, statistical analysis, and clean methodology")
    
    try:
        # Run optimized benchmark
        results = run_optimized_benchmark()
        
        # Analyze results
        analysis = analyze_optimized_results(results)
        
        # Final recommendation
        print(f"\n🎯 Final Assessment:")
        if analysis["has_meaningful_benefits"]:
            print(f"✅ MLX shows clear performance benefits!")
            print(f"   Average speedup: {analysis.get('avg_mean_speedup', 0):.2f}x")
            print(f"   Best case speedup: {analysis.get('avg_best_speedup', 0):.2f}x")
            print(f"   Recommendation: Continue with MLX optimization!")
        else:
            print(f"⚠️ MLX benefits are marginal")
            print(f"   Recommendation: MLX is competitive but not dramatically better")
        
        return analysis["has_meaningful_benefits"]
        
    except Exception as e:
        print(f"❌ Optimized benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)