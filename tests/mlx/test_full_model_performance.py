#!/usr/bin/env python3

"""
Test full model performance with real weights to validate MLX value proposition.
This test will determine if MLX provides meaningful speedups and memory savings.
"""

import sys
import os
import time
import torch
import esm
import numpy as np
import psutil
import gc
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from esm_mlx import ESM2MLX, ESM2Config
from esm_mlx.convert_weights import extract_esm2_config_from_pytorch


class PerformanceProfiler:
    """Simple performance profiler for memory and timing."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
    
    def start(self):
        """Start profiling."""
        gc.collect()  # Clean up before measurement
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
    
    def stop(self) -> Dict[str, float]:
        """Stop profiling and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "duration": end_time - self.start_time,
            "memory_start": self.start_memory,
            "memory_end": end_memory,
            "memory_peak": self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else end_memory,
            "memory_used": end_memory - self.start_memory
        }


def create_full_models_with_weights():
    """Create both PyTorch and MLX models with transferred weights."""
    print("🏗️ Creating Full Models with Real Weights")
    print("=" * 50)
    
    # Load PyTorch model
    print("Loading PyTorch ESM2-8M model...")
    pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    pytorch_model.eval()
    
    print(f"✅ PyTorch model loaded: {sum(p.numel() for p in pytorch_model.parameters()):,} parameters")
    
    # Create MLX model
    print("Creating MLX model...")
    config = extract_esm2_config_from_pytorch(pytorch_model)
    mlx_model = ESM2MLX(config)
    
    print(f"✅ MLX model created with config: {config.hidden_size}D, {config.num_hidden_layers} layers")
    
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
    
    # Disable dropout for consistent testing
    def disable_dropout_recursive(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = 0.0
            else:
                disable_dropout_recursive(child)
    
    disable_dropout_recursive(pytorch_model)
    
    # For MLX, we'll disable dropout in our forward calls
    print("✅ Disabled dropout for consistent testing")
    
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


def benchmark_model(model, tokens, framework_name: str, is_pytorch: bool = True) -> Dict[str, float]:
    """Benchmark a model with given tokens."""
    
    profiler = PerformanceProfiler()
    
    # Warmup run (don't measure)
    try:
        if is_pytorch:
            with torch.no_grad():
                _ = model(tokens, repr_layers=[model.num_layers])
        else:
            # MLX model
            _ = model(tokens)
    except Exception as e:
        print(f"    ❌ Warmup failed: {e}")
        return {"duration": float('inf'), "memory_used": float('inf'), "error": str(e)}
    
    # Measured run
    profiler.start()
    
    try:
        if is_pytorch:
            with torch.no_grad():
                outputs = model(tokens, repr_layers=[model.num_layers])
        else:
            # MLX model  
            outputs = model(tokens)
        
        # Force computation to complete
        if is_pytorch:
            _ = outputs["representations"][model.num_layers].cpu()
        else:
            _ = np.array(outputs["last_hidden_state"])
        
        metrics = profiler.stop()
        
        # Add output info
        if is_pytorch:
            output_shape = outputs["representations"][model.num_layers].shape
        else:
            output_shape = outputs["last_hidden_state"].shape
        
        metrics["output_shape"] = output_shape
        metrics["success"] = True
        
        return metrics
        
    except Exception as e:
        metrics = profiler.stop()
        metrics["error"] = str(e)
        metrics["success"] = False
        return metrics


def run_performance_comparison():
    """Run comprehensive performance comparison."""
    print("\n🚀 Performance Comparison: PyTorch vs MLX")
    print("=" * 60)
    
    # Create models
    pytorch_model, mlx_model, alphabet = create_full_models_with_weights()
    
    # Create test sequences
    print(f"\nCreating test sequences...")
    test_sequences = create_test_sequences(alphabet)
    
    # Results storage
    results = {
        "pytorch": {},
        "mlx": {}
    }
    
    print(f"\n📊 Running benchmarks...")
    
    for seq_name, pt_tokens, mlx_tokens in test_sequences:
        print(f"\n  Testing: {seq_name}")
        print(f"    Token shape: {pt_tokens.shape}")
        
        # PyTorch benchmark
        print(f"    🐍 PyTorch...")
        pt_metrics = benchmark_model(pytorch_model, pt_tokens, "PyTorch", is_pytorch=True)
        results["pytorch"][seq_name] = pt_metrics
        
        if pt_metrics["success"]:
            print(f"      Time: {pt_metrics['duration']:.3f}s")
            print(f"      Memory: {pt_metrics['memory_used']:.1f}MB")
            print(f"      Output: {pt_metrics['output_shape']}")
        else:
            print(f"      ❌ Failed: {pt_metrics.get('error', 'Unknown error')}")
        
        # MLX benchmark
        print(f"    🍎 MLX...")
        mlx_metrics = benchmark_model(mlx_model, mlx_tokens, "MLX", is_pytorch=False)
        results["mlx"][seq_name] = mlx_metrics
        
        if mlx_metrics["success"]:
            print(f"      Time: {mlx_metrics['duration']:.3f}s")
            print(f"      Memory: {mlx_metrics['memory_used']:.1f}MB")
            print(f"      Output: {mlx_metrics['output_shape']}")
        else:
            print(f"      ❌ Failed: {mlx_metrics.get('error', 'Unknown error')}")
        
        # Compare if both succeeded
        if pt_metrics["success"] and mlx_metrics["success"]:
            speedup = pt_metrics["duration"] / mlx_metrics["duration"]
            
            # Handle zero memory usage case
            if pt_metrics["memory_used"] == 0:
                memory_ratio_str = f"MLX: {mlx_metrics['memory_used']:.1f}MB (PyTorch: 0MB)"
                memory_meaningful = False
            else:
                memory_ratio = mlx_metrics["memory_used"] / pt_metrics["memory_used"]
                memory_ratio_str = f"{memory_ratio:.2f}x {'✅' if memory_ratio < 1.0 else '❌'}"
                memory_meaningful = True
            
            print(f"    📈 Comparison:")
            print(f"      Speedup: {speedup:.2f}x {'✅' if speedup > 1.0 else '❌'}")
            print(f"      Memory ratio: {memory_ratio_str}")
    
    return results


def analyze_results(results: Dict) -> Dict[str, float]:
    """Analyze and summarize results."""
    print(f"\n📋 Performance Analysis")
    print("=" * 40)
    
    analysis = {
        "successful_tests": 0,
        "total_tests": 0,
        "avg_speedup": 0.0,
        "avg_memory_ratio": 0.0,
        "speedups": [],
        "memory_ratios": []
    }
    
    for seq_name in results["pytorch"].keys():
        analysis["total_tests"] += 1
        
        pt_result = results["pytorch"][seq_name]
        mlx_result = results["mlx"][seq_name]
        
        if pt_result["success"] and mlx_result["success"]:
            analysis["successful_tests"] += 1
            
            speedup = pt_result["duration"] / mlx_result["duration"]
            analysis["speedups"].append(speedup)
            
            # Handle zero memory case
            if pt_result["memory_used"] == 0:
                print(f"  {seq_name}:")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Memory: MLX {mlx_result['memory_used']:.1f}MB (PyTorch: 0MB)")
            else:
                memory_ratio = mlx_result["memory_used"] / pt_result["memory_used"]
                analysis["memory_ratios"].append(memory_ratio)
                
                print(f"  {seq_name}:")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Memory ratio: {memory_ratio:.2f}x")
    
    if analysis["speedups"]:
        analysis["avg_speedup"] = np.mean(analysis["speedups"])
        
        print(f"\n  📊 Summary:")
        print(f"    Successful tests: {analysis['successful_tests']}/{analysis['total_tests']}")
        print(f"    Average speedup: {analysis['avg_speedup']:.2f}x")
        
        # Handle memory analysis
        if analysis["memory_ratios"]:
            analysis["avg_memory_ratio"] = np.mean(analysis["memory_ratios"])
            print(f"    Average memory ratio: {analysis['avg_memory_ratio']:.2f}x")
            meaningful_memory = analysis['avg_memory_ratio'] < 0.8  # At least 20% less memory
        else:
            analysis["avg_memory_ratio"] = float('inf')
            print(f"    Memory analysis: Inconclusive (PyTorch reported 0MB usage)")
            meaningful_memory = False
        
        # Value proposition assessment
        meaningful_speedup = analysis['avg_speedup'] > 1.5  # At least 50% faster
        
        print(f"\n  🎯 Value Proposition:")
        print(f"    Meaningful speedup (>1.5x): {'✅' if meaningful_speedup else '❌'} ({analysis['avg_speedup']:.2f}x)")
        if analysis["memory_ratios"]:
            print(f"    Meaningful memory savings (<0.8x): {'✅' if meaningful_memory else '❌'} ({analysis['avg_memory_ratio']:.2f}x)")
        else:
            print(f"    Meaningful memory savings: ❓ (Unable to measure accurately)")
        
        analysis["has_meaningful_benefits"] = meaningful_speedup or meaningful_memory
        
        if analysis["has_meaningful_benefits"]:
            print(f"    🚀 MLX shows clear benefits! Worth optimizing further.")
        else:
            print(f"    ⚠️ MLX benefits unclear. May not justify optimization effort.")
    else:
        print(f"    ❌ No successful comparisons to analyze")
        analysis["has_meaningful_benefits"] = False
    
    return analysis


def main():
    """Main performance testing function."""
    print("🧪 Full Model Performance Validation")
    print("=" * 60)
    print("Testing MLX value proposition: speed and memory efficiency")
    
    try:
        # Run performance comparison
        results = run_performance_comparison()
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Final recommendation
        print(f"\n🎯 Final Assessment:")
        if analysis["has_meaningful_benefits"]:
            print(f"✅ MLX shows clear performance benefits!")
            print(f"   Recommendation: Continue optimizing attention mechanism")
            print(f"   Average speedup: {analysis.get('avg_speedup', 0):.2f}x")
            print(f"   Average memory ratio: {analysis.get('avg_memory_ratio', 1):.2f}x")
        else:
            print(f"❌ MLX benefits are not compelling enough")
            print(f"   Recommendation: Reconsider the approach or target")
        
        return analysis["has_meaningful_benefits"]
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)