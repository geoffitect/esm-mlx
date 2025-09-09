#!/usr/bin/env python3

"""
Debug PyTorch backend and device usage.
"""

import torch
import esm
import time
import numpy as np

def check_pytorch_backend():
    """Check what backend PyTorch is using."""
    print("🔍 PyTorch Backend Analysis")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Check default device
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    
    print(f"Default device: {device}")
    
    # Load model and check where it runs
    print(f"\nLoading ESM2 model...")
    pytorch_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    
    # Check model device
    model_device = next(pytorch_model.parameters()).device
    print(f"Model device: {model_device}")
    
    # Create test input
    batch_converter = alphabet.get_batch_converter()
    data = [("test", "MKTVRQERLK")]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    print(f"Input tokens device: {batch_tokens.device}")
    
    return pytorch_model, batch_tokens, device

def benchmark_pytorch_devices():
    """Benchmark PyTorch on different devices."""
    print(f"\n🚀 PyTorch Device Benchmark")
    print("=" * 50)
    
    pytorch_model, batch_tokens, _ = check_pytorch_backend()
    
    devices_to_test = ["cpu"]
    if torch.backends.mps.is_available():
        devices_to_test.append("mps")
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
    
    results = {}
    
    for device_name in devices_to_test:
        print(f"\nTesting on {device_name}...")
        device = torch.device(device_name)
        
        # Move model and input to device
        model = pytorch_model.to(device)
        tokens = batch_tokens.to(device)
        
        # Warmup
        with torch.no_grad():
            _ = model(tokens, repr_layers=[model.num_layers])
        
        # Benchmark
        times = []
        for i in range(5):
            start_time = time.time()
            with torch.no_grad():
                outputs = model(tokens, repr_layers=[model.num_layers])
                # Force computation to complete
                _ = outputs["representations"][model.num_layers].cpu()
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times[1:])  # Skip first run
        results[device_name] = avg_time
        
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Output shape: {outputs['representations'][model.num_layers].shape}")
    
    # Compare results
    print(f"\n📊 Device Comparison:")
    cpu_time = results.get("cpu", float('inf'))
    for device_name, time_taken in results.items():
        if device_name != "cpu":
            speedup = cpu_time / time_taken
            print(f"  {device_name} vs CPU: {speedup:.2f}x speedup")
    
    return results

def check_attention_implementation():
    """Check how PyTorch implements attention."""
    print(f"\n🧠 PyTorch Attention Implementation")
    print("=" * 50)
    
    pytorch_model, batch_tokens, _ = check_pytorch_backend()
    attention_layer = pytorch_model.layers[0].self_attn
    
    print(f"Attention class: {attention_layer.__class__}")
    print(f"Attention module: {attention_layer.__class__.__module__}")
    
    # Check if it's using optimized implementations
    try:
        print(f"Attention implementation details:")
        print(f"  embed_dim: {attention_layer.embed_dim}")
        print(f"  num_heads: {attention_layer.num_heads}")
        print(f"  batch_first: {getattr(attention_layer, 'batch_first', 'not specified')}")
        print(f"  add_bias_kv: {getattr(attention_layer, 'add_bias_kv', 'not specified')}")
        print(f"  add_zero_attn: {getattr(attention_layer, 'add_zero_attn', 'not specified')}")
        
        # Check for optimized kernels
        if hasattr(attention_layer, '_qkv_same_embed_dim'):
            print(f"  _qkv_same_embed_dim: {attention_layer._qkv_same_embed_dim}")
        
    except Exception as e:
        print(f"Could not get attention details: {e}")

def profile_pytorch_operations():
    """Profile individual PyTorch operations."""
    print(f"\n⏱️ PyTorch Operation Profiling")
    print("=" * 50)
    
    pytorch_model, batch_tokens, device = check_pytorch_backend()
    
    # Move to best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        pytorch_model = pytorch_model.to(device)
        batch_tokens = batch_tokens.to(device)
    
    # Profile individual operations
    with torch.no_grad():
        # Embedding
        start_time = time.time()
        embeddings = pytorch_model.embed_tokens(batch_tokens)
        embed_time = time.time() - start_time
        
        # Layer norm
        start_time = time.time()
        normed_embeddings = pytorch_model.emb_layer_norm_after(embeddings)
        norm_time = time.time() - start_time
        
        # Single attention layer
        start_time = time.time()
        attn_input = normed_embeddings.transpose(0, 1)  # seq_first
        attn_output, _ = pytorch_model.layers[0].self_attn(attn_input, attn_input, attn_input)
        attn_time = time.time() - start_time
        
        # Full forward pass
        start_time = time.time()
        outputs = pytorch_model(batch_tokens, repr_layers=[pytorch_model.num_layers])
        full_time = time.time() - start_time
    
    print(f"Operation timings (device: {device}):")
    print(f"  Embedding: {embed_time*1000:.2f}ms")
    print(f"  Layer norm: {norm_time*1000:.2f}ms") 
    print(f"  Single attention: {attn_time*1000:.2f}ms")
    print(f"  Full forward pass: {full_time*1000:.2f}ms")
    
    return {
        "embed_time": embed_time,
        "norm_time": norm_time,
        "attn_time": attn_time,
        "full_time": full_time,
        "device": str(device)
    }

def main():
    """Main debugging function."""
    try:
        # Check backend
        check_pytorch_backend()
        
        # Benchmark devices
        device_results = benchmark_pytorch_devices()
        
        # Check attention implementation
        check_attention_implementation()
        
        # Profile operations
        profile_results = profile_pytorch_operations()
        
        print(f"\n🎯 Key Insights:")
        print(f"1. PyTorch device performance: {device_results}")
        print(f"2. PyTorch operation breakdown: {profile_results}")
        print(f"3. This will help us understand what we need to match in MLX")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend debugging failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()