#!/usr/bin/env python3

"""
Debug MLX performance bottlenecks.
"""

import sys
import os
import time
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import mlx.core as mx
from esm_mlx import ESM2MLX, ESM2Config
from esm_mlx.convert_weights import extract_esm2_config_from_pytorch

def profile_mlx_operations():
    """Profile individual MLX operations."""
    print("⏱️ MLX Operation Profiling")
    print("=" * 50)
    
    # Create simple test data
    batch_size, seq_len, hidden_size = 1, 12, 320
    num_heads = 20
    head_dim = hidden_size // num_heads
    
    print(f"Test dimensions: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
    
    # Create test tensors
    test_input = mx.random.normal((batch_size, seq_len, hidden_size))
    test_weight = mx.random.normal((hidden_size, hidden_size))
    test_bias = mx.random.normal((hidden_size,))
    
    # Force evaluation to ensure tensors are ready
    mx.eval(test_input, test_weight, test_bias)
    
    print(f"\n🧪 Basic Operations:")
    
    # Test 1: Simple matrix multiplication
    start_time = time.time()
    for _ in range(10):
        result = mx.matmul(test_input, test_weight.T)
        mx.eval(result)  # Force evaluation
    matmul_time = (time.time() - start_time) / 10
    print(f"  Matrix multiplication: {matmul_time*1000:.2f}ms")
    
    # Test 2: Linear layer equivalent
    start_time = time.time()
    for _ in range(10):
        result = mx.matmul(test_input, test_weight.T) + test_bias
        mx.eval(result)
    linear_time = (time.time() - start_time) / 10
    print(f"  Linear layer (matmul + bias): {linear_time*1000:.2f}ms")
    
    # Test 3: Tensor reshaping (our attention uses lots of this)
    start_time = time.time()
    for _ in range(10):
        # Mimic our attention reshaping
        reshaped = test_input.transpose(1, 0, 2)  # (seq, batch, hidden)
        reshaped = reshaped.reshape(seq_len, batch_size * num_heads, head_dim)
        reshaped = reshaped.transpose(1, 0, 2)  # Back to (batch*heads, seq, head_dim)
        mx.eval(reshaped)
    reshape_time = (time.time() - start_time) / 10
    print(f"  Complex reshaping: {reshape_time*1000:.2f}ms")
    
    # Test 4: Softmax
    attention_scores = mx.random.normal((batch_size * num_heads, seq_len, seq_len))
    mx.eval(attention_scores)
    
    start_time = time.time()
    for _ in range(10):
        probs = mx.softmax(attention_scores, axis=-1)
        mx.eval(probs)
    softmax_time = (time.time() - start_time) / 10
    print(f"  Softmax: {softmax_time*1000:.2f}ms")
    
    return {
        "matmul": matmul_time,
        "linear": linear_time,
        "reshape": reshape_time,
        "softmax": softmax_time
    }

def profile_mlx_attention():
    """Profile our MLX attention implementation."""
    print(f"\n🧠 MLX Attention Profiling")
    print("=" * 50)
    
    # Create MLX model
    config = ESM2Config(
        vocab_size=33,
        hidden_size=320,
        num_hidden_layers=6,
        num_attention_heads=20,
        intermediate_size=1280,
        attention_head_dim=16
    )
    
    from esm_mlx.esm2_mlx import ESM2AttentionLayer
    attention = ESM2AttentionLayer(config)
    
    # Create test input
    batch_size, seq_len = 1, 12
    test_input = mx.random.normal((batch_size, seq_len, config.hidden_size))
    mx.eval(test_input)
    
    print(f"Testing attention with shape: {test_input.shape}")
    
    # Warmup
    output, _ = attention(test_input, output_attentions=False)
    mx.eval(output)
    
    # Profile full attention
    start_time = time.time()
    for _ in range(5):
        output, weights = attention(test_input, output_attentions=True)
        mx.eval(output, weights)
    attention_time = (time.time() - start_time) / 5
    
    print(f"  Full attention: {attention_time*1000:.2f}ms")
    print(f"  Output shape: {output.shape}")
    print(f"  Weights shape: {weights.shape if weights is not None else None}")
    
    # Profile attention steps
    print(f"\n  Breaking down attention steps:")
    
    # Step 1: Q, K, V projections
    start_time = time.time()
    q = attention.q_proj(test_input)
    k = attention.k_proj(test_input)
    v = attention.v_proj(test_input)
    mx.eval(q, k, v)
    projection_time = time.time() - start_time
    print(f"    Q, K, V projections: {projection_time*1000:.2f}ms")
    
    # Step 2: Reshaping for multi-head
    start_time = time.time()
    q_reshaped = q.transpose(1, 0, 2).reshape(seq_len, batch_size * config.num_attention_heads, config.attention_head_dim)
    k_reshaped = k.transpose(1, 0, 2).reshape(seq_len, batch_size * config.num_attention_heads, config.attention_head_dim)
    v_reshaped = v.transpose(1, 0, 2).reshape(seq_len, batch_size * config.num_attention_heads, config.attention_head_dim)
    mx.eval(q_reshaped, k_reshaped, v_reshaped)
    reshape_time = time.time() - start_time
    print(f"    Reshaping: {reshape_time*1000:.2f}ms")
    
    # Step 3: Attention scores
    start_time = time.time()
    scores = mx.matmul(q_reshaped.transpose(1, 0, 2), k_reshaped.transpose(1, 2, 0)) * attention.scale
    mx.eval(scores)
    scores_time = time.time() - start_time
    print(f"    Attention scores: {scores_time*1000:.2f}ms")
    
    # Step 4: Softmax
    start_time = time.time()
    probs = mx.softmax(scores, axis=-1)
    mx.eval(probs)
    softmax_time = time.time() - start_time
    print(f"    Softmax: {softmax_time*1000:.2f}ms")
    
    # Step 5: Apply to values
    start_time = time.time()
    v_transposed = v_reshaped.transpose(1, 0, 2)
    attn_output = mx.matmul(probs, v_transposed)
    mx.eval(attn_output)
    apply_time = time.time() - start_time
    print(f"    Apply to values: {apply_time*1000:.2f}ms")
    
    # Step 6: Final reshaping and projection
    start_time = time.time()
    attn_output_reshaped = attn_output.reshape(batch_size, config.num_attention_heads, seq_len, config.attention_head_dim)
    attn_output_final = attn_output_reshaped.transpose(2, 0, 1, 3).reshape(seq_len, batch_size, config.hidden_size)
    attn_output_final = attn_output_final.transpose(1, 0, 2)
    final_output = attention.out_proj(attn_output_final)
    mx.eval(final_output)
    final_time = time.time() - start_time
    print(f"    Final reshaping + projection: {final_time*1000:.2f}ms")
    
    total_steps = projection_time + reshape_time + scores_time + softmax_time + apply_time + final_time
    print(f"    Total step time: {total_steps*1000:.2f}ms")
    print(f"    vs Full attention: {attention_time*1000:.2f}ms")
    
    return attention_time

def profile_full_mlx_model():
    """Profile full MLX model."""
    print(f"\n🏗️ Full MLX Model Profiling")
    print("=" * 50)
    
    # Load real config
    import torch
    import esm
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    config = extract_esm2_config_from_pytorch(pytorch_model)
    
    # Create MLX model
    mlx_model = ESM2MLX(config)
    
    # Create test input
    test_tokens = mx.array([[0, 4, 5, 6, 2]])
    
    # Warmup
    output = mlx_model(test_tokens)
    mx.eval(output["last_hidden_state"])
    
    # Profile full model
    start_time = time.time()
    for _ in range(3):
        output = mlx_model(test_tokens)
        mx.eval(output["last_hidden_state"])
    full_model_time = (time.time() - start_time) / 3
    
    print(f"  Full model forward: {full_model_time*1000:.2f}ms")
    print(f"  Output shape: {output['last_hidden_state'].shape}")
    
    # Profile individual components
    print(f"\n  Breaking down model components:")
    
    # Embeddings
    start_time = time.time()
    embeddings = mlx_model.embeddings(test_tokens)
    mx.eval(embeddings)
    embed_time = time.time() - start_time
    print(f"    Embeddings: {embed_time*1000:.2f}ms")
    
    # Single transformer layer
    start_time = time.time()
    layer_output, _ = mlx_model.encoder.layers[0](embeddings)
    mx.eval(layer_output)
    layer_time = time.time() - start_time
    print(f"    Single layer: {layer_time*1000:.2f}ms")
    
    # All layers
    start_time = time.time()
    encoder_output = mlx_model.encoder(embeddings)
    mx.eval(encoder_output["last_hidden_state"])
    encoder_time = time.time() - start_time
    print(f"    All 6 layers: {encoder_time*1000:.2f}ms")
    
    # LM head
    start_time = time.time()
    lm_output = mlx_model.lm_head(encoder_output["last_hidden_state"])
    mx.eval(lm_output)
    lm_time = time.time() - start_time
    print(f"    LM head: {lm_time*1000:.2f}ms")
    
    return full_model_time

def check_mlx_compilation():
    """Check if MLX functions are being compiled efficiently."""
    print(f"\n🔧 MLX Compilation Check")
    print("=" * 50)
    
    # Simple operation that should be fast
    x = mx.random.normal((1000, 1000))
    y = mx.random.normal((1000, 1000))
    
    # First run (compilation)
    start_time = time.time()
    result = mx.matmul(x, y)
    mx.eval(result)
    first_run = time.time() - start_time
    
    # Second run (compiled)
    start_time = time.time()
    result = mx.matmul(x, y)
    mx.eval(result)
    second_run = time.time() - start_time
    
    print(f"  Large matrix multiply (1000x1000):")
    print(f"    First run: {first_run*1000:.2f}ms (includes compilation)")
    print(f"    Second run: {second_run*1000:.2f}ms (compiled)")
    print(f"    Compilation overhead: {(first_run - second_run)*1000:.2f}ms")
    
    if first_run > second_run * 2:
        print(f"    ⚠️ Significant compilation overhead detected!")
    else:
        print(f"    ✅ Compilation overhead is reasonable")

def main():
    """Main profiling function."""
    try:
        # Profile basic operations
        basic_ops = profile_mlx_operations()
        
        # Profile attention
        attention_time = profile_mlx_attention()
        
        # Profile full model
        full_model_time = profile_full_mlx_model()
        
        # Check compilation
        check_mlx_compilation()
        
        print(f"\n🎯 Performance Summary:")
        print(f"  Basic operations: {basic_ops}")
        print(f"  Attention layer: {attention_time*1000:.2f}ms")
        print(f"  Full model: {full_model_time*1000:.2f}ms")
        
        print(f"\n🔍 Potential Issues:")
        if attention_time > 0.01:  # 10ms
            print(f"  ⚠️ Attention is very slow ({attention_time*1000:.1f}ms)")
        if basic_ops["reshape"] > 0.001:  # 1ms
            print(f"  ⚠️ Tensor reshaping is slow ({basic_ops['reshape']*1000:.1f}ms)")
        if full_model_time > 0.05:  # 50ms
            print(f"  ⚠️ Full model is slow ({full_model_time*1000:.1f}ms)")
        
        return True
        
    except Exception as e:
        print(f"❌ MLX profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()