#!/usr/bin/env python3

"""
Inspect the actual PyTorch ESM MultiheadAttention implementation.
"""

import torch
import esm
import inspect

def inspect_esm_attention():
    """Inspect ESM's MultiheadAttention implementation."""
    print("🔍 Inspecting ESM MultiheadAttention")
    print("=" * 50)
    
    # Load model
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    pt_attn = pytorch_model.layers[0].self_attn
    
    print(f"Attention class: {pt_attn.__class__}")
    print(f"Attention module: {pt_attn.__class__.__module__}")
    
    # Get the source code
    try:
        source = inspect.getsource(pt_attn.__class__)
        print(f"\nSource code length: {len(source)} characters")
        
        # Look for forward method
        lines = source.split('\n')
        print(f"\nKey implementation details:")
        
        in_forward = False
        forward_lines = []
        
        for i, line in enumerate(lines):
            if 'def forward(' in line:
                in_forward = True
                print(f"\n🔄 Forward method found at line {i+1}")
            
            if in_forward:
                forward_lines.append(line)
                if line.strip().startswith('def ') and 'forward' not in line:
                    break
        
        # Print forward method
        print(f"\nForward method implementation:")
        for line in forward_lines[:30]:  # First 30 lines
            if line.strip():  # Skip empty lines
                print(f"  {line}")
        
        if len(forward_lines) > 30:
            print(f"  ... ({len(forward_lines) - 30} more lines)")
            
    except Exception as e:
        print(f"Could not get source: {e}")
    
    # Check the actual call signature
    print(f"\n📋 Method signatures:")
    methods = ['forward', '__call__']
    for method_name in methods:
        if hasattr(pt_attn, method_name):
            method = getattr(pt_attn, method_name)
            try:
                sig = inspect.signature(method)
                print(f"  {method_name}{sig}")
            except Exception as e:
                print(f"  {method_name}: Could not get signature - {e}")

def test_attention_call_formats():
    """Test different ways to call PyTorch attention."""
    print("\n🧪 Testing Attention Call Formats")
    print("=" * 50)
    
    # Load model
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    pt_attn = pytorch_model.layers[0].self_attn
    
    # Create test input
    with torch.no_grad():
        test_tokens = torch.tensor([[0, 4, 5, 6, 2]])
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
        pt_normed_embeddings = pytorch_model.emb_layer_norm_after(pt_embeddings)
    
    print(f"Test input shape: {pt_normed_embeddings.shape}")
    
    # Test 1: Standard call with batch_first
    print(f"\n1. Standard call (batch_first):")
    try:
        with torch.no_grad():
            output1, weights1 = pt_attn(
                query=pt_normed_embeddings,
                key=pt_normed_embeddings,
                value=pt_normed_embeddings,
                need_weights=True
            )
        print(f"   ✅ Success!")
        print(f"   Output: {output1.shape}")
        print(f"   Weights: {weights1.shape if weights1 is not None else None}")
        print(f"   Output range: {output1.min():.4f} to {output1.max():.4f}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: Transposed input (seq_first)
    print(f"\n2. Transposed input (seq_first):")
    transposed_input = pt_normed_embeddings.transpose(0, 1)  # (seq, batch, dim)
    print(f"   Transposed shape: {transposed_input.shape}")
    
    try:
        with torch.no_grad():
            output2, weights2 = pt_attn(
                query=transposed_input,
                key=transposed_input,
                value=transposed_input,
                need_weights=True
            )
        print(f"   ✅ Success!")
        print(f"   Output: {output2.shape}")
        print(f"   Weights: {weights2.shape if weights2 is not None else None}")
        print(f"   Output range: {output2.min():.4f} to {output2.max():.4f}")
        
        # Transpose back
        output2_transposed = output2.transpose(0, 1)
        print(f"   Output transposed back: {output2_transposed.shape}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Check if outputs match between formats
    try:
        diff = torch.abs(output1 - output2_transposed).max()
        print(f"\n📊 Comparison between formats:")
        print(f"   Max difference: {diff:.8f}")
        if diff < 1e-6:
            print(f"   ✅ Formats give same result!")
        else:
            print(f"   ❌ Formats give different results!")
    except:
        print(f"   Could not compare formats")

def examine_attention_weights():
    """Examine attention weight computation in detail."""
    print("\n🔍 Examining Attention Weights")
    print("=" * 50)
    
    # Load model
    pytorch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    pt_attn = pytorch_model.layers[0].self_attn
    
    # Create test input
    with torch.no_grad():
        test_tokens = torch.tensor([[0, 4, 5, 6, 2]])
        pt_embeddings = pytorch_model.embed_tokens(test_tokens)
        pt_normed_embeddings = pytorch_model.emb_layer_norm_after(pt_embeddings)
    
    # Call attention with different options
    with torch.no_grad():
        # Standard call
        output, weights = pt_attn(
            query=pt_normed_embeddings,
            key=pt_normed_embeddings,
            value=pt_normed_embeddings,
            need_weights=True,
            need_head_weights=False
        )
        
        print(f"Standard attention weights:")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weights range: {weights.min():.6f} to {weights.max():.6f}")
        print(f"  Weights sum per row: {weights.sum(dim=-1)}")
        
        # Try with head weights
        try:
            output_head, weights_head = pt_attn(
                query=pt_normed_embeddings,
                key=pt_normed_embeddings,
                value=pt_normed_embeddings,
                need_weights=True,
                need_head_weights=True
            )
            print(f"\nWith head weights:")
            print(f"  Head weights shape: {weights_head.shape}")
            print(f"  Head weights range: {weights_head.min():.6f} to {weights_head.max():.6f}")
        except Exception as e:
            print(f"\nWith head weights failed: {e}")

def main():
    """Main inspection function."""
    try:
        inspect_esm_attention()
        test_attention_call_formats()
        examine_attention_weights()
        
        print(f"\n🎯 Key Findings:")
        print(f"1. Check the exact PyTorch attention implementation")
        print(f"2. Verify tensor format requirements")
        print(f"3. Understand attention weight shapes")
        print(f"4. Compare with standard multi-head attention")
        
        return True
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()