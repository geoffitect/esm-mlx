#!/usr/bin/env python3

"""
Debug the differences between PyTorch ESM2 and our MLX implementation.
"""

import sys
import torch
import esm

def analyze_pytorch_esm_architecture():
    """Analyze PyTorch ESM2 architecture in detail."""
    print("🔍 Detailed PyTorch ESM2 Architecture Analysis")
    print("=" * 60)
    
    # Load model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model file: {model.__class__.__module__}")
    
    # Analyze the actual model structure
    print(f"\n🏗️ Model Structure:")
    print(f"  embed_dim: {model.embed_dim}")
    print(f"  num_layers: {model.num_layers}")  
    print(f"  attention_heads: {model.attention_heads}")
    
    # Look at first layer
    first_layer = model.layers[0]
    print(f"\n🔧 First Layer Analysis:")
    print(f"  Layer class: {first_layer.__class__.__name__}")
    print(f"  Layer module: {first_layer.__class__.__module__}")
    
    # Check self-attention
    self_attn = first_layer.self_attn
    print(f"\n👁️ Self-Attention Analysis:")
    print(f"  Attention class: {self_attn.__class__.__name__}")
    print(f"  Attention module: {self_attn.__class__.__module__}")
    
    # Check for specific attributes
    attrs_to_check = [
        'embed_dim', 'kdim', 'vdim', 'num_heads', 'dropout', 'bias',
        'add_bias_kv', 'add_zero_attn', 'self_attention', 'encoder_decoder_attention'
    ]
    
    print(f"  Self-attention attributes:")
    for attr in attrs_to_check:
        if hasattr(self_attn, attr):
            value = getattr(self_attn, attr)
            print(f"    {attr}: {value}")
    
    # Check activation function
    print(f"\n⚡ Activation Function Analysis:")
    if hasattr(first_layer, 'activation_fn'):
        print(f"  Activation: {first_layer.activation_fn}")
    
    # Check feed forward
    print(f"  FC1 -> FC2 dimensions:")
    if hasattr(first_layer, 'fc1') and hasattr(first_layer, 'fc2'):
        fc1_out = first_layer.fc1.out_features
        fc2_in = first_layer.fc2.in_features
        print(f"    FC1 output: {fc1_out}")
        print(f"    FC2 input: {fc2_in}")
        print(f"    Match: {fc1_out == fc2_in}")
    
    # Check layer norms
    print(f"\n📏 Layer Normalization:")
    if hasattr(first_layer, 'self_attn_layer_norm'):
        ln1 = first_layer.self_attn_layer_norm
        print(f"  Self-attn LayerNorm: {ln1.__class__.__name__}")
        print(f"    Normalized shape: {ln1.normalized_shape}")
        print(f"    Eps: {ln1.eps}")
        
    if hasattr(first_layer, 'final_layer_norm'):
        ln2 = first_layer.final_layer_norm
        print(f"  Final LayerNorm: {ln2.__class__.__name__}")
        print(f"    Normalized shape: {ln2.normalized_shape}")
        print(f"    Eps: {ln2.eps}")
    
    # Test a simple forward pass to see the flow
    print(f"\n🧪 Testing Forward Flow:")
    
    # Create test input
    test_input = torch.randn(1, 10, model.embed_dim)
    print(f"  Input shape: {test_input.shape}")
    
    # Test first layer
    with torch.no_grad():
        layer_output, _ = first_layer(test_input)
    
    print(f"  Layer output shape: {layer_output.shape}")
    print(f"  Output range: {layer_output.min():.4f} to {layer_output.max():.4f}")
    
    # Check if there are any hooks or special behaviors
    print(f"\n🔗 Model Hooks and Special Behaviors:")
    print(f"  Forward hooks: {len(model._forward_hooks)}")
    print(f"  Backward hooks: {len(model._backward_hooks)}")
    
    return model, first_layer

def test_individual_components():
    """Test individual components of the PyTorch model."""
    print(f"\n🧩 Testing Individual Components")
    print("=" * 40)
    
    model, first_layer = analyze_pytorch_esm_architecture()
    
    # Test embeddings
    print(f"\n📝 Testing Embeddings:")
    embed_tokens = model.embed_tokens
    test_tokens = torch.tensor([[0, 4, 5, 6, 2]])  # CLS, A, C, D, EOS
    
    with torch.no_grad():
        embeddings = embed_tokens(test_tokens)
    
    print(f"  Input tokens: {test_tokens}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding range: {embeddings.min():.4f} to {embeddings.max():.4f}")
    
    # Test layer norm after embeddings
    if hasattr(model, 'emb_layer_norm_after'):
        with torch.no_grad():
            normed_embeddings = model.emb_layer_norm_after(embeddings)
        
        print(f"  After layer norm shape: {normed_embeddings.shape}")
        print(f"  After layer norm range: {normed_embeddings.min():.4f} to {normed_embeddings.max():.4f}")
    
    return model

def compare_attention_mechanisms():
    """Compare attention mechanisms."""
    print(f"\n👁️ Attention Mechanism Deep Dive")
    print("=" * 40)
    
    model, _ = analyze_pytorch_esm_architecture()
    first_layer = model.layers[0]
    attn = first_layer.self_attn
    
    # Create test input
    hidden_size = model.embed_dim
    seq_len = 10
    batch_size = 1
    
    test_input = torch.randn(seq_len, batch_size, hidden_size)  # Note: PyTorch uses seq_first
    
    print(f"  Input shape (PyTorch format): {test_input.shape}")
    
    # Test attention
    with torch.no_grad():
        attn_output, attn_weights = attn(test_input, test_input, test_input)
    
    print(f"  Attention output shape: {attn_output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
    
    # Check the q, k, v projections
    print(f"\n  Projection dimensions:")
    print(f"    Q proj: {attn.q_proj.weight.shape}")
    print(f"    K proj: {attn.k_proj.weight.shape}")
    print(f"    V proj: {attn.v_proj.weight.shape}")
    print(f"    Out proj: {attn.out_proj.weight.shape}")
    
    return attn

def main():
    """Main debugging function."""
    try:
        # Detailed architecture analysis
        model = analyze_pytorch_esm_architecture()
        
        # Test individual components
        test_individual_components()
        
        # Compare attention mechanisms
        compare_attention_mechanisms()
        
        print(f"\n🎯 Key Findings for MLX Implementation:")
        print("1. Check attention mechanism - PyTorch vs MLX differences")
        print("2. Verify layer normalization placement and parameters")
        print("3. Check activation functions (GELU vs ReLU)")  
        print("4. Verify tensor shape conventions (batch_first vs seq_first)")
        print("5. Check embedding processing and position encoding")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()