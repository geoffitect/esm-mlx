#!/usr/bin/env python3

"""
Inspect the actual PyTorch TransformerLayer implementation to understand the exact flow.
"""

import torch
import esm
import inspect

def inspect_transformer_layer():
    """Inspect the PyTorch TransformerLayer implementation."""
    
    print("🔍 Inspecting PyTorch TransformerLayer")
    print("=" * 50)
    
    # Load model to get transformer layer
    model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    first_layer = model.layers[0]
    
    # Get the class
    layer_class = first_layer.__class__
    print(f"Layer class: {layer_class}")
    
    # Check the source code
    try:
        source = inspect.getsource(layer_class)
        print(f"\nSource code length: {len(source)} characters")
        
        # Look for key parts
        lines = source.split('\n')
        print(f"\nKey implementation details:")
        
        in_forward = False
        forward_lines = []
        
        for i, line in enumerate(lines):
            if 'def forward(' in line or 'def __call__(' in line:
                in_forward = True
                print(f"\n🔄 Forward method found at line {i+1}")
            
            if in_forward:
                forward_lines.append(line)
                if line.strip().startswith('def ') and 'forward' not in line:
                    break
        
        # Print forward method
        print(f"\nForward method implementation:")
        for line in forward_lines[:20]:  # First 20 lines
            print(f"  {line}")
        
        if len(forward_lines) > 20:
            print(f"  ... ({len(forward_lines) - 20} more lines)")
            
    except Exception as e:
        print(f"Could not get source: {e}")
    
    # Check the actual forward method
    print(f"\n🧪 Testing forward method behavior:")
    
    # Create test input
    batch_size, seq_len, hidden_size = 1, 5, 320
    test_input = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        try:
            output = first_layer(test_input)
            if isinstance(output, tuple):
                layer_output, attn_weights = output
                print(f"Output: {layer_output.shape}")
                print(f"Attention weights: {attn_weights.shape if attn_weights is not None else None}")
            else:
                layer_output = output
                print(f"Output: {layer_output.shape}")
        except Exception as e:
            print(f"Forward failed: {e}")
    
    # Check what activation function is used
    print(f"\n⚡ Activation function investigation:")
    if hasattr(first_layer, 'activation_fn'):
        print(f"Activation fn attribute: {first_layer.activation_fn}")
    
    # Test specific activation
    test_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    
    # Test different activations
    relu_result = torch.relu(test_tensor)
    gelu_result = torch.nn.functional.gelu(test_tensor)
    
    print(f"Test input: {test_tensor}")
    print(f"ReLU result: {relu_result}")
    print(f"GELU result: {gelu_result}")
    
    # Let's test what the layer actually does by hooking into FC1
    if hasattr(first_layer, 'fc1'):
        fc1_output = None
        def hook_fc1(module, input, output):
            global fc1_output
            fc1_output = output.clone()
        
        hook = first_layer.fc1.register_forward_hook(hook_fc1)
        
        with torch.no_grad():
            first_layer(test_input)
        
        hook.remove()
        
        if fc1_output is not None:
            # Test what happens after FC1
            print(f"\nFC1 output sample: {fc1_output[0, 0, :5]}")
            
            # Apply different activations
            relu_activated = torch.relu(fc1_output[0, 0, :5])
            gelu_activated = torch.nn.functional.gelu(fc1_output[0, 0, :5])
            
            print(f"After ReLU: {relu_activated}")
            print(f"After GELU: {gelu_activated}")

def check_layer_norm_behavior():
    """Check layer norm behavior."""
    print(f"\n📏 Layer Normalization Behavior")
    print("=" * 40)
    
    model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    first_layer = model.layers[0]
    
    # Test input
    test_input = torch.randn(1, 5, 320)
    
    with torch.no_grad():
        # Test layer norms
        ln1 = first_layer.self_attn_layer_norm
        ln2 = first_layer.final_layer_norm
        
        normalized1 = ln1(test_input)
        normalized2 = ln2(test_input)
        
        print(f"Input range: {test_input.min():.4f} to {test_input.max():.4f}")
        print(f"After LN1: {normalized1.min():.4f} to {normalized1.max():.4f}")
        print(f"After LN2: {normalized2.min():.4f} to {normalized2.max():.4f}")
        
        # Check eps value
        print(f"LN1 eps: {ln1.eps}")
        print(f"LN2 eps: {ln2.eps}")

def main():
    """Main inspection function."""
    try:
        inspect_transformer_layer()
        check_layer_norm_behavior()
        
        print(f"\n🎯 Summary of findings:")
        print("1. Need to verify exact forward pass implementation")
        print("2. Check activation function (likely GELU)")
        print("3. Verify layer norm eps values")
        print("4. Check attention mechanism details")
        
        return True
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()