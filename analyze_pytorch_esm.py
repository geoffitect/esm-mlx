#!/usr/bin/env python3

"""
Analyze PyTorch ESM2 model structure to understand the exact architecture
and parameter names for accurate weight conversion.
"""

import sys
import torch
import esm
from collections import OrderedDict

def load_smallest_esm2():
    """Load the smallest ESM2 model for analysis."""
    print("🔍 Loading ESM2-8M (smallest model) for analysis...")
    try:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        model.eval()
        print(f"✅ Loaded {model.__class__.__name__}")
        return model, alphabet
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def analyze_model_structure(model, alphabet):
    """Analyze the model structure in detail."""
    print(f"\n📊 Model Analysis")
    print("=" * 50)
    
    # Basic model info
    print(f"Model class: {model.__class__.__name__}")
    print(f"Alphabet size: {len(alphabet)}")
    
    # Model attributes
    if hasattr(model, 'embed_dim'):
        print(f"Embed dimension: {model.embed_dim}")
    if hasattr(model, 'num_layers'):
        print(f"Number of layers: {model.num_layers}")
    if hasattr(model, 'attention_heads'):
        print(f"Attention heads: {model.attention_heads}")
    
    # Tokenizer info
    print(f"\n🔤 Alphabet Details:")
    print(f"  Padding idx: {alphabet.padding_idx}")
    print(f"  Mask idx: {alphabet.mask_idx}")
    print(f"  CLS idx: {alphabet.cls_idx}")
    print(f"  EOS idx: {alphabet.eos_idx}")
    print(f"  Prepend BOS: {alphabet.prepend_bos}")
    print(f"  Append EOS: {alphabet.append_eos}")
    
    # Show some tokens
    print(f"  First 10 tokens: {[alphabet.get_tok(i) for i in range(min(10, len(alphabet)))]}")
    
    return True

def analyze_state_dict(model):
    """Analyze the model's state dictionary to understand parameter structure."""
    print(f"\n🔧 Parameter Structure Analysis")
    print("=" * 50)
    
    state_dict = model.state_dict()
    total_params = sum(p.numel() for p in state_dict.values())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Total parameter tensors: {len(state_dict)}")
    
    # Group parameters by component
    param_groups = {
        'embeddings': [],
        'encoder_layers': [],
        'layer_norms': [],
        'attention': [],
        'ffn': [],
        'output': [],
        'other': []
    }
    
    for name, param in state_dict.items():
        if 'embed' in name:
            param_groups['embeddings'].append((name, param.shape))
        elif 'layers.' in name:
            param_groups['encoder_layers'].append((name, param.shape))
            if 'self_attn' in name:
                param_groups['attention'].append((name, param.shape))
            elif any(x in name for x in ['fc1', 'fc2']):
                param_groups['ffn'].append((name, param.shape))
            elif 'layer_norm' in name or 'LayerNorm' in name:
                param_groups['layer_norms'].append((name, param.shape))
        elif 'lm_head' in name or 'contact_head' in name:
            param_groups['output'].append((name, param.shape))
        else:
            param_groups['other'].append((name, param.shape))
    
    # Print analysis
    for group, params in param_groups.items():
        if params:
            print(f"\n{group.upper()} ({len(params)} parameters):")
            for name, shape in params[:5]:  # Show first 5
                print(f"  {name}: {shape}")
            if len(params) > 5:
                print(f"  ... and {len(params) - 5} more")
    
    return state_dict

def test_model_inference(model, alphabet):
    """Test inference with the PyTorch model."""
    print(f"\n🧪 Testing PyTorch Inference")
    print("=" * 50)
    
    # Create a test sequence
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    print(f"Test sequence: {sequence[:50]}...")
    print(f"Sequence length: {len(sequence)}")
    
    # Tokenize
    batch_converter = alphabet.get_batch_converter()
    data = [("test_protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    print(f"Tokenized shape: {batch_tokens.shape}")
    print(f"Token range: {batch_tokens.min().item()} to {batch_tokens.max().item()}")
    
    # Forward pass
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=True)
    
    print(f"✅ Inference successful!")
    
    # Analyze outputs
    if "representations" in results:
        for layer, repr_tensor in results["representations"].items():
            print(f"  Layer {layer} representations: {repr_tensor.shape}")
    
    if "logits" in results:
        print(f"  Logits: {results['logits'].shape}")
    
    if "contacts" in results:
        print(f"  Contacts: {results['contacts'].shape}")
    
    return results

def compare_with_our_config(model):
    """Compare PyTorch model with our MLX config."""
    print(f"\n⚖️ Comparing with our MLX Config")
    print("=" * 50)
    
    # Import our config
    sys.path.append('.')
    from esm_mlx.config import ESM2Config
    
    # Create config that should match PyTorch model
    our_config = ESM2Config(
        vocab_size=len(model.alphabet) if hasattr(model, 'alphabet') else 33,
        hidden_size=model.embed_dim,
        num_hidden_layers=model.num_layers,
        num_attention_heads=model.attention_heads,
        intermediate_size=model.embed_dim * 4,  # Standard transformer ratio
        attention_head_dim=model.embed_dim // model.attention_heads
    )
    
    print(f"PyTorch model:")
    print(f"  Vocab size: {len(model.alphabet) if hasattr(model, 'alphabet') else 'unknown'}")
    print(f"  Hidden size: {model.embed_dim}")
    print(f"  Layers: {model.num_layers}")
    print(f"  Attention heads: {model.attention_heads}")
    
    print(f"\nOur MLX config:")
    print(f"  Vocab size: {our_config.vocab_size}")
    print(f"  Hidden size: {our_config.hidden_size}")
    print(f"  Layers: {our_config.num_hidden_layers}")
    print(f"  Attention heads: {our_config.num_attention_heads}")
    print(f"  Head dim: {our_config.attention_head_dim}")
    
    # Check if they match
    matches = (
        our_config.hidden_size == model.embed_dim and
        our_config.num_hidden_layers == model.num_layers and
        our_config.num_attention_heads == model.attention_heads
    )
    
    print(f"\n{'✅' if matches else '❌'} Configurations {'match!' if matches else 'do not match!'}")
    
    return our_config

def main():
    """Main analysis function."""
    print("🔬 PyTorch ESM2 Model Analysis")
    print("=" * 60)
    
    # Load model
    model, alphabet = load_smallest_esm2()
    if model is None:
        return False
    
    try:
        # Analyze structure
        analyze_model_structure(model, alphabet)
        
        # Analyze parameters
        state_dict = analyze_state_dict(model)
        
        # Test inference
        results = test_model_inference(model, alphabet)
        
        # Compare configs
        our_config = compare_with_our_config(model)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"\nKey findings:")
        print(f"- Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"- Hidden dimension: {model.embed_dim}")
        print(f"- Number of layers: {model.num_layers}")
        print(f"- Attention heads: {model.attention_heads}")
        print(f"- Vocabulary size: {len(alphabet)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)