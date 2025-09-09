#!/usr/bin/env python3

"""
Basic usage example for ESM2-MLX.

This example demonstrates how to use the MLX implementation of ESM2
for protein sequence analysis and structure prediction preparation.
"""

import sys
import os
import numpy as np

# Add parent directory to path to import esm_mlx
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mlx.core as mx
from esm_mlx import ESM2MLX, ESM2Config


def create_sample_protein_data():
    """Create sample protein sequences for testing."""
    
    # Some example protein sequences (simplified tokenization for demo)
    proteins = [
        {
            "name": "Small peptide",
            "sequence": "MKTVRQERLKSIVRILERSKEPVSG",
            "description": "Short example peptide"
        },
        {
            "name": "Medium protein",  
            "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "description": "Medium-sized protein example"
        }
    ]
    
    return proteins


def simple_tokenize(sequence: str, vocab_size: int = 33) -> mx.array:
    """
    Simple tokenization for demo purposes.
    
    In a real implementation, you would use the ESM tokenizer
    that maps amino acids to specific token IDs.
    """
    # For demo, map each character to a random token ID
    # In practice, you'd have a proper amino acid -> token mapping
    np.random.seed(42)  # For reproducible results
    tokens = []
    
    for char in sequence:
        # Map common amino acids to consistent IDs (simplified)
        amino_acid_map = {
            'A': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10,
            'I': 11, 'K': 12, 'L': 13, 'M': 14, 'N': 15, 'P': 16, 
            'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'V': 21, 'W': 22, 'Y': 23
        }
        token_id = amino_acid_map.get(char, 3)  # Default to 3 for unknown
        tokens.append(token_id)
    
    return mx.array(tokens).reshape(1, -1)  # Add batch dimension


def run_protein_analysis():
    """Run protein analysis using ESM2-MLX."""
    
    print("🧬 ESM2-MLX Protein Analysis Example")
    print("=" * 50)
    
    # Create a small model for demo
    config = ESM2Config(
        vocab_size=33,
        hidden_size=256,      # Smaller for demo
        num_hidden_layers=6,   # Fewer layers for demo  
        num_attention_heads=8,
        intermediate_size=1024,
        attention_head_dim=32
    )
    
    print(f"📋 Model Configuration:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    
    # Create model
    print(f"\n🔧 Creating ESM2-MLX model...")
    model = ESM2MLX(config)
    print("✅ Model created successfully!")
    
    # Get sample proteins
    proteins = create_sample_protein_data()
    
    # Analyze each protein
    for protein in proteins:
        print(f"\n🔬 Analyzing: {protein['name']}")
        print(f"  Sequence: {protein['sequence'][:50]}{'...' if len(protein['sequence']) > 50 else ''}")
        print(f"  Length: {len(protein['sequence'])} residues")
        
        # Tokenize sequence
        input_ids = simple_tokenize(protein['sequence'], config.vocab_size)
        print(f"  Tokenized shape: {input_ids.shape}")
        
        # Run inference
        print("  Running inference...")
        
        # Get basic outputs
        outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
        
        print(f"  ✅ Inference completed!")
        print(f"    - Prediction scores: {outputs['prediction_scores'].shape}")
        print(f"    - Hidden states: {len(outputs['hidden_states'])} layers")
        print(f"    - Attention layers: {len(outputs['attentions'])}")
        
        # Get sequence embeddings
        embeddings = model.get_sequence_embeddings(input_ids)
        print(f"    - Sequence embedding: {embeddings.shape}")
        
        # Contact prediction (if available)
        contact_outputs = model(input_ids, return_contacts=True, output_attentions=True)
        if contact_outputs['contacts'] is not None:
            contacts = contact_outputs['contacts']
            print(f"    - Contact predictions: {contacts.shape}")
            
            # Show some contact statistics
            contact_probs = mx.softmax(contacts, axis=-1)
            max_contact_prob = mx.max(contact_probs)
            avg_contact_prob = mx.mean(contact_probs)
            print(f"    - Max contact probability: {float(max_contact_prob):.3f}")
            print(f"    - Avg contact probability: {float(avg_contact_prob):.3f}")


def demonstrate_embeddings():
    """Demonstrate embedding extraction and comparison."""
    
    print(f"\n🎯 Embedding Analysis")
    print("-" * 30)
    
    # Create model with consistent dimensions
    config = ESM2Config(
        hidden_size=128, 
        num_hidden_layers=3, 
        num_attention_heads=4,
        attention_head_dim=32  # 128 // 4
    )
    model = ESM2MLX(config)
    
    # Create two similar sequences and one different
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSG",  # Reference
        "MKTVRQERLKSIVRILERSKEPVSG",  # Identical 
        "AAAAAAAAAAAAAAAAAAAAAAAAAA",  # Very different
    ]
    
    embeddings = []
    
    for i, seq in enumerate(sequences):
        tokens = simple_tokenize(seq, config.vocab_size)
        embedding = model.get_sequence_embeddings(tokens)
        embeddings.append(embedding)
        print(f"Sequence {i+1} embedding shape: {embedding.shape}")
    
    # Calculate similarities (simplified cosine similarity)
    def cosine_similarity(a, b):
        a_norm = a / mx.linalg.norm(a, axis=-1, keepdims=True)
        b_norm = b / mx.linalg.norm(b, axis=-1, keepdims=True)
        return mx.sum(a_norm * b_norm, axis=-1)
    
    print(f"\nEmbedding similarities:")
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  Seq {i+1} vs Seq {j+1}: {float(sim[0]):.3f}")


def main():
    """Main example function."""
    try:
        # Run basic analysis
        run_protein_analysis()
        
        # Demonstrate embeddings
        demonstrate_embeddings()
        
        print(f"\n🎉 Example completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Install fair-esm to access real ESM2 models and tokenizers")
        print(f"2. Convert PyTorch ESM2 weights using: python -m esm_mlx.convert_weights")
        print(f"3. Use converted weights for real protein analysis")
        print(f"4. Explore structure prediction with ESMFold components")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)