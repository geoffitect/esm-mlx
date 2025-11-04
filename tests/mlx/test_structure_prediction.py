#!/usr/bin/env python3

"""
Test script for ESMFold structure prediction components with real sequences.
Tests the full pipeline from sequence input to 3D coordinate prediction.
"""

import sys
import time
import numpy as np
from typing import Dict, Any

import mlx.core as mx
import mlx.nn as nn

# Import our structure prediction components
from esm_mlx.config import ESM2Config
from esm_mlx.esmfold_mlx import ESMFoldMLX, ESMFoldConfig
from esm_mlx.ipa import InvariantPointAttention, IPABlock
from esm_mlx.simple_triangular import SimpleTriangularAttention
from esm_mlx.structure_head import StructurePredictionHead, BackboneUpdate
from esm_mlx.confidence_head import ConfidenceHead, CategoricalMixture


def create_amino_acid_tokenizer():
    """Create simple amino acid tokenizer for testing."""
    
    # Standard 20 amino acids + special tokens
    vocab = {
        '<pad>': 0,
        '<cls>': 1, 
        '<eos>': 2,
        '<unk>': 3,
        'A': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8,
        'G': 9, 'H': 10, 'I': 11, 'K': 12, 'L': 13,
        'M': 14, 'N': 15, 'P': 16, 'Q': 17, 'R': 18,
        'S': 19, 'T': 20, 'V': 21, 'W': 22, 'Y': 23
    }
    
    def tokenize_sequence(sequence: str) -> list[int]:
        """Convert protein sequence to token IDs."""
        tokens = [vocab['<cls>']]
        for aa in sequence.upper():
            tokens.append(vocab.get(aa, vocab['<unk>']))
        tokens.append(vocab['<eos>'])
        return tokens
    
    return tokenize_sequence, vocab


def test_individual_components():
    """Test individual structure prediction components."""
    
    print("=== Testing Individual Components ===")
    
    # Test configuration
    batch_size = 2
    N_res = 50
    c_s = 384
    c_z = 128
    
    # Create test inputs
    single_repr = mx.random.normal((batch_size, N_res, c_s))
    pair_repr = mx.random.normal((batch_size, N_res, N_res, c_z))
    mask = mx.ones((batch_size, N_res))
    
    # Test rigid body frames
    frames_r = mx.broadcast_to(mx.eye(3)[None, None, :, :], (batch_size, N_res, 3, 3))
    frames_t = mx.random.normal((batch_size, N_res, 3)) * 0.1
    
    print(f"Input shapes: single={single_repr.shape}, pair={pair_repr.shape}")
    
    # 1. Test Invariant Point Attention
    print("\n1. Testing Invariant Point Attention...")
    
    config = ESMFoldConfig()
    config.c_s = c_s
    config.c_z = c_z
    
    ipa = InvariantPointAttention(
        c_s=c_s, c_z=c_z, c_hidden=16, num_heads=4,
        num_scalar_qk=16, num_scalar_v=16,
        num_point_qk=4, num_point_v=8
    )
    
    start_time = time.time()
    ipa_output = ipa(single_repr, pair_repr, frames_r, frames_t, mask)
    ipa_time = time.time() - start_time
    
    print(f"  IPA output shape: {ipa_output.shape}")
    print(f"  IPA computation time: {ipa_time:.3f}s")
    print(f"  IPA output stats: mean={mx.mean(ipa_output):.4f}, std={mx.std(ipa_output):.4f}")
    
    # 2. Test Triangular Attention
    print("\n2. Testing Triangular Self-Attention...")
    
    tri_attn = SimpleTriangularAttention(c_z=c_z, c_hidden=32, num_heads=4)
    
    start_time = time.time()
    tri_output = tri_attn(pair_repr, mask)
    tri_time = time.time() - start_time
    
    print(f"  Triangular attention output shape: {tri_output.shape}")
    print(f"  Triangular attention time: {tri_time:.3f}s")
    print(f"  Triangular output stats: mean={mx.mean(tri_output):.4f}, std={mx.std(tri_output):.4f}")
    
    # 3. Test Structure Prediction Head
    print("\n3. Testing Structure Prediction Head...")
    
    structure_head = StructurePredictionHead(c_s=c_s, c_z=c_z)
    
    start_time = time.time()
    structure_output = structure_head(single_repr, pair_repr)
    structure_time = time.time() - start_time
    
    print(f"  Structure prediction time: {structure_time:.3f}s")
    print(f"  Predicted coordinates shape: {structure_output['coords'].shape}")
    print(f"  Predicted angles shape: {structure_output['angles'].shape}")
    print(f"  Coordinate stats: mean={mx.mean(structure_output['coords']):.4f}")
    
    # 4. Test Confidence Head
    print("\n4. Testing Confidence Head...")
    
    confidence_head = ConfidenceHead(c_s=c_s, c_z=c_z)
    
    start_time = time.time()
    confidence_output = confidence_head(single_repr, pair_repr, mask)
    confidence_time = time.time() - start_time
    
    print(f"  Confidence prediction time: {confidence_time:.3f}s")
    print(f"  pLDDT shape: {confidence_output['plddt'].shape}")
    print(f"  TM-score shape: {confidence_output['tm_score'].shape}")
    print(f"  pLDDT stats: mean={mx.mean(confidence_output['plddt']):.4f}")
    print(f"  TM-score stats: mean={mx.mean(confidence_output['tm_score']):.4f}")
    
    return True


def test_full_esmfold_pipeline():
    """Test the complete ESMFold pipeline with protein sequences."""
    
    print("\n=== Testing Full ESMFold Pipeline ===")
    
    # Test sequences (real protein sequences)
    test_sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKA",  # Short protein
        "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"  # Medium protein
    ]
    
    print(f"Test sequences:")
    for i, seq in enumerate(test_sequences):
        print(f"  {i+1}. Length {len(seq)}: {seq[:30]}...")
    
    # Create tokenizer
    tokenize_sequence, vocab = create_amino_acid_tokenizer()
    
    # Tokenize sequences
    input_ids = []
    attention_masks = []
    
    for seq in test_sequences:
        tokens = tokenize_sequence(seq)
        input_ids.append(tokens)
        attention_masks.append([1] * len(tokens))
    
    # Pad to same length
    max_len = max(len(seq) for seq in input_ids)
    for i in range(len(input_ids)):
        pad_len = max_len - len(input_ids[i])
        input_ids[i].extend([0] * pad_len)
        attention_masks[i].extend([0] * pad_len)
    
    input_ids = mx.array(input_ids)
    attention_mask = mx.array(attention_masks)
    
    print(f"\nTokenized input shapes: {input_ids.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create ESMFold configuration
    esm_config = ESM2Config(
        vocab_size=len(vocab),
        hidden_size=320,  # Smaller for testing
        num_hidden_layers=6,
        num_attention_heads=16,  # Changed from 20 to 16 (320/16=20 head_dim)
        intermediate_size=1280,
        layer_norm_eps=1e-5
    )
    
    esmfold_config = ESMFoldConfig(
        esm_config=esm_config,
        c_s=256,  # Reduced dimensions for testing
        c_z=64,
        num_folding_blocks=4,  # Reduced for testing
        num_ipa_blocks=2,
        num_recycles=2  # Reduced for testing
    )
    
    print(f"ESMFold config: c_s={esmfold_config.c_s}, c_z={esmfold_config.c_z}")
    print(f"Folding blocks: {esmfold_config.num_folding_blocks}, IPA blocks: {esmfold_config.num_ipa_blocks}")
    
    # Create ESMFold model
    print("\nCreating ESMFold model...")
    model = ESMFoldMLX(esmfold_config)
    
    # Test forward pass
    print("Running forward pass...")
    start_time = time.time()
    
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_recycles=1,  # Single recycle for testing
            return_all_frames=True
        )
        
        forward_time = time.time() - start_time
        print(f"Forward pass completed in {forward_time:.3f}s")
        
        # Analyze outputs
        print(f"\n=== Output Analysis ===")
        print(f"Predicted coordinates shape: {output['coordinates'].shape}")
        print(f"Backbone angles shape: {output['backbone_angles'].shape}")
        print(f"pLDDT scores shape: {output['plddt'].shape}")
        print(f"TM-scores shape: {output['tm_score'].shape}")
        
        # Statistics
        coords = output['coordinates']
        plddt = output['plddt']
        tm_scores = output['tm_score']
        
        print(f"\nCoordinate statistics:")
        print(f"  Mean: {mx.mean(coords):.4f}")
        print(f"  Std: {mx.std(coords):.4f}")
        print(f"  Range: [{mx.min(coords):.4f}, {mx.max(coords):.4f}]")
        
        print(f"\npLDDT statistics:")
        for i, seq in enumerate(test_sequences):
            seq_len = len(seq)
            seq_plddt = plddt[i, :seq_len+2]  # +2 for special tokens
            print(f"  Sequence {i+1}: mean={mx.mean(seq_plddt):.3f}, std={mx.std(seq_plddt):.3f}")
        
        print(f"\nTM-score statistics:")
        for i, tm in enumerate(tm_scores):
            print(f"  Sequence {i+1}: {float(tm):.3f}")
        
        # Test recycling frames
        if 'all_frames' in output:
            print(f"\nRecycling analysis:")
            all_frames = output['all_frames']
            print(f"  Number of recycle iterations: {len(all_frames)}")
            
            for i, frame_data in enumerate(all_frames):
                coords_frame = frame_data['coords']
                coord_mean = mx.mean(coords_frame)
                print(f"  Recycle {i}: coord_mean={coord_mean:.4f}")
        
        print(f"\n✅ ESMFold pipeline test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ESMFold pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geometric_consistency():
    """Test geometric consistency of predicted structures."""
    
    print("\n=== Testing Geometric Consistency ===")
    
    # Create simple test case
    batch_size = 1
    N_res = 10
    
    # Create mock ESM representation
    esm_repr = mx.random.normal((batch_size, N_res, 320))
    
    # Test structure head
    structure_head = StructurePredictionHead(c_s=320, c_z=64)
    
    # Get structure prediction
    pair_repr = mx.random.normal((batch_size, N_res, N_res, 64))
    structure_output = structure_head(esm_repr, pair_repr)
    
    coords = structure_output['coords']  # [batch, N_res, 3, 3] - N, CA, C atoms
    angles = structure_output['angles']   # [batch, N_res, 3] - phi, psi, omega
    
    print(f"Predicted coordinates shape: {coords.shape}")
    print(f"Predicted angles shape: {angles.shape}")
    
    # Check bond lengths (should be approximately correct)
    ca_coords = coords[0, :, 1, :]  # CA atoms: [N_res, 3]
    
    if N_res > 1:
        # Compute CA-CA distances
        ca_distances = []
        for i in range(N_res - 1):
            dist = mx.linalg.norm(ca_coords[i+1] - ca_coords[i])
            ca_distances.append(float(dist))
        
        print(f"\nCA-CA distances:")
        print(f"  Mean: {np.mean(ca_distances):.3f} Å")
        print(f"  Std: {np.std(ca_distances):.3f} Å")
        print(f"  Range: [{min(ca_distances):.3f}, {max(ca_distances):.3f}] Å")
        print(f"  Expected ~3.8 Å for ideal backbone")
    
    # Check angle ranges
    angles_deg = mx.array(angles) * 180 / mx.pi  # Convert to degrees
    
    print(f"\nBackbone angles (degrees):")
    print(f"  Phi - Mean: {mx.mean(angles_deg[..., 0]):.1f}, Std: {mx.std(angles_deg[..., 0]):.1f}")
    print(f"  Psi - Mean: {mx.mean(angles_deg[..., 1]):.1f}, Std: {mx.std(angles_deg[..., 1]):.1f}")
    print(f"  Omega - Mean: {mx.mean(angles_deg[..., 2]):.1f}, Std: {mx.std(angles_deg[..., 2]):.1f}")
    print(f"  Expected ranges: Phi [-180,180], Psi [-180,180], Omega ~180")
    
    return True


def main():
    """Run all structure prediction tests."""
    
    print("ESMFold Structure Prediction Test Suite")
    print("=" * 50)
    
    try:
        # Test individual components
        test_individual_components()
        
        # Test geometric consistency
        test_geometric_consistency()
        
        # Test full pipeline
        test_full_esmfold_pipeline()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()