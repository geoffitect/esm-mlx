#!/usr/bin/env python3

"""Test the high-level ESMFold API."""

from esm_mlx import ESMFold, fold_protein, fold_and_save

def test_simple_api():
    """Test the simple fold_protein function."""
    
    print("🧪 Testing simple API...")
    
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLI"
    
    # Test simple folding
    result = fold_protein(sequence, model_size="small")
    
    print(f"✅ Simple folding result: {result}")
    print(f"📊 Confidence: {result.mean_confidence:.3f}")
    print(f"🏆 Quality: {result.structure_quality}")
    
    return result

def test_advanced_api():
    """Test the advanced ESMFold class."""
    
    print("\n🔬 Testing advanced API...")
    
    # Load model with quantization
    model = ESMFold.from_pretrained("small", use_quantization=True)
    
    # Fold protein
    sequence = "MVLSPADKTNVKAAW"
    result = model.fold(sequence)
    
    print(f"✅ Advanced folding result: {result}")
    print(f"📊 Confidence: {result.mean_confidence:.3f}")
    print(f"🔥 Coordinates shape: {result.coordinates.shape}")
    
    # Save to PDB
    result.save_pdb("test_structure.pdb")
    
    return result

def test_batch_folding():
    """Test batch folding."""
    
    print("\n📦 Testing batch folding...")
    
    model = ESMFold.from_pretrained("small")
    
    sequences = [
        "MKTAYIAK",
        "MVLSPADK",
        "GGEVQAPIL"
    ]
    
    results = model.fold_batch(sequences)
    
    for i, result in enumerate(results):
        print(f"  Sequence {i+1}: {result}")
    
    return results

def main():
    """Test all API functionality."""
    
    print("🔥 Testing ESMFold MLX High-Level API")
    print("=" * 40)
    
    try:
        # Test simple API
        test_simple_api()
        
        # Test advanced API  
        test_advanced_api()
        
        # Test batch folding
        test_batch_folding()
        
        print("\n" + "=" * 40)
        print("✅ All API tests passed!")
        
    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()