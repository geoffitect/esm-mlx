#!/usr/bin/env python3

"""
Test ESMFold-MLX installation and basic functionality.
This script validates that the package is properly installed and working.
"""

import sys
import time

def test_basic_imports():
    """Test that all core modules can be imported."""
    
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        import esm_mlx
        print(f"  ✅ esm_mlx v{esm_mlx.__version__}")
        
        from esm_mlx import ESMFold, fold_protein
        print("  ✅ High-level API")
        
        from esm_mlx import ESM2MLX, ESMFoldMLX
        print("  ✅ Core models")
        
        from esm_mlx import quantize_esmfold_model
        print("  ✅ Quantization")
        
        import mlx.core as mx
        print(f"  ✅ MLX framework")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic protein folding functionality."""
    
    print("\n🧪 Testing basic functionality...")
    
    try:
        from esm_mlx import fold_protein
        
        # Test simple folding
        sequence = "MKTAYIAK"  # Short test sequence
        
        start_time = time.time()
        result = fold_protein(sequence, model_size="small")
        fold_time = time.time() - start_time
        
        print(f"  ✅ Folded {len(sequence)} residue protein in {fold_time:.3f}s")
        print(f"  📊 Confidence: {result.mean_confidence:.3f}")
        print(f"  🏆 Quality: {result.structure_quality}")
        print(f"  📐 Coordinates shape: {result.coordinates.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Folding failed: {e}")
        return False


def test_quantization():
    """Test quantization functionality."""
    
    print("\n🔥 Testing quantization...")
    
    try:
        from esm_mlx import ESMFold
        
        # Test quantized model
        model = ESMFold.from_pretrained("small", use_quantization=True)
        
        sequence = "MVLSPADK"
        result = model.fold(sequence)
        
        print(f"  ✅ Quantized folding successful")
        print(f"  📊 Confidence: {result.mean_confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quantization failed: {e}")
        return False


def test_cli_availability():
    """Test command-line interface availability."""
    
    print("\n⚡ Testing CLI availability...")
    
    try:
        from esm_mlx.cli import fold_command, benchmark_command, convert_command
        print("  ✅ CLI commands available")
        
        # Test that entry points would work
        import pkg_resources
        try:
            for entry_point in pkg_resources.iter_entry_points('console_scripts'):
                if entry_point.name.startswith('esm-'):
                    print(f"  ✅ Entry point: {entry_point.name}")
        except:
            print("  ⚠️  Entry points not installed (expected in development)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False


def test_package_metadata():
    """Test package metadata and structure."""
    
    print("\n📦 Testing package metadata...")
    
    try:
        import esm_mlx
        
        # Check version
        version = getattr(esm_mlx, '__version__', 'unknown')
        print(f"  ✅ Version: {version}")
        
        # Check key modules exist
        expected_modules = [
            'esm_mlx.api',
            'esm_mlx.esmfold_mlx', 
            'esm_mlx.esm2_mlx',
            'esm_mlx.quantization',
            'esm_mlx.config',
            'esm_mlx.cli'
        ]
        
        for module_name in expected_modules:
            try:
                __import__(module_name)
                print(f"  ✅ {module_name}")
            except ImportError:
                print(f"  ❌ {module_name} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Metadata test failed: {e}")
        return False


def main():
    """Run all installation tests."""
    
    print("🚀 ESMFold-MLX Installation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Functionality", test_basic_functionality), 
        ("Quantization", test_quantization),
        ("CLI Availability", test_cli_availability),
        ("Package Metadata", test_package_metadata)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed! ESMFold-MLX is ready for action! 🚀")
        return 0
    else:
        print("⚠️  Some tests failed. Check installation and dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())