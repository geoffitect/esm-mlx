# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line interface for ESMFold-MLX."""

import argparse
import sys
from pathlib import Path

from .api import fold_protein, ESMFold
from .quantization import quantize_esmfold_model
import mlx.core as mx


def fold_command():
    """Command-line protein folding interface."""
    
    parser = argparse.ArgumentParser(
        description="ESMFold-MLX: Lightning-fast protein folding on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--sequence", "-s",
        type=str,
        required=True,
        help="Protein sequence to fold (single letter amino acid codes)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str,
        default="structure.pdb",
        help="Output PDB file path"
    )
    
    parser.add_argument(
        "--model-size", "-m",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size for folding"
    )
    
    parser.add_argument(
        "--quantized", "-q",
        action="store_true",
        help="Use 4-bit quantization for maximum speed"
    )
    
    parser.add_argument(
        "--quantization-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (4 or 8)"
    )
    
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycling iterations (uses model default if not specified)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🚀 ESMFold-MLX: Folding {len(args.sequence)} residue protein")
        print(f"📊 Model: {args.model_size}")
        print(f"🔥 Quantized: {args.quantized}")
    
    try:
        # Fold protein
        if args.quantized:
            model = ESMFold.from_pretrained(
                args.model_size, 
                use_quantization=True,
                quantization_bits=args.quantization_bits
            )
            result = model.fold(args.sequence, num_recycles=args.num_recycles)
        else:
            result = fold_protein(args.sequence, args.model_size)
        
        # Save result
        result.save_pdb(args.output)
        
        # Print summary
        print(f"✅ Folding complete!")
        print(f"📊 Confidence: {result.mean_confidence:.3f} ({result.structure_quality})")
        print(f"💾 Structure saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Folding failed: {e}")
        sys.exit(1)


def benchmark_command():
    """Command-line benchmarking interface."""
    
    parser = argparse.ArgumentParser(
        description="Benchmark ESMFold-MLX performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-size", "-m",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size to benchmark"
    )
    
    parser.add_argument(
        "--quantized", "-q",
        action="store_true",
        help="Benchmark quantized model"
    )
    
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[50, 100, 200],
        help="Sequence lengths to test"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for benchmark results"
    )
    
    args = parser.parse_args()
    
    print(f"🏁 Benchmarking ESMFold-MLX {args.model_size} model")
    
    try:
        # Run benchmarks
        from .quantization import benchmark_quantized_model
        import time
        import json
        
        results = {}
        
        for seq_len in args.sequence_lengths:
            print(f"📏 Testing {seq_len} residue sequence...")
            
            # Create test sequence
            test_sequence = "M" + "K" * (seq_len - 2) + "G"
            
            # Load model
            model = ESMFold.from_pretrained(args.model_size, use_quantization=args.quantized)
            
            # Benchmark
            times = []
            for i in range(args.num_runs):
                start = time.time()
                result = model.fold(test_sequence)
                mx.eval(result.coordinates)
                end = time.time()
                times.append(end - start)
                print(f"  Run {i+1}: {times[-1]:.4f}s")
            
            import numpy as np
            results[seq_len] = {
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "time_per_residue": float(np.mean(times)) / seq_len
            }
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Benchmark complete! Results saved to {args.output}")
        
        # Print summary
        print(f"\n📊 Performance Summary:")
        for seq_len, metrics in results.items():
            print(f"  {seq_len} residues: {metrics['mean_time']:.4f}s "
                  f"({metrics['time_per_residue']*1000:.1f}ms/residue)")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


def convert_command():
    """Command-line weight conversion interface."""
    
    parser = argparse.ArgumentParser(
        description="Convert PyTorch ESMFold weights to MLX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input PyTorch checkpoint file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True, 
        help="Output MLX weights file (.npz)"
    )
    
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate converted weights"
    )
    
    args = parser.parse_args()
    
    print(f"🔄 Converting PyTorch weights to MLX format")
    print(f"📂 Input: {args.input}")
    print(f"💾 Output: {args.output}")
    
    try:
        # Import conversion function
        from .weight_converter import convert_esmfold_checkpoint, validate_converted_weights
        from .esmfold_mlx import ESMFoldConfig
        from .config import ESM2Config
        
        # Default config for conversion
        config = ESMFoldConfig(
            esm_config=ESM2Config(
                vocab_size=33,
                hidden_size=1280,
                num_hidden_layers=33,
                num_attention_heads=20
            )
        )
        
        # Convert weights
        output_path = convert_esmfold_checkpoint(args.input, args.output, config)
        
        # Validate if requested
        if args.validate:
            print("🧪 Validating converted weights...")
            validation_results = validate_converted_weights(args.input, output_path, config)
            
            print("📊 Validation Results:")
            for metric, value in validation_results.items():
                status = "✅" if value else "❌"
                print(f"  {status} {metric}: {value}")
        
        print(f"✅ Conversion complete!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Simple CLI dispatcher
    if len(sys.argv) > 1:
        command = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove command from args
        
        if command == "fold":
            fold_command()
        elif command == "benchmark":
            benchmark_command()
        elif command == "convert":
            convert_command()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: fold, benchmark, convert")
            sys.exit(1)
    else:
        print("ESMFold-MLX CLI")
        print("Commands: fold, benchmark, convert")
        print("Use 'python -m esm_mlx.cli <command> --help' for help")