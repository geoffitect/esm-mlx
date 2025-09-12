# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MLX implementation of ESM models for protein structure prediction.

ESMFold-MLX: Lightning-fast protein folding on Apple Silicon
============================================================

Quick Start:
    ```python
    from esm_mlx import ESMFold, fold_protein
    
    # Simple folding
    result = fold_protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLI")
    
    # Advanced usage
    model = ESMFold.from_pretrained("medium", use_quantization=True)
    result = model.fold("PROTEIN_SEQUENCE_HERE")
    result.save_pdb("structure.pdb")
    ```

Features:
- 🚀 2-4x speedup on Apple Silicon
- 🔥 4-bit/8-bit quantization support  
- 📱 Native MLX implementation
- 🧬 Full protein structure prediction pipeline
- 📊 Confidence scoring and validation
"""

# High-level API (recommended for most users)
from .api import ESMFold, FoldingResult, fold_protein, fold_and_save

# Core implementation (for advanced users)
from .esm2_mlx import ESM2MLX
from .config import ESM2Config
from .esmfold_mlx import ESMFoldMLX, ESMFoldConfig

# Components (for researchers and developers)
from .ipa import InvariantPointAttention, IPABlock
from .triangular_attention import TriangularSelfAttentionBlock, TriangularMultiplicativeUpdate
from .structure_head import StructurePredictionHead, BackboneUpdate
from .confidence_head import ConfidenceHead, LDDTHead, CategoricalMixture

# Quantization and optimization
from .quantization import (
    quantize_esmfold_model,
    get_quantization_stats,
    benchmark_quantized_model,
    save_quantized_model,
    load_quantized_model
)

__version__ = "1.0.0"

__all__ = [
    # High-level API
    "ESMFold", 
    "FoldingResult",
    "fold_protein",
    "fold_and_save",
    
    # Core implementation
    "ESM2MLX", 
    "ESM2Config",
    "ESMFoldMLX",
    "ESMFoldConfig", 
    
    # Components
    "InvariantPointAttention",
    "IPABlock",
    "TriangularSelfAttentionBlock",
    "TriangularMultiplicativeUpdate",
    "StructurePredictionHead",
    "BackboneUpdate",
    "ConfidenceHead",
    "LDDTHead",
    "CategoricalMixture",
    
    # Quantization
    "quantize_esmfold_model",
    "get_quantization_stats", 
    "benchmark_quantized_model",
    "save_quantized_model",
    "load_quantized_model"
]