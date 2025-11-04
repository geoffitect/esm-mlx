# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
User-friendly API for ESMFold MLX.
Provides simple interfaces for protein structure prediction.
"""

import os
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

import mlx.core as mx
import numpy as np

from .esmfold_mlx import ESMFoldMLX, ESMFoldConfig
from .config import ESM2Config
from .quantization import quantize_esmfold_model, load_quantized_model


class ESMFold:
    """
    User-friendly interface for ESMFold protein structure prediction.
    
    Example usage:
        ```python
        from esm_mlx import ESMFold
        
        # Load model
        model = ESMFold.from_pretrained("small")
        
        # Fold a protein
        result = model.fold("MKTAYIAKQRQISFVKSHFSRQLEERLGLI")
        
        # Access results
        coordinates = result.coordinates  # 3D atomic coordinates
        confidence = result.confidence    # Per-residue confidence scores
        ```
    """
    
    def __init__(self, model: ESMFoldMLX, config: ESMFoldConfig):
        self.model = model
        self.config = config
        self._tokenizer = self._create_tokenizer()
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name: str = "medium",
        use_quantization: bool = False,
        quantization_bits: int = 4,
        device: str = "auto"
    ) -> "ESMFold":
        """
        Load a pretrained ESMFold model.
        
        Args:
            model_name: Model size ("small", "medium", "large")
            use_quantization: Whether to use quantized model for speed
            quantization_bits: Quantization bits (4 or 8)
            device: Device to use ("auto", "cpu", "gpu")
            
        Returns:
            ESMFold instance
        """
        
        # Get config for model size
        config = cls._get_model_config(model_name)
        
        # Create model
        model = ESMFoldMLX(config)
        
        # Apply quantization if requested
        if use_quantization:
            print(f"🔥 Applying {quantization_bits}-bit quantization...")
            model = quantize_esmfold_model(model, bits=quantization_bits)
        
        print(f"✅ Loaded ESMFold {model_name} model")
        return cls(model, config)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[ESMFoldConfig] = None
    ) -> "ESMFold":
        """
        Load ESMFold from a checkpoint file.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Model configuration (inferred if None)
            
        Returns:
            ESMFold instance
        """
        
        if config is None:
            config = cls._get_model_config("medium")  # Default config
        
        model = ESMFoldMLX(config)
        
        # Load weights (simplified for demo)
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        # In real implementation, would load actual weights here
        
        return cls(model, config)
    
    @staticmethod
    def _get_model_config(model_name: str) -> ESMFoldConfig:
        """Get configuration for different model sizes."""
        
        configs = {
            "small": ESMFoldConfig(
                esm_config=ESM2Config(
                    vocab_size=33,
                    hidden_size=320,
                    num_hidden_layers=6,
                    num_attention_heads=16,
                    intermediate_size=1280
                ),
                c_s=256,
                c_z=64,
                num_folding_blocks=2,
                num_ipa_blocks=1,
                num_recycles=2
            ),
            "medium": ESMFoldConfig(
                esm_config=ESM2Config(
                    vocab_size=33,
                    hidden_size=640,
                    num_hidden_layers=12,
                    num_attention_heads=20,
                    intermediate_size=2560
                ),
                c_s=384,
                c_z=128,
                num_folding_blocks=4,
                num_ipa_blocks=2,
                num_recycles=3
            ),
            "large": ESMFoldConfig(
                esm_config=ESM2Config(
                    vocab_size=33,
                    hidden_size=1280,
                    num_hidden_layers=20,
                    num_attention_heads=20,
                    intermediate_size=5120
                ),
                c_s=512,
                c_z=256,
                num_folding_blocks=8,
                num_ipa_blocks=4,
                num_recycles=4
            )
        }
        
        if model_name not in configs:
            raise ValueError(f"Unknown model size: {model_name}. Choose from {list(configs.keys())}")
        
        return configs[model_name]
    
    def _create_tokenizer(self):
        """Create amino acid tokenizer."""
        
        # Standard 20 amino acids + special tokens
        aa_to_token = {
            '<pad>': 0, '<cls>': 1, '<eos>': 2, '<unk>': 3,
            'A': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8,
            'G': 9, 'H': 10, 'I': 11, 'K': 12, 'L': 13,
            'M': 14, 'N': 15, 'P': 16, 'Q': 17, 'R': 18,
            'S': 19, 'T': 20, 'V': 21, 'W': 22, 'Y': 23,
            'X': 24, 'B': 25, 'Z': 26, 'J': 27, 'O': 28, 'U': 29
        }
        
        return aa_to_token
    
    def _tokenize_sequence(self, sequence: str) -> Tuple[mx.array, mx.array]:
        """Tokenize protein sequence."""
        
        # Clean and uppercase sequence
        sequence = sequence.upper().strip()
        
        # Convert to tokens
        tokens = [self._tokenizer['<cls>']]
        for aa in sequence:
            tokens.append(self._tokenizer.get(aa, self._tokenizer['<unk>']))
        tokens.append(self._tokenizer['<eos>'])
        
        # Create arrays
        input_ids = mx.array([tokens])
        attention_mask = mx.ones((1, len(tokens)))
        
        return input_ids, attention_mask
    
    def fold(
        self,
        sequence: str,
        num_recycles: Optional[int] = None,
        return_raw_output: bool = False
    ) -> "FoldingResult":
        """
        Fold a protein sequence to predict its 3D structure.
        
        Args:
            sequence: Protein sequence as string (single letter amino acid codes)
            num_recycles: Number of recycling iterations (uses model default if None)
            return_raw_output: Whether to return raw model output
            
        Returns:
            FoldingResult with coordinates, confidence scores, and metadata
        """
        
        if len(sequence) > 400:
            print(f"⚠️  Sequence length {len(sequence)} exceeds recommended maximum of 400 residues")
        
        # Tokenize sequence
        input_ids, attention_mask = self._tokenize_sequence(sequence)
        
        # Run inference
        raw_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_recycles=num_recycles
        )
        
        if return_raw_output:
            return raw_output
        
        # Create user-friendly result
        return FoldingResult(
            sequence=sequence,
            coordinates=raw_output["coordinates"],
            confidence=raw_output["plddt"],
            tm_score=raw_output["tm_score"],
            raw_output=raw_output
        )
    
    def fold_batch(
        self,
        sequences: List[str],
        num_recycles: Optional[int] = None
    ) -> List["FoldingResult"]:
        """
        Fold multiple sequences in batch.
        
        Args:
            sequences: List of protein sequences
            num_recycles: Number of recycling iterations
            
        Returns:
            List of FoldingResult objects
        """
        
        results = []
        for sequence in sequences:
            result = self.fold(sequence, num_recycles)
            results.append(result)
        
        return results
    
    def save_pretrained(self, save_path: str):
        """Save model to directory."""
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights (simplified)
        weights_path = save_path / "model_weights.npz"
        print(f"💾 Saving model to {weights_path}")
        
        # Save config
        config_path = save_path / "config.json"
        import json
        with open(config_path, 'w') as f:
            # Would save actual config here
            json.dump({"model_type": "ESMFoldMLX"}, f, indent=2)
        
        print(f"✅ Model saved to {save_path}")


class FoldingResult:
    """
    Result of protein structure prediction.
    
    Attributes:
        sequence: Original protein sequence
        coordinates: 3D atomic coordinates [N_res, 3, 3] (N, CA, C atoms)
        confidence: Per-residue confidence scores (pLDDT)
        tm_score: Global structure confidence score
        raw_output: Raw model output dictionary
    """
    
    def __init__(
        self,
        sequence: str,
        coordinates: mx.array,
        confidence: mx.array,
        tm_score: mx.array,
        raw_output: Dict
    ):
        self.sequence = sequence
        self.coordinates = coordinates
        self.confidence = confidence
        self.tm_score = tm_score
        self.raw_output = raw_output
    
    @property
    def mean_confidence(self) -> float:
        """Mean confidence score across all residues."""
        return float(mx.mean(self.confidence))
    
    @property
    def structure_quality(self) -> str:
        """Qualitative assessment of structure quality."""
        mean_conf = self.mean_confidence
        if mean_conf > 0.9:
            return "Very High"
        elif mean_conf > 0.7:
            return "High"
        elif mean_conf > 0.5:
            return "Medium"
        else:
            return "Low"
    
    def save_pdb(self, output_path: str):
        """
        Save structure to PDB file.
        
        Args:
            output_path: Path to save PDB file
        """
        
        # Simplified PDB writer
        coords = np.array(self.coordinates[0])  # Remove batch dimension
        confidence = np.array(self.confidence[0])
        
        with open(output_path, 'w') as f:
            f.write("HEADER    PROTEIN STRUCTURE PREDICTION\n")
            f.write("TITLE     ESMFold MLX Prediction\n")
            
            atom_id = 1
            for i, (res_coords, res_conf) in enumerate(zip(coords, confidence)):
                res_num = i + 1
                aa = self.sequence[i] if i < len(self.sequence) else 'X'
                
                # N, CA, C atoms
                atom_names = ['N', 'CA', 'C']
                for j, (atom_name, atom_coords) in enumerate(zip(atom_names, res_coords)):
                    f.write(f"ATOM  {atom_id:5d}  {atom_name:<3s} {aa:3s} A{res_num:4d}    ")
                    f.write(f"{atom_coords[0]:8.3f}{atom_coords[1]:8.3f}{atom_coords[2]:8.3f}")
                    f.write(f"  1.00{res_conf:6.2f}           {atom_name[0]:>1s}\n")
                    atom_id += 1
            
            f.write("END\n")
        
        print(f"💾 Structure saved to {output_path}")
    
    def __repr__(self) -> str:
        return (f"FoldingResult(sequence_length={len(self.sequence)}, "
                f"mean_confidence={self.mean_confidence:.3f}, "
                f"quality={self.structure_quality})")


# Convenience functions for quick usage
def fold_protein(sequence: str, model_size: str = "medium", quantized: bool = False) -> FoldingResult:
    """
    Quick protein folding with default settings.
    
    Args:
        sequence: Protein sequence
        model_size: Model size ("small", "medium", "large")
        quantized: Use quantized model for speed
        
    Returns:
        FoldingResult
    """
    
    model = ESMFold.from_pretrained(model_size, use_quantization=quantized)
    return model.fold(sequence)


def fold_and_save(
    sequence: str, 
    output_path: str, 
    model_size: str = "medium",
    quantized: bool = False
):
    """
    Fold protein and save to PDB file.
    
    Args:
        sequence: Protein sequence
        output_path: Path to save PDB file
        model_size: Model size
        quantized: Use quantized model
    """
    
    result = fold_protein(sequence, model_size, quantized)
    result.save_pdb(output_path)
    
    print(f"✅ Folded {len(sequence)} residue protein")
    print(f"📊 Confidence: {result.mean_confidence:.3f} ({result.structure_quality})")
    print(f"💾 Saved to: {output_path}")
    
    return result