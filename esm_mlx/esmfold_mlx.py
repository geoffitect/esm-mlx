# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""MLX implementation of ESMFold for protein structure prediction."""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .esm2_mlx import ESM2MLX
from .config import ESM2Config
from .ipa import IPABlock
from .simple_triangular import SimpleTriangularAttention
from .structure_head import StructurePredictionHead
from .confidence_head import ConfidenceHead


@dataclass
class ESMFoldConfig:
    """Configuration for ESMFold model."""
    
    # ESM backbone config
    esm_config: ESM2Config = None
    
    # Structure module dimensions
    c_s: int = 384           # Single representation dimension
    c_z: int = 128           # Pair representation dimension  
    
    # IPA configuration
    ipa_num_heads: int = 12
    ipa_hidden_dim: int = 16
    ipa_num_scalar_qk: int = 16
    ipa_num_scalar_v: int = 16
    ipa_num_point_qk: int = 4
    ipa_num_point_v: int = 8
    
    # Triangular attention configuration
    tri_attn_hidden: int = 32
    tri_attn_heads: int = 4
    
    # Structure module architecture
    num_folding_blocks: int = 48  # Number of triangular attention blocks
    num_ipa_blocks: int = 8       # Number of IPA blocks per folding block
    num_recycles: int = 4         # Number of recycling iterations
    
    # Dropout and regularization
    dropout_rate: float = 0.1
    
    # Confidence prediction
    lddt_bins: int = 50
    distogram_bins: int = 64


class FoldingTrunk(nn.Module):
    """
    Folding trunk containing triangular attention blocks and IPA layers.
    
    This is the core structure prediction module that processes the ESM representations
    through geometric reasoning layers to predict 3D coordinates.
    """
    
    def __init__(self, config: ESMFoldConfig):
        super().__init__()
        
        self.config = config
        
        # Input projections from ESM to structure representations
        self.single_proj = nn.Linear(config.esm_config.hidden_size, config.c_s)
        
        # Initial pair representation (learned embeddings for relative positions)
        self.pair_embed = nn.Embedding(2048, config.c_z)  # Max relative distance embedding
        
        # Folding blocks (triangular attention + IPA)
        self.folding_blocks = []
        for _ in range(config.num_folding_blocks):
            block = FoldingBlock(config)
            self.folding_blocks.append(block)
        
        # Layer normalization for outputs
        self.single_norm = nn.LayerNorm(config.c_s)
        self.pair_norm = nn.LayerNorm(config.c_z)
    
    def _create_pair_representation(self, batch_size: int, N_res: int) -> mx.array:
        """Create initial pair representation from relative positions."""
        
        # Create relative position matrix
        pos_i = mx.arange(N_res)[:, None]      # [N_res, 1]
        pos_j = mx.arange(N_res)[None, :]      # [1, N_res]
        
        relative_pos = pos_j - pos_i  # [N_res, N_res]
        
        # Clip to embedding range and shift to positive
        max_rel_pos = 1023  # Half of embedding size for symmetry
        relative_pos = mx.clip(relative_pos, -max_rel_pos, max_rel_pos)
        relative_pos = relative_pos + max_rel_pos  # Shift to [0, 2047]
        
        # Expand to batch dimension
        relative_pos = mx.broadcast_to(relative_pos[None, :, :], (batch_size, N_res, N_res))
        
        # Embed relative positions
        pair_repr = self.pair_embed(relative_pos)  # [batch, N_res, N_res, c_z]
        
        return pair_repr
    
    def __call__(
        self, 
        esm_repr: mx.array,
        mask: Optional[mx.array] = None,
        initial_coords: Optional[mx.array] = None
    ) -> Dict[str, Any]:
        """
        Args:
            esm_repr: [batch, N_res, esm_dim] - ESM backbone representations
            mask: [batch, N_res] - sequence mask
            initial_coords: [batch, N_res, 3, 3] - initial backbone coordinates
            
        Returns:
            Dictionary containing structure predictions
        """
        batch_size, N_res, _ = esm_repr.shape
        
        # Project ESM representation to single representation
        single_repr = self.single_proj(esm_repr)  # [batch, N_res, c_s]
        
        # Create initial pair representation
        pair_repr = self._create_pair_representation(batch_size, N_res)
        
        # Initialize rigid body frames (identity if not provided)
        if initial_coords is None:
            # Identity transformations
            frames_r = mx.broadcast_to(
                mx.eye(3)[None, None, :, :], 
                (batch_size, N_res, 3, 3)
            )
            frames_t = mx.zeros((batch_size, N_res, 3))
        else:
            # Compute frames from initial coordinates (simplified)
            frames_r = mx.broadcast_to(
                mx.eye(3)[None, None, :, :], 
                (batch_size, N_res, 3, 3)
            )
            frames_t = initial_coords[..., 1, :]  # Use CA positions as translation
        
        # Process through folding blocks
        for i, block in enumerate(self.folding_blocks):
            single_repr, pair_repr, frames_r, frames_t = block(
                single_repr, pair_repr, frames_r, frames_t, mask
            )
        
        # Apply final layer normalization
        single_repr = self.single_norm(single_repr)
        pair_repr = self.pair_norm(pair_repr)
        
        return {
            "single": single_repr,
            "pair": pair_repr,
            "frames": (frames_r, frames_t)
        }


class FoldingBlock(nn.Module):
    """
    Single folding block containing triangular attention and IPA layers.
    """
    
    def __init__(self, config: ESMFoldConfig):
        super().__init__()
        
        # Simplified triangular attention for pair updates
        self.triangular_attention = SimpleTriangularAttention(
            c_z=config.c_z,
            c_hidden=config.tri_attn_hidden,
            num_heads=config.tri_attn_heads,
            dropout_rate=config.dropout_rate
        )
        
        # IPA blocks for single representation updates
        self.ipa_blocks = []
        for _ in range(config.num_ipa_blocks):
            ipa_block = IPABlock(config)
            self.ipa_blocks.append(ipa_block)
        
        # Backbone update for coordinate refinement
        self.backbone_update = nn.Sequential(
            nn.Linear(config.c_s, config.c_s),
            nn.ReLU(),
            nn.Linear(config.c_s, 6)  # Rigid body update: 3 rotation + 3 translation
        )
        
    def __call__(
        self,
        single: mx.array,
        pair: mx.array,
        frames_r: mx.array,
        frames_t: mx.array,
        mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Args:
            single: [batch, N_res, c_s] - single representation
            pair: [batch, N_res, N_res, c_z] - pair representation  
            frames_r: [batch, N_res, 3, 3] - rotation matrices
            frames_t: [batch, N_res, 3] - translation vectors
            mask: [batch, N_res] - sequence mask
            
        Returns:
            Updated representations and frames
        """
        
        # Update pair representation with simplified triangular attention
        pair = pair + self.triangular_attention(pair, mask)
        
        # Update single representation with IPA blocks
        for ipa_block in self.ipa_blocks:
            single = ipa_block(single, pair, frames_r, frames_t, mask)
        
        # Update backbone coordinates/frames
        backbone_update = self.backbone_update(single)  # [batch, N_res, 6]
        
        # Extract rotation and translation updates
        rot_update = backbone_update[..., :3]  # [batch, N_res, 3] 
        trans_update = backbone_update[..., 3:]  # [batch, N_res, 3]
        
        # Apply updates to frames (simplified - should be proper SE(3) composition)
        frames_t = frames_t + trans_update
        
        # For rotation, apply small angle approximation for updates
        angle_norm = mx.linalg.norm(rot_update, axis=-1, keepdims=True) + 1e-8
        rot_axis = rot_update / angle_norm
        
        # Simple rotation update (should be proper matrix multiplication)
        # frames_r = compose_rotations(frames_r, rot_from_axis_angle(rot_axis, angle_norm))
        # For now, keep frames_r unchanged for simplicity
        
        return single, pair, frames_r, frames_t


class ESMFoldMLX(nn.Module):
    """
    Complete ESMFold model for protein structure prediction using MLX.
    
    Integrates ESM-2 language model backbone with structure prediction modules
    to generate 3D protein coordinates from sequence.
    """
    
    def __init__(self, config: ESMFoldConfig):
        super().__init__()
        
        self.config = config
        
        # ESM-2 backbone for sequence processing
        self.esm = ESM2MLX(config.esm_config)
        
        # Folding trunk for structure prediction  
        self.folding_trunk = FoldingTrunk(config)
        
        # Structure prediction head
        self.structure_head = StructurePredictionHead(config.c_s, config.c_z)
        
        # Confidence prediction head
        self.confidence_head = ConfidenceHead(
            config.c_s, 
            config.c_z,
            config.lddt_bins,
            config.distogram_bins
        )
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        num_recycles: Optional[int] = None,
        return_all_frames: bool = False
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: [batch, N_res] - tokenized protein sequences
            attention_mask: [batch, N_res] - attention mask
            num_recycles: Number of recycling iterations (default from config)
            return_all_frames: Whether to return intermediate frames
            
        Returns:
            Dictionary containing structure predictions and confidence scores
        """
        
        if num_recycles is None:
            num_recycles = self.config.num_recycles
        
        # Get ESM representations
        esm_output = self.esm(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        esm_repr = esm_output["last_hidden_state"]  # [batch, N_res, esm_dim]
        
        # Initialize for recycling
        prev_coords = None
        all_frames = [] if return_all_frames else None
        
        # Recycling loop
        for recycle_idx in range(num_recycles + 1):
            
            # Run folding trunk
            fold_output = self.folding_trunk(
                esm_repr,
                mask=attention_mask, 
                initial_coords=prev_coords
            )
            
            single_repr = fold_output["single"]
            pair_repr = fold_output["pair"]
            frames = fold_output["frames"]
            
            # Predict structure
            structure_output = self.structure_head(
                single_repr,
                pair_repr,
                prev_coords
            )
            
            # Update coordinates for next recycle
            prev_coords = structure_output["coords"]
            
            if return_all_frames:
                all_frames.append({
                    "coords": structure_output["coords"],
                    "frames": frames,
                    "recycle_idx": recycle_idx
                })
        
        # Final structure predictions
        final_coords = structure_output["coords"]
        final_angles = structure_output["angles"]
        
        # Predict confidence scores
        confidence_output = self.confidence_head(
            single_repr,
            pair_repr,
            attention_mask
        )
        
        # Prepare final output
        output = {
            # Structure predictions
            "coordinates": final_coords,           # [batch, N_res, 3, 3] (N, CA, C)
            "backbone_angles": final_angles,       # [batch, N_res, 3] (phi, psi, omega)
            "sidechain_angles": structure_output["sidechain_angles"],
            
            # Confidence scores  
            "plddt": confidence_output["plddt"],              # [batch, N_res]
            "tm_score": confidence_output["tm_score"],        # [batch]
            "calibrated_confidence": confidence_output["calibrated_confidence"],
            
            # Distance predictions
            "predicted_distances": confidence_output["expected_distances"],
            
            # Representations
            "single_repr": single_repr,
            "pair_repr": pair_repr,
            
            # Frames for downstream use
            "frames": frames
        }
        
        if return_all_frames:
            output["all_frames"] = all_frames
        
        return output
    
    @classmethod  
    def from_config(cls, config: ESMFoldConfig):
        """Create ESMFold model from configuration."""
        return cls(config)
    
    def predict_structure(
        self,
        sequences: list[str], 
        tokenizer=None,
        num_recycles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        High-level interface for structure prediction from sequences.
        
        Args:
            sequences: List of protein sequences as strings
            tokenizer: Tokenizer for converting sequences to input_ids
            num_recycles: Number of recycling iterations
            
        Returns:
            Structure predictions with coordinates and confidence scores
        """
        
        # Tokenize sequences (simplified - would need proper tokenizer)
        if tokenizer is None:
            # Simple amino acid to token mapping (placeholder)
            aa_to_token = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
            
            input_ids = []
            attention_masks = []
            
            for seq in sequences:
                tokens = [aa_to_token.get(aa, 0) for aa in seq]
                input_ids.append(tokens)
                attention_masks.append([1] * len(tokens))
            
            # Pad sequences
            max_len = max(len(seq) for seq in input_ids)
            for i in range(len(input_ids)):
                pad_len = max_len - len(input_ids[i])
                input_ids[i].extend([0] * pad_len)
                attention_masks[i].extend([0] * pad_len)
            
            input_ids = mx.array(input_ids)
            attention_mask = mx.array(attention_masks)
        
        # Run prediction
        return self(input_ids, attention_mask, num_recycles)