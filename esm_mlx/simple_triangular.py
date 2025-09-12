# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simplified triangular attention for testing."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class SimpleTriangularAttention(nn.Module):
    """Simplified triangular attention for initial testing."""
    
    def __init__(self, c_z: int, c_hidden: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.c_z = c_z
        self.layer_norm = nn.LayerNorm(c_z)
        
        # Simple feed-forward transformation for now
        self.transform = nn.Sequential(
            nn.Linear(c_z, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_z),
            nn.Dropout(dropout_rate)
        )
    
    def __call__(self, z: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            z: [batch, N_res, N_res, c_z] - pair representation
            mask: [batch, N_res] - residue mask
            
        Returns:
            [batch, N_res, N_res, c_z] - updated pair representation
        """
        # Apply layer norm and simple transformation
        z_norm = self.layer_norm(z)
        output = self.transform(z_norm)
        
        return output