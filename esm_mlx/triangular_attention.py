# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triangular attention blocks for ESMFold structure prediction."""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class TriangularSelfAttentionRowWise(nn.Module):
    """
    Triangular self-attention applied row-wise to the pair representation.
    
    For each residue i, applies self-attention over all j such that we attend
    to the interactions between residue i and all other residues j.
    """
    
    def __init__(self, c_z: int, c_hidden: int, num_heads: int, inf: float = 1e9):
        super().__init__()
        
        self.c_z = c_z
        self.c_hidden = c_hidden  
        self.num_heads = num_heads
        self.inf = inf
        
        self.layer_norm = nn.LayerNorm(c_z)
        
        # Linear projections for Q, K, V
        self.linear_q = nn.Linear(c_z, c_hidden * num_heads)
        self.linear_k = nn.Linear(c_z, c_hidden * num_heads) 
        self.linear_v = nn.Linear(c_z, c_hidden * num_heads)
        self.linear_g = nn.Linear(c_z, c_hidden * num_heads)  # Gating
        
        # Output projection
        self.linear_out = nn.Linear(c_hidden * num_heads, c_z)
        
        # Attention scaling
        self.scale = 1.0 / math.sqrt(c_hidden)
    
    def __call__(self, z: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            z: [batch, N_res, N_res, c_z] - pair representation
            mask: [batch, N_res] - residue mask
            
        Returns:
            [batch, N_res, N_res, c_z] - updated pair representation
        """
        batch_size, N_res, _, c_z = z.shape
        
        # Apply layer norm
        z_norm = self.layer_norm(z)
        
        # Linear projections
        q = self.linear_q(z_norm)  # [batch, N_res, N_res, c_hidden * num_heads]
        k = self.linear_k(z_norm)
        v = self.linear_v(z_norm)
        g = self.linear_g(z_norm)  # Gating values
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, N_res, N_res, self.num_heads, self.c_hidden)
        k = k.reshape(batch_size, N_res, N_res, self.num_heads, self.c_hidden)
        v = v.reshape(batch_size, N_res, N_res, self.num_heads, self.c_hidden)
        g = g.reshape(batch_size, N_res, N_res, self.num_heads, self.c_hidden)
        
        # For row-wise attention, we treat each row (i, :) independently
        # Transpose to put heads first for efficient computation
        q = q.transpose(0, 3, 1, 2, 4)  # [batch, num_heads, N_res, N_res, c_hidden]
        k = k.transpose(0, 3, 1, 2, 4)
        v = v.transpose(0, 3, 1, 2, 4)
        g = g.transpose(0, 3, 1, 2, 4)
        
        # Simplify for row-wise attention: treat as standard attention over the row dimension
        # Reshape to [batch * N_res, num_heads, N_res, c_hidden] to process each row independently
        batch_size, N_res, _, num_heads, c_hidden = q.shape
        
        q_flat = q.reshape(batch_size * N_res, num_heads, N_res, c_hidden)
        k_flat = k.reshape(batch_size * N_res, num_heads, N_res, c_hidden)
        v_flat = v.reshape(batch_size * N_res, num_heads, N_res, c_hidden)
        g_flat = g.reshape(batch_size * N_res, num_heads, N_res, c_hidden)
        
        # Standard attention computation
        attention_scores = mx.matmul(q_flat, k_flat.transpose(0, 1, 3, 2)) * self.scale
        # Result: [batch * N_res, num_heads, N_res, N_res]
        
        # Apply mask if provided (simplified)
        if mask is not None:
            # Create simple mask
            mask_flat = mx.repeat(mask.reshape(-1, 1, 1, N_res), num_heads, axis=1)
            attention_scores = mx.where(mask_flat, attention_scores, -self.inf)
        
        # Apply softmax
        attention_weights = mx.softmax(attention_scores, axis=-1)
        
        # Apply attention to values
        attended_values = mx.matmul(attention_weights, v_flat)
        # Result: [batch * N_res, num_heads, N_res, c_hidden]
        
        # Apply gating
        attended_values = attended_values * mx.sigmoid(g_flat)
        
        # Reshape back to original dimensions
        attended_values = attended_values.reshape(batch_size, N_res, num_heads, N_res, c_hidden)
        attended_values = attended_values.transpose(0, 1, 3, 2, 4)  # [batch, N_res, N_res, num_heads, c_hidden]
        attended_values = attended_values.reshape(batch_size, N_res, N_res, -1)  # [batch, N_res, N_res, num_heads * c_hidden]
        
        # Final output projection
        output = self.linear_out(attended_values)
        
        return output


class TriangularSelfAttentionColWise(nn.Module):
    """
    Triangular self-attention applied column-wise to the pair representation.
    
    Similar to row-wise but operates on columns instead of rows.
    """
    
    def __init__(self, c_z: int, c_hidden: int, num_heads: int, inf: float = 1e9):
        super().__init__()
        
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads  
        self.inf = inf
        
        self.layer_norm = nn.LayerNorm(c_z)
        
        # Linear projections
        self.linear_q = nn.Linear(c_z, c_hidden * num_heads)
        self.linear_k = nn.Linear(c_z, c_hidden * num_heads)
        self.linear_v = nn.Linear(c_z, c_hidden * num_heads)
        self.linear_g = nn.Linear(c_z, c_hidden * num_heads)
        
        self.linear_out = nn.Linear(c_hidden * num_heads, c_z)
        
        self.scale = 1.0 / math.sqrt(c_hidden)
    
    def __call__(self, z: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            z: [batch, N_res, N_res, c_z] - pair representation
            mask: [batch, N_res] - residue mask
            
        Returns:
            [batch, N_res, N_res, c_z] - updated pair representation
        """
        # For column-wise attention, we transpose the spatial dimensions
        # So column operations become row operations
        z_transposed = z.transpose(0, 2, 1, 3)  # [batch, N_res, N_res, c_z] -> [batch, N_res, N_res, c_z]
        
        batch_size, N_res, _, c_z = z_transposed.shape
        
        # Apply layer norm
        z_norm = self.layer_norm(z_transposed)
        
        # Linear projections (same as row-wise)
        q = self.linear_q(z_norm)
        k = self.linear_k(z_norm)
        v = self.linear_v(z_norm)
        g = self.linear_g(z_norm)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, N_res, N_res, self.num_heads, self.c_hidden)
        k = k.reshape(batch_size, N_res, N_res, self.num_heads, self.c_hidden)
        v = v.reshape(batch_size, N_res, N_res, self.num_heads, self.c_hidden)
        g = g.reshape(batch_size, N_res, N_res, self.num_heads, self.c_hidden)
        
        # Transpose for efficient computation
        q = q.transpose(0, 3, 1, 2, 4)
        k = k.transpose(0, 3, 1, 2, 4)
        v = v.transpose(0, 3, 1, 2, 4)
        g = g.transpose(0, 3, 1, 2, 4)
        
        # Compute attention (same as row-wise)
        attention_scores = mx.matmul(q, k.transpose(0, 1, 2, 4, 3)) * self.scale
        
        # Apply mask
        if mask is not None:
            mask_2d = mask[:, None, :, None] * mask[:, None, None, :]
            mask_expanded = mask_2d[:, None, :, :, None]
            attention_scores = mx.where(mask_expanded, attention_scores, -self.inf)
        
        # Softmax and attention
        attention_weights = mx.softmax(attention_scores, axis=-1)
        attended_values = mx.matmul(attention_weights, v)
        
        # Apply gating
        attended_values = attended_values * mx.sigmoid(g)
        
        # Reshape and output projection
        attended_values = attended_values.transpose(0, 2, 3, 1, 4)
        attended_values = attended_values.reshape(batch_size, N_res, N_res, -1)
        output = self.linear_out(attended_values)
        
        # Transpose back to original orientation
        output = output.transpose(0, 2, 1, 3)
        
        return output


class TriangularSelfAttentionBlock(nn.Module):
    """
    Complete triangular self-attention block combining row-wise and column-wise attention
    with residual connections and layer normalization.
    """
    
    def __init__(self, c_z: int, c_hidden: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.row_attention = TriangularSelfAttentionRowWise(c_z, c_hidden, num_heads)
        self.col_attention = TriangularSelfAttentionColWise(c_z, c_hidden, num_heads)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Optional: additional feed-forward layer
        self.ffn = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z * 4),
            nn.ReLU(),
            nn.Linear(c_z * 4, c_z),
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
        # Row-wise attention with residual
        z = z + self.dropout(self.row_attention(z, mask))
        
        # Column-wise attention with residual  
        z = z + self.dropout(self.col_attention(z, mask))
        
        # Feed-forward with residual
        z = z + self.ffn(z)
        
        return z


class TriangularMultiplicativeUpdate(nn.Module):
    """
    Triangular multiplicative update for pair representations.
    
    Updates z_ij based on interactions through intermediate residue k:
    z_ij += sum_k (left_projection(z_ik) * right_projection(z_kj))
    """
    
    def __init__(self, c_z: int, c_hidden: int):
        super().__init__()
        
        self.c_z = c_z
        self.c_hidden = c_hidden
        
        self.layer_norm = nn.LayerNorm(c_z)
        self.left_projection = nn.Linear(c_z, c_hidden)
        self.right_projection = nn.Linear(c_z, c_hidden) 
        self.center_projection = nn.Linear(c_z, c_hidden)
        self.output_projection = nn.Linear(c_hidden, c_z)
        self.gate = nn.Linear(c_z, c_z)
    
    def __call__(self, z: mx.array) -> mx.array:
        """
        Args:
            z: [batch, N_res, N_res, c_z] - pair representation
            
        Returns:
            [batch, N_res, N_res, c_z] - updated pair representation
        """
        batch_size, N_res, _, c_z = z.shape
        
        # Apply layer normalization
        z_norm = self.layer_norm(z)
        
        # Projections
        left = self.left_projection(z_norm)    # [batch, N_res, N_res, c_hidden]
        right = self.right_projection(z_norm)  # [batch, N_res, N_res, c_hidden]
        center = self.center_projection(z_norm) # [batch, N_res, N_res, c_hidden]
        
        # Triangular update: z_ij += sum_k (left_ik * right_kj)
        # left: [batch, N_res, N_res, c_hidden] -> [batch, N_res, k, c_hidden]
        # right: [batch, N_res, N_res, c_hidden] -> [batch, k, N_res, c_hidden]
        # Want: [batch, N_res, N_res, c_hidden]
        
        # Reshape for matrix multiplication
        # left[i,k] * right[k,j] -> result[i,j]
        left_reshaped = left.transpose(0, 1, 3, 2)  # [batch, N_res, c_hidden, N_res]
        right_reshaped = right.transpose(0, 2, 3, 1)  # [batch, N_res, c_hidden, N_res]
        
        # Matrix multiplication over the k dimension
        update = mx.matmul(left_reshaped, right_reshaped)  # [batch, N_res, c_hidden, N_res]
        update = update.transpose(0, 1, 3, 2)  # [batch, N_res, N_res, c_hidden]
        
        # Add center bias
        update = update + center
        
        # Output projection
        update = self.output_projection(update)
        
        # Gating
        gate_values = mx.sigmoid(self.gate(z_norm))
        update = update * gate_values
        
        return update