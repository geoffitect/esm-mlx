# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Invariant Point Attention (IPA) implementation for ESMFold structure prediction."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class InvariantPointAttention(nn.Module):
    """
    Invariant Point Attention layer for protein structure prediction.
    
    Operates on both scalar representations and 3D point clouds, maintaining
    SE(3) equivariance for geometric reasoning.
    """
    
    def __init__(
        self,
        c_s: int,  # Single representation dimension
        c_z: int,  # Pair representation dimension  
        c_hidden: int,  # Hidden dimension
        num_heads: int = 12,
        num_scalar_qk: int = 16,
        num_scalar_v: int = 16,
        num_point_qk: int = 4,
        num_point_v: int = 8,
        inf: float = 1e5,
        eps: float = 1e-8
    ):
        super().__init__()
        
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.inf = inf
        self.eps = eps
        
        # Scalar attention projections
        hc = self.num_heads * self.c_hidden
        self.linear_q_points = nn.Linear(c_s, num_heads * num_point_qk * 3)
        self.linear_k_points = nn.Linear(c_s, num_heads * num_point_qk * 3)
        self.linear_v_points = nn.Linear(c_s, num_heads * num_point_v * 3)
        
        self.linear_q = nn.Linear(c_s, num_heads * num_scalar_qk)
        self.linear_k = nn.Linear(c_s, num_heads * num_scalar_qk)
        self.linear_v = nn.Linear(c_s, num_heads * num_scalar_v)
        self.linear_b = nn.Linear(c_z, num_heads)
        
        self.head_weights = mx.ones((num_heads,))
        
        # Point attention weights - calculate the actual output dimension
        concat_out_dim = (
            num_heads * num_scalar_v +    # scalar outputs
            num_heads * num_point_v +     # point norms  
            num_heads * num_point_v * 3 + # point coordinates
            c_z                           # pair representation
        )
        self.linear_out = nn.Linear(concat_out_dim, c_s)
        
        self.softmax = nn.Softmax()
        self.softplus = nn.Softplus()
    
    def __call__(
        self,
        s: mx.array,  # [batch, N_res, c_s] - single representation
        z: mx.array,  # [batch, N_res, N_res, c_z] - pair representation
        r: mx.array,  # [batch, N_res, 3, 3] - rotation matrices
        t: mx.array,  # [batch, N_res, 3] - translation vectors
        mask: Optional[mx.array] = None  # [batch, N_res]
    ) -> mx.array:
        
        batch_size, N_res, _ = s.shape
        
        # Generate scalar queries, keys, values
        q_scalar = self.linear_q(s)  # [batch, N_res, num_heads * num_scalar_qk]
        k_scalar = self.linear_k(s)  # [batch, N_res, num_heads * num_scalar_qk]
        v_scalar = self.linear_v(s)  # [batch, N_res, num_heads * num_scalar_v]
        
        # Reshape for multi-head attention
        q_scalar = q_scalar.reshape(batch_size, N_res, self.num_heads, self.num_scalar_qk)
        k_scalar = k_scalar.reshape(batch_size, N_res, self.num_heads, self.num_scalar_qk)
        v_scalar = v_scalar.reshape(batch_size, N_res, self.num_heads, self.num_scalar_v)
        
        # Generate point queries, keys, values
        q_points = self.linear_q_points(s)  # [batch, N_res, num_heads * num_point_qk * 3]
        k_points = self.linear_k_points(s)  # [batch, N_res, num_heads * num_point_qk * 3]
        v_points = self.linear_v_points(s)  # [batch, N_res, num_heads * num_point_v * 3]
        
        # Reshape point queries/keys/values
        q_points = q_points.reshape(batch_size, N_res, self.num_heads, self.num_point_qk, 3)
        k_points = k_points.reshape(batch_size, N_res, self.num_heads, self.num_point_qk, 3)
        v_points = v_points.reshape(batch_size, N_res, self.num_heads, self.num_point_v, 3)
        
        # Apply rigid transformations to points
        # q_points: [batch, N_res, num_heads, num_point_qk, 3]
        # r: [batch, N_res, 3, 3], t: [batch, N_res, 3]
        
        # Transform query points: R @ q + t
        q_points_global = mx.matmul(
            r[:, :, None, None, :, :],  # [batch, N_res, 1, 1, 3, 3]
            q_points[..., None]         # [batch, N_res, num_heads, num_point_qk, 3, 1]
        ).squeeze(-1)  # [batch, N_res, num_heads, num_point_qk, 3]
        q_points_global = q_points_global + t[:, :, None, None, :]
        
        # Transform key points: R @ k + t
        k_points_global = mx.matmul(
            r[:, :, None, None, :, :],  # [batch, N_res, 1, 1, 3, 3]
            k_points[..., None]         # [batch, N_res, num_heads, num_point_qk, 3, 1]
        ).squeeze(-1)  # [batch, N_res, num_heads, num_point_qk, 3]
        k_points_global = k_points_global + t[:, :, None, None, :]
        
        # Transform value points: R @ v + t
        v_points_global = mx.matmul(
            r[:, :, None, None, :, :],  # [batch, N_res, 1, 1, 3, 3]
            v_points[..., None]         # [batch, N_res, num_heads, num_point_v, 3, 1]
        ).squeeze(-1)  # [batch, N_res, num_heads, num_point_v, 3]
        v_points_global = v_points_global + t[:, :, None, None, :]
        
        # Compute attention weights
        # Scalar attention: q @ k^T
        # Need to compute attention between all pairs of residues
        # q_scalar: [batch, N_res, num_heads, num_scalar_qk]
        # k_scalar: [batch, N_res, num_heads, num_scalar_qk]
        # Want result: [batch, N_res, num_heads, N_res]
        
        # Transpose to [batch, num_heads, N_res, num_scalar_qk] for efficient computation
        q_scalar_t = q_scalar.transpose(0, 2, 1, 3)  # [batch, num_heads, N_res, num_scalar_qk]
        k_scalar_t = k_scalar.transpose(0, 2, 1, 3)  # [batch, num_heads, N_res, num_scalar_qk]
        
        # Compute attention: [batch, num_heads, N_res, num_scalar_qk] @ [batch, num_heads, num_scalar_qk, N_res]
        scalar_attention = mx.matmul(q_scalar_t, k_scalar_t.transpose(0, 1, 3, 2))
        # Result: [batch, num_heads, N_res, N_res]
        
        # Transpose back to [batch, N_res, num_heads, N_res]
        scalar_attention = scalar_attention.transpose(0, 2, 1, 3)
        
        # Point attention: sum over point pairs of squared distances
        # q_points_global: [batch, N_res, num_heads, num_point_qk, 3]
        # k_points_global: [batch, N_res, num_heads, num_point_qk, 3]
        
        # Compute pairwise squared distances
        q_expanded = q_points_global[:, :, None, :, :, :]  # [batch, N_res, 1, num_heads, num_point_qk, 3]
        k_expanded = k_points_global[:, None, :, :, :, :]  # [batch, 1, N_res, num_heads, num_point_qk, 3]
        
        point_diffs = q_expanded - k_expanded  # [batch, N_res, N_res, num_heads, num_point_qk, 3]
        point_distances_sq = mx.sum(point_diffs ** 2, axis=-1)  # [batch, N_res, N_res, num_heads, num_point_qk]
        point_attention = -0.5 * mx.sum(point_distances_sq, axis=-1)  # [batch, N_res, N_res, num_heads]
        point_attention = point_attention.transpose(0, 1, 3, 2)  # [batch, N_res, num_heads, N_res]
        
        # Pair bias from z
        pair_bias = self.linear_b(z)  # [batch, N_res, N_res, num_heads]
        pair_bias = pair_bias.transpose(0, 1, 3, 2)  # [batch, N_res, num_heads, N_res]
        
        # Combine attention components
        attention_scores = scalar_attention + point_attention + pair_bias
        
        # Apply mask
        if mask is not None:
            mask_2d = mask[:, :, None] * mask[:, None, :]  # [batch, N_res, N_res]
            mask_expanded = mask_2d[:, :, None, :]  # [batch, N_res, 1, N_res]
            attention_scores = mx.where(mask_expanded, attention_scores, -self.inf)
        
        # Apply softmax
        attention_weights = self.softmax(attention_scores)  # [batch, N_res, num_heads, N_res]
        
        # Apply attention to scalar values
        # attention_weights: [batch, N_res, num_heads, N_res]
        # v_scalar: [batch, N_res, num_heads, num_scalar_v]
        
        # Transpose for efficient computation: [batch, num_heads, N_res, N_res] @ [batch, num_heads, N_res, num_scalar_v]
        attention_weights_t = attention_weights.transpose(0, 2, 1, 3)  # [batch, num_heads, N_res, N_res]
        v_scalar_t = v_scalar.transpose(0, 2, 1, 3)  # [batch, num_heads, N_res, num_scalar_v]
        
        scalar_out = mx.matmul(attention_weights_t, v_scalar_t)  # [batch, num_heads, N_res, num_scalar_v]
        scalar_out = scalar_out.transpose(0, 2, 1, 3)  # [batch, N_res, num_heads, num_scalar_v]
        
        # Apply attention to point values more efficiently
        # attention_weights: [batch, N_res, num_heads, N_res]
        # v_points_global: [batch, N_res, num_heads, num_point_v, 3]
        
        # Reshape for efficient computation
        # attention_weights: [batch, N_res, num_heads, N_res] -> [batch, N_res, num_heads, N_res, 1, 1]
        attn_expanded = attention_weights[..., None, None]
        
        # v_points_global: [batch, N_res, num_heads, num_point_v, 3] -> [batch, 1, num_heads, N_res, num_point_v, 3]
        v_points_expanded = v_points_global[:, None, :, :, :, :].transpose(0, 1, 3, 2, 4, 5)
        
        # Apply attention weights: [batch, N_res, num_heads, N_res, 1, 1] * [batch, 1, num_heads, N_res, num_point_v, 3]
        weighted_points = attn_expanded * v_points_expanded
        
        # Sum over the key dimension (axis=3): [batch, N_res, num_heads, num_point_v, 3]
        point_out = mx.sum(weighted_points, axis=3)
        
        # Transform point outputs back to local coordinates
        # point_out: [batch, N_res, num_heads, num_point_v, 3]
        # Need to apply R^T @ (point_out - t)
        point_out_local = point_out - t[:, :, None, None, :]
        point_out_local = mx.matmul(
            r.transpose(0, 1, 3, 2)[:, :, None, None, :, :],  # [batch, N_res, 1, 1, 3, 3]
            point_out_local[..., None]  # [batch, N_res, num_heads, num_point_v, 3, 1]
        ).squeeze(-1)  # [batch, N_res, num_heads, num_point_v, 3]
        
        # Compute point features (norms and cross products)
        point_norms = mx.linalg.norm(point_out_local, axis=-1)  # [batch, N_res, num_heads, num_point_v]
        
        # Concatenate all outputs
        # Reshape for concatenation
        scalar_out_flat = scalar_out.reshape(batch_size, N_res, -1)  # [batch, N_res, num_heads * num_scalar_v]
        point_norms_flat = point_norms.reshape(batch_size, N_res, -1)  # [batch, N_res, num_heads * num_point_v]
        point_out_flat = point_out_local.reshape(batch_size, N_res, -1)  # [batch, N_res, num_heads * num_point_v * 3]
        
        # For pair representation, we need to reduce it to single representation somehow
        # Take mean over the second residue dimension as a simple approach
        pair_out_flat = mx.mean(z, axis=2)  # [batch, N_res, c_z]
        
        # Concatenate all features
        combined_features = mx.concatenate([
            scalar_out_flat,
            point_norms_flat,
            point_out_flat,
            pair_out_flat
        ], axis=-1)
        
        # Final linear projection
        output = self.linear_out(combined_features)  # [batch, N_res, c_s]
        
        return output


class IPABlock(nn.Module):
    """Complete IPA block with residual connections and layer norm."""
    
    def __init__(self, config):
        super().__init__()
        
        self.ipa = InvariantPointAttention(
            c_s=config.c_s,
            c_z=config.c_z,
            c_hidden=config.ipa_hidden_dim,
            num_heads=config.ipa_num_heads,
            num_scalar_qk=config.ipa_num_scalar_qk,
            num_scalar_v=config.ipa_num_scalar_v,
            num_point_qk=config.ipa_num_point_qk,
            num_point_v=config.ipa_num_point_v,
        )
        
        self.layer_norm = nn.LayerNorm(config.c_s)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def __call__(self, s, z, r, t, mask=None):
        # Pre-layer norm
        s_norm = self.layer_norm(s)
        
        # Apply IPA
        ipa_out = self.ipa(s_norm, z, r, t, mask)
        
        # Residual connection with dropout
        s = s + self.dropout(ipa_out)
        
        return s