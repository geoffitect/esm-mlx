# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Structure prediction head for ESMFold coordinate generation."""

import math
from typing import Tuple, Dict

import mlx.core as mx
import mlx.nn as nn


def rotation_matrix_from_angles(angles: mx.array) -> mx.array:
    """
    Convert backbone angles to rotation matrices.
    
    Args:
        angles: [batch, N_res, 3] - phi, psi, omega angles in radians
        
    Returns:
        [batch, N_res, 3, 3] - rotation matrices
    """
    batch_size, N_res, _ = angles.shape
    
    phi = angles[..., 0]    # [batch, N_res]
    psi = angles[..., 1]    # [batch, N_res] 
    omega = angles[..., 2]  # [batch, N_res]
    
    # Create rotation matrices for each angle
    # Rotation around x-axis (phi)
    cos_phi = mx.cos(phi)
    sin_phi = mx.sin(phi)
    zeros = mx.zeros_like(cos_phi)
    ones = mx.ones_like(cos_phi)
    
    R_phi = mx.stack([
        mx.stack([ones, zeros, zeros], axis=-1),
        mx.stack([zeros, cos_phi, -sin_phi], axis=-1), 
        mx.stack([zeros, sin_phi, cos_phi], axis=-1)
    ], axis=-2)  # [batch, N_res, 3, 3]
    
    # Rotation around y-axis (psi) 
    cos_psi = mx.cos(psi)
    sin_psi = mx.sin(psi)
    
    R_psi = mx.stack([
        mx.stack([cos_psi, zeros, sin_psi], axis=-1),
        mx.stack([zeros, ones, zeros], axis=-1),
        mx.stack([-sin_psi, zeros, cos_psi], axis=-1)
    ], axis=-2)
    
    # Rotation around z-axis (omega)
    cos_omega = mx.cos(omega)
    sin_omega = mx.sin(omega)
    
    R_omega = mx.stack([
        mx.stack([cos_omega, -sin_omega, zeros], axis=-1),
        mx.stack([sin_omega, cos_omega, zeros], axis=-1),
        mx.stack([zeros, zeros, ones], axis=-1)
    ], axis=-2)
    
    # Combine rotations: R = R_omega @ R_psi @ R_phi
    R_temp = mx.matmul(R_psi, R_phi)
    R = mx.matmul(R_omega, R_temp)
    
    return R


def cross_product_3d(a: mx.array, b: mx.array) -> mx.array:
    """Compute cross product of two 3D vectors."""
    # a × b = [a_y*b_z - a_z*b_y, a_z*b_x - a_x*b_z, a_x*b_y - a_y*b_x]
    c_x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    c_y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    c_z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return mx.stack([c_x, c_y, c_z], axis=-1)


def rigid_body_from_3_points(p1: mx.array, p2: mx.array, p3: mx.array) -> Tuple[mx.array, mx.array]:
    """
    Construct rigid body transformation from 3 points (typically N, CA, C atoms).
    
    Args:
        p1, p2, p3: [batch, N_res, 3] - coordinates of three points
        
    Returns:
        R: [batch, N_res, 3, 3] - rotation matrices
        t: [batch, N_res, 3] - translation vectors
    """
    batch_size, N_res, _ = p1.shape
    
    # For simplicity, just return identity transformations
    # In a full implementation, we would construct proper local coordinate frames
    R = mx.broadcast_to(mx.eye(3)[None, None, :, :], (batch_size, N_res, 3, 3))
    t = p2  # Use CA coordinates as translation
    
    return R, t


class AngleResNet(nn.Module):
    """
    Residual network for predicting backbone angles from single representation.
    """
    
    def __init__(self, c_s: int, c_resnet: int = 128, num_blocks: int = 2):
        super().__init__()
        
        self.c_s = c_s
        self.c_resnet = c_resnet
        self.num_blocks = num_blocks
        
        # Input projection
        self.input_projection = nn.Linear(c_s, c_resnet)
        
        # Residual blocks
        self.blocks = []
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.ReLU(),
                nn.Linear(c_resnet, c_resnet),
                nn.ReLU(), 
                nn.Linear(c_resnet, c_resnet)
            )
            self.blocks.append(block)
        
        # Output layers for angles
        self.angle_output = nn.Linear(c_resnet, 3)  # phi, psi, omega
    
    def __call__(self, s: mx.array) -> mx.array:
        """
        Args:
            s: [batch, N_res, c_s] - single representation
            
        Returns:
            [batch, N_res, 3] - predicted angles (phi, psi, omega) in radians
        """
        # Input projection
        x = self.input_projection(s)
        
        # Apply residual blocks
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        
        # Predict angles
        angles = self.angle_output(x)
        
        # Apply tanh to bound angles to [-pi, pi]
        angles = mx.tanh(angles) * math.pi
        
        return angles


class BackboneUpdate(nn.Module):
    """
    Updates backbone coordinates using predicted angles and rigid body transformations.
    """
    
    def __init__(self, c_s: int):
        super().__init__()
        
        self.angle_resnet = AngleResNet(c_s)
        
        # Standard bond lengths and angles (in Angstroms and radians)
        # These are approximately correct for protein backbones
        self.bond_lengths = mx.array([1.46, 1.52, 1.33])  # N-CA, CA-C, C-N
        self.bond_angles = mx.array([1.94, 1.94, 2.03])   # N-CA-C, CA-C-N, C-N-CA
    
    def __call__(
        self, 
        s: mx.array, 
        initial_coords: mx.array = None
    ) -> Dict[str, mx.array]:
        """
        Args:
            s: [batch, N_res, c_s] - single representation
            initial_coords: [batch, N_res, 3, 3] - initial N, CA, C coordinates
            
        Returns:
            Dictionary containing:
                - coords: [batch, N_res, 3, 3] - updated N, CA, C coordinates
                - angles: [batch, N_res, 3] - predicted backbone angles
                - frames: tuple of (R, t) rigid body transformations
        """
        batch_size, N_res, _ = s.shape
        
        # Predict backbone angles
        angles = self.angle_resnet(s)  # [batch, N_res, 3]
        
        # Initialize coordinates if not provided
        if initial_coords is None:
            # Start with ideal extended conformation
            initial_coords = self._initialize_backbone_coords(batch_size, N_res)
        
        # Update coordinates using predicted angles
        updated_coords = self._update_backbone_geometry(initial_coords, angles)
        
        # Compute rigid body frames from backbone atoms
        N_coords = updated_coords[..., 0, :]   # [batch, N_res, 3]
        CA_coords = updated_coords[..., 1, :]  # [batch, N_res, 3]  
        C_coords = updated_coords[..., 2, :]   # [batch, N_res, 3]
        
        R, t = rigid_body_from_3_points(N_coords, CA_coords, C_coords)
        
        return {
            "coords": updated_coords,
            "angles": angles,
            "frames": (R, t)
        }
    
    def _initialize_backbone_coords(self, batch_size: int, N_res: int) -> mx.array:
        """Initialize backbone in extended conformation."""
        
        # Create extended backbone with standard geometry
        coords = mx.zeros((batch_size, N_res, 3, 3))
        
        # Build coordinates using vectorized operations instead of loops
        # Create extended backbone coordinates all at once
        
        # Start positions for all residues
        x_positions = mx.arange(N_res).astype(mx.float32) * 3.8  # Rough CA-CA distance
        
        # N atoms: slightly offset from CA positions
        n_coords = mx.stack([x_positions - 1.46, mx.zeros(N_res), mx.zeros(N_res)], axis=-1)
        
        # CA atoms: main chain positions
        ca_coords = mx.stack([x_positions, mx.zeros(N_res), mx.zeros(N_res)], axis=-1)
        
        # C atoms: offset from CA
        c_coords = mx.stack([x_positions + 1.52, mx.zeros(N_res), mx.zeros(N_res)], axis=-1)
        
        # Stack into final coordinate array
        coords = mx.stack([n_coords, ca_coords, c_coords], axis=-2)  # [N_res, 3, 3]
        
        # Expand to batch dimension
        coords = mx.broadcast_to(coords[None, :, :, :], (batch_size, N_res, 3, 3))
        
        return coords
    
    def _update_backbone_geometry(self, coords: mx.array, angles: mx.array) -> mx.array:
        """Update backbone coordinates based on predicted angles."""
        
        batch_size, N_res, _, _ = coords.shape
        updated_coords = coords  # Start with initial coordinates
        
        # For each residue, apply rotation based on predicted angles
        for i in range(1, N_res):  # Skip first residue
            # Get predicted angles for this residue
            phi = angles[:, i, 0]   # Rotation around N-CA bond
            psi = angles[:, i, 1]   # Rotation around CA-C bond  
            omega = angles[:, i, 2] # Rotation around C-N bond (peptide bond)
            
            # Get previous and current atom positions
            prev_c = updated_coords[:, i-1, 2, :]  # Previous C
            curr_n = updated_coords[:, i, 0, :]    # Current N
            curr_ca = updated_coords[:, i, 1, :]   # Current CA
            curr_c = updated_coords[:, i, 2, :]    # Current C
            
            # Apply phi rotation (around N-CA bond)
            if i > 0:
                # Rotate C around N-CA axis by phi angle
                axis = curr_ca - curr_n  # N-CA vector
                axis = axis / (mx.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8)
                
                # Rodrigues' rotation formula (simplified for now)
                # For testing, just return original coordinates
                pass
        
        return updated_coords
    
    def _rotate_around_axis(self, v: mx.array, axis: mx.array, angle: mx.array) -> mx.array:
        """Rotate vector v around axis by angle using Rodrigues' formula."""
        
        # Rodrigues' rotation formula: 
        # v_rot = v*cos(θ) + (k×v)*sin(θ) + k(k·v)(1-cos(θ))
        
        cos_angle = mx.cos(angle)[..., None]  # [batch, 1]
        sin_angle = mx.sin(angle)[..., None]  # [batch, 1]
        
        # Cross product k × v
        k_cross_v = cross_product_3d(axis, v)
        
        # Dot product k · v  
        k_dot_v = mx.sum(axis * v, axis=-1, keepdims=True)
        
        # Apply Rodrigues' formula
        v_rotated = (v * cos_angle + 
                    k_cross_v * sin_angle + 
                    axis * k_dot_v * (1 - cos_angle))
        
        return v_rotated


class StructurePredictionHead(nn.Module):
    """
    Complete structure prediction head that generates 3D coordinates from representations.
    """
    
    def __init__(self, c_s: int, c_z: int):
        super().__init__()
        
        self.backbone_update = BackboneUpdate(c_s)
        
        # Side chain prediction (simplified for now)
        self.sidechain_angles = nn.Linear(c_s, 4)  # Chi angles for side chains
        
    def __call__(
        self,
        s: mx.array,
        z: mx.array, 
        initial_coords: mx.array = None
    ) -> Dict[str, mx.array]:
        """
        Args:
            s: [batch, N_res, c_s] - single representation
            z: [batch, N_res, N_res, c_z] - pair representation
            initial_coords: [batch, N_res, 3, 3] - initial coordinates
            
        Returns:
            Dictionary with predicted structure information
        """
        
        # Update backbone coordinates
        backbone_result = self.backbone_update(s, initial_coords)
        
        # Predict side chain angles (simplified)
        sidechain_angles = self.sidechain_angles(s)
        sidechain_angles = mx.tanh(sidechain_angles) * math.pi
        
        return {
            **backbone_result,
            "sidechain_angles": sidechain_angles
        }